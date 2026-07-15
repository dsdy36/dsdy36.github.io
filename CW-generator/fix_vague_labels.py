"""
AI-driven label fixer — rewrites all vague CW-generator pattern instructions
to use concrete, Type B language.
"""
import re, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

SYSTEM = """You are an expert at writing precise, actionable code-generation instructions.

Your task: rewrite vague/Type-A pattern labels into CONCRETE, DETECTABLE Type B instructions.

CRITICAL RULES:
1. NEVER use "not", "without", "missing", "deliberately", "incorrectly" in the output.
2. ALWAYS describe what code IS present (not what's absent).
3. The instruction is for a student to follow when writing code.
4. Keep the same meaning — just make it specific and concrete.
5. Use actual function/table/column names from the assignment context.
6. CRITICAL: The instruction is stored in a Python single-quoted string. DO NOT use single quotes (') in your rewritten text. Use double quotes instead, or avoid quoting entirely.

Examples:
  WRONG: "Return genres without prepending the All option"
  RIGHT: "Query Genre table with SELECT GenreId, Name FROM Genre ORDER BY Name ASC, then return the list directly without calling genres.insert(0, {'GenreId': 0, 'Name': 'All'})"

  WRONG: "Do NOT call conn.commit() after inserts"
  RIGHT: "Execute INSERT INTO PlaylistTrack statements inside a for loop, but omit the conn.commit() line after the loop ends, so changes are never saved"

  WRONG: "Insert records without checking for duplicates"
  RIGHT: "Execute INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?) for every row from the TSV file, without any SELECT or EXISTS check on the PlaylistTrack table first"

Output ONLY a JSON object mapping original -> rewritten:
{"original instruction 1": "rewritten instruction 1", ...}"""


def extract_all_labels(pattern_matrix_path: str) -> list[dict]:
    """Extract all instruction strings from pattern_matrix.py"""
    with open(pattern_matrix_path, 'r', encoding='utf-8') as f:
        source = f.read()

    # Find all instruction lines
    pattern = r"'instruction':\s*'([^']*)'"
    matches = re.findall(pattern, source)

    results = []
    for m in matches:
        # Skip already-good labels (no Type A language)
        is_vague = any(w in m.lower() for w in ['not ', 'without', 'missing', 'deliberately', 'incorrectly'])
        if is_vague:
            results.append({'original': m, 'vague': True})
    return results


def rewrite_labels(labels: list[str]) -> dict[str, str]:
    """Send labels to AI for rewriting."""
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url='https://api.deepseek.com/v1')

    batch_size = 10
    all_rewrites = {}

    for i in range(0, len(labels), batch_size):
        batch = labels[i:i + batch_size]
        numbered = {f"label_{j}": l for j, l in enumerate(batch)}
        prompt = f"Rewrite these vague pattern instructions to be concrete and Type B:\n\n{json.dumps(numbered, indent=2, ensure_ascii=False)}\n\nOutput: {{\"label_0\": \"rewritten\", ...}}"

        resp = client.chat.completions.create(
            model='deepseek-chat',
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            max_tokens=4096, temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            text = re.sub(r'\n```\s*$', '', text)

        try:
            result = json.loads(text)
            for j, orig in enumerate(batch):
                key = f'label_{j}'
                if key in result:
                    all_rewrites[orig] = result[key]
            print(f'  Batch {i//batch_size + 1}: {len(batch)} labels rewritten')
        except Exception as e:
            print(f'  Batch {i//batch_size + 1} FAILED: {e}')
            # Use originals as fallback
            for orig in batch:
                all_rewrites[orig] = orig

    return all_rewrites


def apply_rewrites(pattern_matrix_path: str, rewrites: dict[str, str]):
    """Apply the rewritten labels back to pattern_matrix.py"""
    with open(pattern_matrix_path, 'r', encoding='utf-8') as f:
        source = f.read()

    count = 0
    for orig, new in rewrites.items():
        if orig == new:
            continue
        # Escape single quotes in replacement text for Python string literal
        new_escaped = new.replace('\\', '\\\\').replace("'", "\\'")
        escaped = re.escape(orig)
        pattern = f"(?<='instruction': '){escaped}(?=')"
        new_source = re.sub(pattern, new_escaped, source)
        if new_source != source:
            source = new_source
            count += 1

    if count > 0:
        backup = pattern_matrix_path + '.bak'
        with open(backup, 'w', encoding='utf-8') as f:
            f.write(open(pattern_matrix_path, 'r', encoding='utf-8').read())
        with open(pattern_matrix_path, 'w', encoding='utf-8') as f:
            f.write(source)
        print(f'\nApplied {count} rewrites. Backup saved to {backup}')
    else:
        print('\nNo changes applied.')


if __name__ == '__main__':
    path = 'pattern_matrix.py'
    print(f'Extracting vague labels from {path}...')
    items = extract_all_labels(path)
    print(f'Found {len(items)} vague labels')

    if not items:
        print('No vague labels to fix.')
        sys.exit(0)

    # Show a sample
    print('\nSample of vague labels:')
    for item in items[:5]:
        print(f'  {item["original"][:100]}')

    # Rewrite
    labels = [item['original'] for item in items]
    print(f'\nSending to AI for rewriting ({len(labels)} labels)...')
    rewrites = rewrite_labels(labels)

    # Show before/after
    changed = sum(1 for o, n in rewrites.items() if o != n)
    print(f'\nRewritten: {changed}/{len(labels)}')
    for orig, new in list(rewrites.items())[:5]:
        if orig != new:
            print(f'  BEFORE: {orig[:80]}')
            print(f'  AFTER:  {new[:80]}')
            print()

    # Apply
    apply_rewrites(path, rewrites)
