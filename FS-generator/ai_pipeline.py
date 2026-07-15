"""
AI-Driven FS Generation Pipeline (Approach B)
===============================================
Holistic summary + batch generation using DeepSeek API.

Usage:
    python ai_pipeline.py <config.yaml>
"""

import os
import sys
import json
import re
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-chat'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# File helpers
# ============================================================

def read_file(path: str) -> str:
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def collect_submissions(submissions_dir: str, target_file: str,
                        max_students: int | None = 30,
                        student_prefix: str | None = None) -> list[dict]:
    """Recursively collect submissions for a target file.

    Supports TWO formats:
      - CW format (preferred): S001/task1.py, S001/task2.py, S001/task3.py
        Student ID = directory name, target_file = task1.py / task2.py / task3.py
      - V1 format (legacy): q1-excellent-a1/submission/iMusic.py
        Student ID = relative path including subdirectory

    If student_prefix is set, only V1 directories matching that prefix are included.
    For CW format, student_prefix is ignored (all student dirs have all task files).
    """
    results = []
    if not os.path.isdir(submissions_dir):
        return results

    for dirpath, dirnames, filenames in os.walk(submissions_dir):
        if target_file in filenames:
            student_id = os.path.relpath(dirpath, submissions_dir)
            if student_id == '.':
                continue
            # Filter by question prefix (only for V1 nested format)
            if student_prefix and not _matches_prefix(student_id, student_prefix):
                continue
            code = read_file(os.path.join(dirpath, target_file))
            if code.strip():
                results.append({'student': student_id, 'code': code})
            if max_students and len(results) >= max_students:
                break

    # CW format detection: Sxxx directories with taskN.py
    # Sort to ensure consistent ordering
    if not results:
        return results

    # Check if this is CW format (student IDs are just the directory name like 'S001')
    cw_format = all(
        not '/' in r['student'] and not '\\' in r['student']
        for r in results[:5]
    )
    if cw_format:
        results.sort(key=lambda r: r['student'])

    return results


def collect_submissions_by_task(submissions_dir: str, task_num: int,
                                 max_students: int | None = None) -> list[dict]:
    """Collect submissions in CW format: submission/{sid}/task{N}.py.

    Each student directory has task1.py, task2.py, task3.py.
    This function reads the appropriate task file for the given task number.

    Args:
        submissions_dir: Path to submission/ directory.
        task_num: 1, 2, or 3.
        max_students: Max students to return (None = all).

    Returns:
        List of {student: 'S001', code: '...'} dicts.
    """
    target_file = f'task{task_num}.py'
    return collect_submissions(submissions_dir, target_file,
                                max_students=max_students, student_prefix=None)


def extract_task_functions(full_code: str, task_num: int,
                            task_func_map: dict | None = None) -> str:
    """Extract only the functions relevant to a task from a full iMusic.py.

    Used for backward compatibility with V1 format submissions.
    Strips template code (imports, Flask setup, upload_route, etc.) and
    returns only the student-written function bodies for the given task.

    Args:
        full_code: Complete iMusic.py content.
        task_num: 1, 2, or 3.
        task_func_map: {task_num: [function_names]}. Uses default iMusic mapping if None.

    Returns:
        Concatenated function bodies for the given task, or full_code if
        extraction fails.
    """
    if task_func_map is None:
        task_func_map = {
            1: ['update_playlist_tracks'],
            2: ['statistics', 'get_all_genres', 'get_statistics'],
            3: ['playlists', 'get_all_playlists', 'create_playlist',
                'rename_playlist', 'delete_playlist',
                'add_tracks_by_genre', 'remove_tracks_by_genre'],
        }

    func_names = task_func_map.get(task_num, [])
    if not func_names:
        return full_code

    # Regex-based extraction (AST-free, handles syntax errors)
    extracted = []
    for func_name in func_names:
        # Match: optional decorator + def func_name(...): + body until next top-level def
        pattern = rf'(?:@[^\n]+\n\s*)?def\s+{func_name}\s*\([^)]*\)\s*(?:->\s*\w+\s*)?\s*:.*?(?=\n(?:@[^\n]+\n\s*)?def\s+\w+\s*\(|\Z)'
        m = re.search(pattern, full_code, re.DOTALL)
        if m:
            extracted.append(m.group().strip())

    if extracted:
        return '\n\n'.join(extracted)
    # Fallback: return full code
    return full_code


def _matches_prefix(student_id: str, prefix: str) -> bool:
    """Check if student path starts with prefix (handles nested paths)."""
    parts = student_id.replace('\\', '/').split('/')
    return parts[0].startswith(prefix) or student_id.startswith(prefix)


def collect_references(ref_dir: str, ref_files: list[str]) -> str:
    """Read all reference solutions for a task into one string."""
    parts = []
    for rf in ref_files:
        path = os.path.join(ref_dir, rf)
        if os.path.exists(path):
            parts.append(f'### {rf}\n```python\n{read_file(path)}\n```')
    return '\n\n'.join(parts)


# ============================================================
# DeepSeek API call
# ============================================================

def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.3,
                  model_override: str | None = None) -> str | None:
    """Call DeepSeek API and return response text.
    Supports both deepseek-chat and deepseek-reasoner models.
    Reasoner ignores temperature and may return reasoning_content before final answer.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        temperature: Sampling temperature (ignored by reasoner models).
        model_override: Use this model instead of DEEPSEEK_MODEL env var.
                        e.g. 'deepseek-reasoner' for reasoning-heavy tasks.
    """
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    model = model_override or DEEPSEEK_MODEL
    is_reasoner = 'reasoner' in model

    # DeepSeek-R1 (reasoner) does NOT support system messages.
    # Merge system prompt into user message as a prefix block.
    if is_reasoner:
        merged_user = (
            f'[System Instructions — follow these rules strictly]\n\n'
            f'{system_prompt}\n\n'
            f'---\n\n'
            f'{user_prompt}'
        )
        messages = [{'role': 'user', 'content': merged_user}]
    else:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=8192 if is_reasoner else 16384,
    )
    if not is_reasoner:
        kwargs['temperature'] = temperature

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # For reasoner, also check reasoning_content for debugging
        if is_reasoner and hasattr(response.choices[0].message, 'reasoning_content'):
            rc = response.choices[0].message.reasoning_content
            if rc:
                pass  # reasoning_content is logged separately if needed
        return content
    except Exception as e:
        print(f'  API call failed: {e}')
        return None


def _repair_json(text: str) -> str:
    """Attempt to fix common JSON formatting errors from AI output."""
    import re as _re
    # Remove trailing commas before } or ]
    text = _re.sub(r',\s*(\}|\])', r'\1', text)
    # Fix unescaped newlines in strings (common in feedback fields)
    # Try to find and fix unbalanced quotes
    lines = text.split('\n')
    fixed = []
    in_string = False
    for line in lines:
        fixed.append(line)
    return '\n'.join(fixed)


def extract_json(text: str) -> dict:
    """Extract JSON from model response (may be wrapped in markdown fences).

    Handles common AI output issues: markdown fences, trailing commas,
    and attempts to find the outermost JSON object if extra text is present.
    """
    text = text.strip()
    # Remove markdown code fences
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
    # Try to extract just the JSON part (between first { and last })
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        text = text[start:end + 1]
    # Fix trailing commas (common AI error)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return json.loads(text)


# ============================================================
# Phase 0: Question Analysis Prompt
# ============================================================

PHASE0_SYSTEM = """You are an educational technology assistant. Your job is to analyze
a programming assignment and extract its structure. Be precise. Output ONLY valid JSON."""


def build_phase0_prompt(question_dir: str) -> str:
    """Build prompt for AI to analyze question folder and understand task structure."""
    # Read FULL PDF text (not truncated — marking scheme must be visible)
    pdf_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        for fname in sorted(filenames):
            if fname.endswith('.pdf'):
                pdf_text = _read_pdf(os.path.join(dirpath, fname))
                break

    # Read starter code
    code_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        for fname in sorted(filenames):
            if fname.endswith('.py'):
                code_text += f'### {fname}\n```python\n{read_file(os.path.join(dirpath, fname))}\n```\n\n'

    # Read data files
    data_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        for fname in sorted(filenames):
            if fname.endswith(('.tsv', '.csv', '.db')):
                rel = os.path.relpath(os.path.join(dirpath, fname), question_dir)
                data_text += f'### {rel} (data file present)\n'

    return f"""Extract the EXACT grading rubric from this programming assignment.
Your job is to COPY — not to interpret, restructure, or create.

## Complete Assignment Document (data files, starter code, full assignment PDF)
{data_text}
{code_text}

## Full Assignment PDF
{pdf_text}

## EXTRACTION RULES — Violating any of these is an error

### Rule 1: Find the authoritative source
The document contains a section that defines the grading rubric. This is the ONLY
source of truth. Look for:
  - A table or list with criterion identifiers (e.g., RQ1_1, C1, Task1.1, etc.)
  - Each row/entry has: an ID, a name/requirement, a description/details, and a mark/point value.
  - Sections labeled "Marking Scheme", "Marking Criteria", "Rubric", "Grading", or similar.
If the document describes an implementation requirement in prose but does NOT assign
it a criterion ID and a mark value, it is NOT part of the rubric — ignore it.

### Rule 2: Copy verbatim, do not create
For EVERY criterion found in the authoritative source:
  - **id**: The EXACT identifier string from the document. Do not renumber or rename.
  - **name**: The EXACT "Requirement" or name text. Do not paraphrase.
  - **description**: The EXACT "Details" or description text. Do not shorten or expand.
  - **marks**: The numeric mark value from the document.
  - **good_patterns**: Infer from the requirement text — what code patterns would
    SATISFY this criterion? Be specific (e.g., "uses parameterized cursor.execute(...)
    with ? placeholders", not "writes good SQL").
  - **bad_patterns**: Infer from the requirement text — what code patterns would
    VIOLATE this criterion? Be specific.

### Rule 3: Count must match exactly
The number of criteria you output for each task MUST equal the number of criteria
in the document's marking scheme for that task. If the document has 4 criteria
for Task 1, you output exactly 4 — no more, no less.

### Rule 4: No splitting, no merging
- If the document lists one criterion row, output one criterion. Do not split
  it into sub-criteria just because it covers multiple behaviors.
- If the document lists two separate criterion rows (two IDs, two mark values),
  output two criteria. Do not merge them into one.

### Rule 5: Prose descriptions are not criteria
The document may describe requirements in narrative prose (e.g., "You should also
handle errors gracefully"). If this prose does NOT have a dedicated criterion ID
and mark value in the grading table, it is instructional text, not a rubric
criterion. Do NOT create a criterion from it.

## Output format
{{"question_name": "...", "tasks": [
  {{"id": "Task1", "target_file": "filename.py", "target_functions": ["..."],
    "rubric_criteria": [
      {{"id": "EXACT_ID_FROM_DOC", "name": "Exact requirement name",
        "description": "Exact details from document", "marks": 2,
        "good_patterns": ["..."], "bad_patterns": ["..."]
      }}
    ]
  }}
]}}"""


# ============================================================
# Phase 1: Reference Solution Generation Prompt
# ============================================================

def build_phase1_prompt(task_id: str, task_cfg: dict, question_dir: str) -> str:
    """Build prompt for AI to generate per-criterion reference implementation variants.

    For EACH rubric criterion, generates 3-5 distinct correct code snippets
    demonstrating different valid approaches. This provides the Phase 2 FS
    generator with a comprehensive catalog of correct patterns.
    """
    # Collect full context
    pdf_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fname in sorted(filenames):
            if fname.endswith('.pdf'):
                pdf_text = _read_pdf(os.path.join(dirpath, fname))
                break

    code_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        for fname in sorted(filenames):
            if fname.endswith('.py'):
                code_text += f'### {fname}\n```python\n{read_file(os.path.join(dirpath, fname))}\n```\n\n'

    criteria_text = json.dumps(task_cfg.get('rubric_criteria', []), indent=2)

    return f"""You are generating reference implementations for ONE specific task.
Generate per-criterion variants ONLY for the criteria listed below.

## TARGET TASK: {task_id}
File: {task_cfg.get('target_file', '')}
Functions: {json.dumps(task_cfg.get('target_functions', []))}

## GRADING CRITERIA FOR {task_id} — generate variants for THESE EXACT criteria
{criteria_text}

CRITICAL: You MUST generate variants for the {len(task_cfg.get('rubric_criteria', []))} criteria
listed above. The criterion IDs are {json.dumps([c['id'] for c in task_cfg.get('rubric_criteria', [])])}.
DO NOT generate variants for any other task's criteria.

## Starter Code (the template students start from)
{code_text}

## Assignment Context (relevant task description from PDF)
{pdf_text[:5000]}

## CRITICAL Requirements

1. **Per-criterion granularity**: For EACH criterion (RQ1_1, RQ1_2, etc.), generate
   3-5 distinct code snippets that satisfy THAT criterion's good_patterns.
   Each snippet is a complete, runnable function.

2. **Diversity within each criterion**: The 3-5 variants must use DIFFERENT approaches:
   - Different library APIs (e.g., csv.reader vs csv.DictReader)
   - Different validation styles (e.g., SELECT COUNT + >0 vs SELECT 1 + is not None)
   - Different control flow (e.g., early return vs if/else chains)
   - Different error handling patterns (try/except/finally vs with-statement)
   Each variant must be a COMPLETE, WORKING implementation of the criterion.

3. **RUBRIC COMPLIANCE (MANDATORY)**: Before outputting each snippet, verify it
   satisfies ALL good_patterns for its criterion. If a good_pattern is missing,
   the snippet is WRONG — do not output it.

4. **Exact schema**: Use EXACT table/column names from the assignment.
   Check carefully: Playlist (not playlists), Track (not tracks), etc.

5. **Self-contained**: Each snippet is a complete function with imports if needed.

## Output format
{{"criterion_implementations": {{
  "RQ1_1": [
    {{"variant": "A", "approach": "csv.reader with next() and numeric indexing",
      "code": "def update_playlist_tracks(path: Path):\\n    ..."}},
    {{"variant": "B", "approach": "csv.DictReader with dict key access",
      "code": "def update_playlist_tracks(path: Path):\\n    ..."}},
    ...
  ],
  "RQ1_2": [...]
}}}}"""


def _read_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        return text.strip()
    except Exception as e:
        return f'[PDF extraction failed: {e}]'


def _collect_question_files(question_dir: str) -> str:
    """Read all files from question folder into a single context string.
    Handles: .py, .md, .txt, .csv, .tsv, .pdf (extracts text)."""
    parts = []
    for dirpath, dirnames, filenames in os.walk(question_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, question_dir)

            if fname.endswith('.pdf'):
                content = _read_pdf(full)[:5000]
                if content.strip():
                    parts.append(f'### {rel} (PDF text extracted)\n{content}')
            elif fname.endswith(('.py', '.md', '.txt', '.csv', '.tsv')):
                content = read_file(full)[:5000]
                if content.strip():
                    parts.append(f'### {rel}\n```\n{content}\n```')
    return '\n\n'.join(parts)


# ============================================================
# Phase 2: FS Generation Prompt
# ============================================================

SYSTEM_PROMPT = """You are an automated grading assistant using the TAFFIES
(Tailored Automated Feedback Framework) methodology.

Your job: generate Feedback Signatures (FS) from student code.
Each FS = a regex matching a specific code pattern + student-facing feedback.

CRITICAL RULES:
1. Work criterion-by-criterion. For EACH rubric criterion, scan EVERY student
   to find ALL distinct ways they implemented (or violated) it.
2. Every distinct code variant gets its own FS. If 5 students write the same
   validation 5 different ways, generate 5 FS -- one regex per variant.
3. Every FS MUST have a non-null regex matching actual student code.
   BAD FS: match the WRONG code itself (f-string SQL, print(), wrong table name).
4. Every student must map to at least one FS per criterion.
   Cross-check: after generating FS, verify each student has >=1 FS per criterion.
   IMPORTANT: FS must match STUDENT-WRITTEN code, not starter template code.
   The template provides imports, Flask setup, and function stubs with pass.
   If your regex matches these unchanged template lines, it gives credit for
   code the student didn't write. Your regex should match what the student
   ADDED or CHANGED — not what was already there.

HARD CONSTRAINTS (violations will cause incorrect grading):
5. Regex MUST match executable CODE, NOT comments, docstrings, or #-prefixed lines.
   If your regex matches text inside a comment, it is WRONG and will be rejected.
   Test: mentally strip all comments from the student code, then check if your
   regex still matches something meaningful.
6. NEGATIVE FS MUST NOT match reference solutions. Reference code is CORRECT.
   If your negative regex matches reference code, it means your pattern is too
   broad and would incorrectly penalize correct implementations.
   Test: for each negative FS, mentally verify it does NOT match the reference code.

NEGATIVE FS REGEX RULES (CRITICAL — wrong regex = false grading):

There are TWO types of negative FS. CLASSIFY before writing the regex:

7a. TYPE A — "Missing a required good pattern"
   Use when the student FAILED to do something correct.
   Examples: missing validation, missing parameterized queries, missing decorator.
   Approach: Use negative lookahead (?!...) to verify the good pattern is ABSENT.

   WRONG for Type A (pure positive — matches everyone):
     name: "No validation before INSERT"
     regex: "INSERT\\s+INTO\\s+PlaylistTrack"
     → Matches ALL code with INSERT, even if validation exists. FALSE NEGATIVE.

   RIGHT for Type A (negative lookahead):
     name: "Missing PlaylistId validation before INSERT"
     regex: "def\\s+update_playlist_tracks\\s*\\([^)]*\\)\\s*:(?!.*SELECT\\s+COUNT.*Playlist)"
     → Only matches when function body truly lacks a validation query.

7b. TYPE B — "Presence of a bad pattern / mistake"
   Use when the student DID something WRONG that should NOT be there.
   Examples: f-string SQL injection, % formatting SQL, string concatenation SQL,
   hardcoded path, wrong table name, print() debug, .format() SQL.
   Approach: Pure positive regex IS CORRECT. Match the bad pattern directly.
   The regex does NOT need (?!...) because you are detecting PRESENCE of bad code.

   CORRECT for Type B (pure positive — this IS the right approach):
     name: "f-string SQL injection in update_playlist_tracks"
     regex: "f[\"'].*\\b(?:INSERT|UPDATE|DELETE)\\b"
     → Detects f-string used in SQL. Reference code uses ? placeholders,
       so this does NOT match reference. Correct negative FS.

     name: "% formatting in SQL query"
     regex: "\\.execute\\s*\\(\\s*[\"'].*%[sd]\\s*[\"']\\s*%\\s*\\("
     → Detects %-formatting SQL. Reference uses ? placeholders. Correct.

     name: "String concatenation in SQL"
     regex: "\\.execute\\s*\\(\\s*[\"'].*[\"']\\s*\\+\\s*str\\("
     → Detects string concatenation SQL. Reference uses ?. Correct.

     name: "Hardcoded file path instead of parameter"
     regex: "(?:open|path)\\s*\\(\\s*[\"'](?:/|[A-Z]:)"
     → Detects hardcoded paths. Reference uses Path objects. Correct.

   KEY RULE for Type B: The bad pattern must NOT appear in reference code.
   If reference also has this pattern, it's not really a mistake → use Type A.

8. GOLDEN RULE (applies to BOTH types): After writing a negative FS regex, test:
   "Would this regex match the REFERENCE SOLUTION (which is 100% correct)?"
   If YES → the regex is wrong. For Type A, tighten the (?!...). For Type B,
   the "bad" pattern isn't actually bad — reconsider whether this FS is needed.

9. CLASSIFY before writing: Is this FS about something MISSING (Type A → (?!...))
   or something WRONG (Type B → pure positive)? For Type B, verify the pattern
   genuinely does NOT appear in reference code. Pure positive IS the correct
   approach for Type B — do not force (?!...) where it doesn't belong.

FEEDBACK QUALITY RULES (critical for student learning):
10. Every feedback MUST be unique -- never repeat the same phrasing across FS.
    Two FS for the same criterion should read like two different human graders wrote them.
11. Be CONCRETE about what the student's code does. Name the specific function,
    library, pattern, or value you observed. Instead of "you used the correct API",
    say "csv.DictReader with delimiter='\\t' correctly parses the TSV header row."
12. For POSITIVE feedback: explain WHY this specific implementation choice is good
    for THIS specific criterion. Don't say "good practice" -- say WHY it matters here.
    CRITICAL: A positive FS MUST award credit for code that ACTUALLY WORKS.
    Do NOT mark code as positive if it uses a pattern that will fail at runtime.
    Example: "ORDER BY ? ?" with parameterized placeholders does NOT work in
    SQLite — the ? becomes a literal string, not a column reference. This must
    be NEGATIVE, not positive, because it produces incorrect results.
13. For NEGATIVE feedback: name the CONSEQUENCE of the mistake. Instead of "this is
    incorrect", say "this will raise a NameError because 'playlists' is not defined"
    or "this bypasses the validation check, allowing duplicate entries."
    IMPORTANT: Only mark code as negative if it is GENUINELY BROKEN (will error,
    produce wrong results, or create security issues). Do NOT penalise code that
    uses a different but functionally correct approach.
    Example: `.split('\\t')` parses TSV correctly for simple cases — do not flag
    it as negative just because the "preferred" approach is csv.DictReader.
    Example: `SELECT * FROM Genre` returns correct data — only flag it if the
    rubric EXPLICITLY requires specific columns. When in doubt, be POSITIVE.
14. Vary your sentence openers. Don't start every feedback with "You used..."
    or "Correct:". Use a mix of structures: questions, observations, comparisons.
15. Target 2-4 sentences per feedback. One-sentence feedback is too vague; five
    sentences is too verbose. Aim for: what you saw + why it's right/wrong + what
    to do next (for negatives) or what this enables (for positives).

CRITICAL — SOURCE CODE ESCAPE SEQUENCES (regex \t vs literal text):
16. When your regex matches Python SOURCE CODE (not runtime values), remember
    that escape sequences like \\t, \\n, \\r are TWO CHARACTERS in the .py file.
    Your regex MUST use \\\\t (four backslashes in JSON = \\t in regex = matches
    literal backslash-t in source code). Using \\t (two chars in JSON = tab char
    in regex) matches an actual TAB character (ASCII 0x09), which NEVER appears
    in student source code.
      WRONG: delimiter\\s*=\\s*['\"]\\\\t['\"]  → matches literal TAB char (0x09)
      RIGHT: delimiter\\s*=\\s*['\"]\\\\\\\\t['\"] → matches source code '\\t'
    This applies to ALL regexes matching string literals in Python source:
    delimiter, split, replace, or any argument containing escape sequences.

CRITICAL — NEGATIVE FS FUNCTION SCOPE (do NOT use \\w+ for function names):
17. Negative FS that detect missing code MUST target the SPECIFIC function
    by its EXACT name. Never use \\w+ as a wildcard for the function name
    in a def ... pattern — this matches EVERY function in the file, including
    template stubs from other tasks and Flask framework functions.
      WRONG: def\\s+\\w+\\s*\\([^)]*\\)\\s*:(?!.*INSERT)
             → Matches upload_route(), index(), statistics(), EVERY function!
      RIGHT: def\\s+create_playlist\\s*\\([^)]*\\)\\s*:(?!.*INSERT)
             → Only matches the create_playlist function.
    For cross-student generalisation, list ALL known function names with |:
      def\\s+(?:create_playlist|rename_playlist|delete_playlist)\\s*\\

CRITICAL — CONFLICT PREVENTION (positive + negative FS must not overlap):
18. For each criterion, a student MUST NOT be matched by BOTH a positive AND a
    negative FS. If a student passes some good_patterns but fails others, choose
    ONE verdict (positive if ALL good_patterns pass, negative otherwise).
    Never generate both a positive AND negative FS matching the same student.

    After generating FS for a criterion, mentally test: "Would my positive FS
    and negative FS both match student X?" If yes, the negative FS is too broad
    and must be narrowed — add a negative lookahead (?!...) that excludes the
    specific correct pattern that the positive FS matches.

CRITICAL — REGEX WILDCARD RULES BY CONTEXT:
19. Single-line matching: use [^\\n]* (safe, no backtracking)
    Cross-line matching in negative lookahead: use [\\s\\S]*? (non-greedy,
    safe inside (?!...) which fails fast)
    NEVER use .* or .+ (catastrophic backtracking risk)
    NEVER use [^\\n]* inside a (?!...) that needs to see across multiple
    lines. The lookahead will NOT find patterns on other lines.

CRITICAL — PYTHON FUNCTION SYNTAX (this is NOT C/Java — no curly braces):
20. Python functions use `def name(params):` with COLON and INDENTATION.
    NEVER use `{` or `}` as function body delimiters in regex.
    WRONG: def\\s+func[^{]*\\{  → this matches C/Java, not Python!
    WRONG: def\\s+func[\\s\\S]*?\\{  → curly brace means dict/f-string, not function body
    RIGHT: def\\s+func_name\\s*\\([^)]*\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:
    When matching "from def to end of function", anchor to the NEXT def:
      def\\s+target_func\\s*\\([^)]*\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:[\\s\\S]*?(?=\\ndef\\s+\\w+\\s*\\(|$)
    When using negative lookahead to detect MISSING pattern inside a function:
      def\\s+target_func\\s*\\([^)]*\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:(?!.*required_pattern)
    TEST: Does your regex contain { or }? If yes, rewrite using the patterns above.

CRITICAL — NEGATIVE FS FOR MISSING DECORATORS (two-line match required):
20. A negative FS for "missing @app.route decorator" MUST verify the decorator
    is actually absent. Matching just "def function_name():" penalizes ALL
    students including those WITH the decorator on the line above.
      WRONG: (?:^|\\n)\\s*def\\s+statistics\\s*\\([^)]*\\)\\s*:
             -> Matches ALL statistics() functions, decorator or not.
      RIGHT: Check the two-line pattern — def at line start means no decorator:
             (?:^|\\n)\\s*def\\s+(?:statistics|playlists)\\s*\\([^)]*\\)\\s*:
             Only works if the def is genuinely at line start (no @app.route
             on the preceding line). Test against reference code first.
    Alternative: detect truly unimplemented functions via pass/empty body:
      def\\s+function_name\\s*\\([^)]*\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:\\s*\\n\\s*(?:#.*\\n\\s*)*pass\\b
    This only matches when the function body is a pass statement.

POST-GENERATION SELF-CHECK (run mentally before outputting JSON):
For each criterion you generated FS for:
  [ ] Every distinct CORRECT variant has at least 1 positive FS
  [ ] Every distinct ERROR variant has at least 1 negative FS
  [ ] NO student is matched by BOTH positive AND negative FS
  [ ] Every negative FS regex does NOT match reference code
  [ ] Every positive FS regex does NOT match template stub code
  [ ] Every FS has 2-3 sentence non-empty feedback

CRITICAL — IDENTIFIER GENERALISATION (applies to ALL regexes you write):
The following are the ONLY identifiers allowed to appear LITERALLY in regex.
ALL other identifiers MUST be matched with \\\\w+ instead.

{WHITELIST_RULE}

This is MANDATORY for cross-student generalisation. If you hardcode a variable
name that is not in the whitelist (e.g., a student's local variable), your regex
will fail to match other students who use a different variable name.

Examples:
  WRONG: cursor\\.execute\\(query,\\s*\\(sort_column,\\s*sort_order\\)\\)
         → Only matches students who named their variables sort_column/sort_order
  RIGHT: \\w+\\.execute\\(\\w+,\\s*\\(\\w+,\\s*\\w+\\)\\)
         → Matches ALL students regardless of variable naming

Output ONLY valid JSON."""


def build_bad_pattern_summary(all_readmes: dict, task_id: str) -> str:
    """Build a text summary of bad patterns per criterion from READMEs.

    Uses pre-parsed criteria dict from load_all_readmes().
    """
    from collections import Counter

    crit_bad_patterns: dict[str, Counter] = defaultdict(lambda: Counter())
    crit_total_students: dict[str, int] = defaultdict(int)

    task_num_str = task_id.replace('Task', '')

    for sid, rdata in all_readmes.items():
        criteria = rdata.get('criteria', {})
        # Handle old format where criteria is a string
        if isinstance(criteria, str):
            try:
                criteria = eval(criteria)
            except Exception:
                continue
        if not isinstance(criteria, dict):
            continue

        for crit, patterns in criteria.items():
            if not crit.startswith(f'RQ{task_num_str}'):
                continue
            crit_total_students[crit] += 1
            for bp in patterns.get('bad', []):
                if bp:
                    crit_bad_patterns[crit][bp] += 1

    lines = []
    lines.append('## Bad Pattern Distribution (from ground truth READMEs)')
    lines.append('This tells you which criteria have intentional errors in student code.')
    lines.append('')

    all_criteria_in_task = sorted(set(
        c for c in crit_bad_patterns.keys() | crit_total_students.keys()
    ))

    for crit in sorted(all_criteria_in_task):
        patterns = crit_bad_patterns.get(crit, Counter())
        total = crit_total_students.get(crit, 0)
        if patterns:
            lines.append(f'**{crit}**: {total} students have bad patterns:')
            for bp_desc, count in patterns.most_common():
                lines.append(f'  - [{count} students] {bp_desc}')
            lines.append(f'  -> Generate 1 Type B negative FS per distinct bad pattern variant above.')
        else:
            lines.append(f'**{crit}**: NO students have bad patterns.')
            lines.append(f'  -> Do NOT generate any negative FS for this criterion.')

    lines.append('')
    return '\n'.join(lines)


def _extract_code_variants(submissions: list[dict]) -> dict[str, list[str]]:
    """Scan student code to find naming variants for tables, columns, functions.
    Returns {category: [unique_variants_found]}.
    """
    import re as _re
    variants = {
        'table_names': set(),
        'function_names': set(),
        'variable_patterns': set(),
    }
    for s in submissions[:15]:
        code = s.get('code', '')
        # Table names in SQL
        for m in _re.finditer(r'(?:FROM|INTO|UPDATE|TABLE)\s+(\w+)', code, _re.IGNORECASE):
            variants['table_names'].add(m.group(1))
        for m in _re.finditer(r'(?:JOIN)\s+(\w+)', code, _re.IGNORECASE):
            variants['table_names'].add(m.group(1))
        # Function definitions
        for m in _re.finditer(r'def\s+(\w+)', code):
            variants['function_names'].add(m.group(1))
        # .execute( patterns
        for m in _re.finditer(r'\.(\w+)\s*\(', code):
            variants['variable_patterns'].add(m.group(1))
    return {k: sorted(v)[:20] for k, v in variants.items()}


def build_fs_prompt(task_id: str, task_cfg: dict, ref_code: str, submissions: list[dict],
                    template_code: str = '', readme_patterns: str = '',
                    previous_fs: list[dict] | None = None,
                    batch_label: str = '',
                    bad_pattern_summary: str = '') -> str:
    """Build the FS generation prompt for one task.

    Enhanced version:
    - Student code is per-task (template already stripped by CW format)
    - No 2000-char truncation — sends complete task functions
    - Includes README ground truth patterns (what to look for)
    - Includes bad pattern distribution (which criteria need negative FS)
    - Includes previous batches' FS (to avoid duplicates)
    - Does NOT include full template code (students already have task-only code)
    """
    # Student code is already task-specific (task1.py etc.), template-free
    # Send complete code, no truncation
    student_text = '\n\n'.join(
        f"### Student: {s['student']}\n```python\n{s['code']}\n```"
        for s in submissions
    )
    criteria = json.dumps(task_cfg.get('rubric_criteria', []), indent=2)

    # Extract naming variants from actual student code
    variants = _extract_code_variants(submissions)
    variants_text = f"""
## Student Code Variants (found in actual submissions — use these in regex)
Table names found: {json.dumps(variants.get('table_names', []))}
Function names found: {json.dumps(variants.get('function_names', []))}
CRITICAL: Your regex MUST use (?:variant1|variant2|...) to match ALL table name variants.
For example, if students use both 'PlaylistTrack' and 'playlist_track', write:
  (?:PlaylistTrack|playlist_track)
DO NOT hardcode just one spelling — regex WILL miss students using the other.
"""

    # Bad pattern summary section (tells AI which criteria need negative FS)
    bad_pattern_section = bad_pattern_summary if bad_pattern_summary else ''

    # README ground truth section
    readme_section = ''
    if readme_patterns:
        readme_section = f'\n{readme_patterns}\n\nCRITICAL: Generate at least one FS for EVERY pattern listed above (both good and bad).\n'

    # Previous FS section (from earlier batches)
    prev_fs_section = ''
    if previous_fs:
        prev_lines = [
            f'\n## Previous Batches FS ({len(previous_fs)} FS already generated)',
            'These FS were generated from other student batches. DO NOT duplicate them.',
            'Your job: generate FS for NEW patterns not already covered below.',
            '',
        ]
        for fs in previous_fs:
            prev_lines.append(
                f"- {fs.get('id','?')} [{fs.get('fs_type','?')}] "
                f"c={fs.get('criterion','?')}: {fs.get('name','?')}\n"
                f"  regex: `{fs.get('regex','null')[:100]}`"
            )
        prev_lines.append(
            '\nIMPORTANT: If a pattern in the GROUND TRUTH list above is already '
            'covered by one of these FS, do NOT create a duplicate. Only create FS '
            'for patterns that do NOT have a matching regex above.'
        )
        prev_fs_section = '\n'.join(prev_lines)

    # Template section — only show function signatures that students must NOT match
    # (template code is already stripped from student submissions, so this is for
    #  awareness of what the starter code looks like)
    template_note = ''
    if template_code:
        # Only extract function signatures, not full template
        sigs = re.findall(r'(def\s+\w+\s*\([^)]*\))', template_code)
        if sigs:
            template_note = (
                '\n## Template Function Signatures (for reference — these are NOT student code)\n'
                + '\n'.join(f'  - {s}' for s in sorted(set(sigs)))
                + '\n'
            )

    batch_header = f' [{batch_label}]' if batch_label else ''

    return f"""## Rubric Criteria with Individual Good/Bad Patterns{batch_header}
Each criterion has good_patterns (what to do) and bad_patterns (what NOT to do).
Generate ONE positive FS + ONE negative FS PER good_pattern, not per criterion.
This enables partial credit: a student satisfying 3/4 good_patterns gets 3 pos + 1 neg.

{criteria}

## Reference Implementations (diverse correct approaches)
{ref_code if ref_code else '(No references)'}
{bad_pattern_section}
{readme_section}{prev_fs_section}
{template_note}{variants_text}
## Student Submissions ({len(submissions)} students, COMPLETE task code)
{student_text}

## Instructions — Work GOOD_PATTERN BY GOOD_PATTERN

### For EACH good_pattern of EACH criterion:

**Step 1: Find all correct implementations**
Scan ALL students. Find every distinct way a student SATISFIES this specific
good_pattern. Each distinct approach → one positive FS.

**Step 2: Find all violations**
Find every distinct way a student VIOLATES this specific good_pattern.
Each distinct violation → one negative FS.

CRITICAL — Before generating negative FS, check the "Bad Pattern Distribution" section:
  - If a criterion has NO bad patterns → SKIP negative FS entirely for that criterion.
  - If a criterion HAS bad patterns → use TYPE B (detect the EXACT bad pattern variants listed).
    TYPE A (detecting absence of good pattern) should ONLY be used for criteria with
    bad patterns that are NOT listed as specific variants (e.g., general "no validation").

**Step 3: Generate regex — CRITICAL RULES**

  POSITIVE FS regex:
    - Match ALL student naming variants: (?:PlaylistTrack|playlist_track)
    - Use \\\\w+ for variable names, \\\\d+ for numbers
    - Do NOT hardcode specific variable names

  NEGATIVE FS regex — TWO types, choose correctly:

    TYPE A — "Missing good pattern" (student didn't do what's required):
      - MUST use (?!...) to detect ABSENCE of the required pattern
      - The (?!...) must span the ENTIRE function body
      - NEVER use bare `def function_name` — matches everyone including correct code
      - Example: def\\s+func\\s*\\([^)]*\\)\\s*:(?!.*required_pattern)

    TYPE B — "Present bad pattern" (student did something wrong):
      - Pure positive regex IS correct — match the bad pattern directly
      - Does NOT need (?!...) because you are detecting PRESENCE of bad code
      - Must NOT appear in reference code (if it does, it's not really bad)
      - Example: f[\"'].*\\bINSERT\\b — detects f-string SQL injection
      - Example: \\.execute\\s*\\(\\s*[\"'].*%[sd].*%\\s*\\( — detects % formatting SQL

    For source-code escape sequences: \\\\t matches literal backslash-t (two chars)

**Step 4: Cross-check**
Each student MUST have >=1 FS per criterion (any good_pattern).
At least 2 POSITIVE FS per criterion covering different approaches.

**Step 5: Quality checks**
  a) POSITIVE FS: regex must NOT match template starter code
  b) NEGATIVE FS: regex must NOT match reference code
  c) NEGATIVE FS — Type A: (?!...) correctly positioned; Type B: bad pattern genuinely absent from reference

**Step 6: Write feedback (2-3 sentences per FS, MANDATORY)**
Each FS targets ONE specific good_pattern — name it in the feedback.

### Output Format
{{"fs_registry": [
  {{
    "name": "GP: validates name not empty — early return pattern",
    "fs_type": "positive",
    "criterion": "RQ3_2",
    "good_pattern": "validates that playlist name is not empty",
    "regex": "if\\s+not\\s+\\w+[\\s\\S]*?flash\\([^)]*danger",
    "regex_flags": "IGNORECASE",
    "feedback": "You check whether the playlist name is empty before inserting..."
  }}
]}}"""


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(question_dir: str, submissions_dir: str,
                 question_id: str, question_name: str = '',
                 ref_dir: str = '',
                 student_prefix: str | None = None):
    """
    Full AI-driven FS generation pipeline.

    Args:
        question_dir: Path to question folder.
        submissions_dir: Path to submissions folder.
        question_id: Short identifier for this question.
        question_name: Human-readable name (auto-detected if empty).
        ref_dir: Path to reference solutions (empty = AI will generate).
        student_prefix: Filter for student directories (auto-detected if None).
    """
    print('=' * 60)
    print('  AI-DRIVEN FS GENERATION')
    print(f'  Model: {DEEPSEEK_MODEL}')
    print('=' * 60)

    # ================================================================
    # Phase 0: AI analyzes question folder -> task config
    # ================================================================
    # Check for existing FS early (for incremental/gap-only mode on retry)
    existing_fs_path = os.path.join(BASE_DIR, 'output', question_id, 'fs_registry.json')
    has_existing = os.path.exists(existing_fs_path)

    print('\n--- Phase 0: AI analyzing question folder ---')
    p0_prompt = build_phase0_prompt(question_dir)
    print(f'  Prompt size: {len(p0_prompt)} chars')

    MAX_P0_RETRIES = 3
    question_config = None
    for p0_attempt in range(1, MAX_P0_RETRIES + 1):
        p0_response = call_deepseek(PHASE0_SYSTEM, p0_prompt)
        if not p0_response:
            print(f'  FAILED (attempt {p0_attempt}/{MAX_P0_RETRIES}) -- empty response')
            continue
        try:
            question_config = extract_json(p0_response)
            break
        except Exception as e:
            print(f'  Parse error (attempt {p0_attempt}/{MAX_P0_RETRIES}): {e}')
            if p0_attempt < MAX_P0_RETRIES:
                print(f'  Retrying...')

    if not question_config:
        print('  FAILED -- cannot proceed without question analysis')
        return

    tasks = question_config.get('tasks', [])
    if not tasks:
        print('  No tasks detected')
        return

    question_name = question_name or question_config.get('question_name', question_id)
    print(f'  Detected: {question_name} -- {len(tasks)} tasks')
    for t in tasks:
        print(f'    {t["id"]}: {t.get("target_file", "?")} -> {t.get("target_functions", [])}')

    # Cross-validate: AI's target_file may include path prefix (code/iMusic.py)
    # or be completely wrong. Match by basename against actual submission files.
    from config_generator import scan_submissions_dir
    si = scan_submissions_dir(submissions_dir)
    actual_files = list(si.get('common_files', {}).keys())

    if actual_files:
        for task in tasks:
            guessed = task.get('target_file', '')
            guessed_basename = os.path.basename(guessed)

            # Already correct?
            if guessed in actual_files or guessed_basename in actual_files:
                if guessed != guessed_basename:
                    task['target_file'] = guessed_basename
                    print(f'    Stripped path: {guessed} -> {guessed_basename}')
                continue

            # AI guessed wrong -- find matching file
            tid = task['id']
            candidates = [f for f in actual_files if tid.lower() in f.lower()]
            if not candidates and len(actual_files) >= len(tasks):
                idx = list(tasks).index(task)
                if idx < len(actual_files):
                    candidates = [actual_files[idx]]
            if candidates:
                task['target_file'] = candidates[0]
                print(f'    Corrected {tid} target_file: {guessed} -> {candidates[0]}')

    # Auto-detect student prefix if not provided
    if student_prefix is None and student_prefix != '':
        from config_generator import scan_submissions_dir
        si = scan_submissions_dir(submissions_dir)
        prefixes = si.get('detected_prefixes', [])
        if prefixes:
            # Pick the prefix that matches this question_id
            for p in prefixes:
                if question_id.lower().startswith(p.lower().rstrip('-_')):
                    student_prefix = p
                    break
            if not student_prefix and len(prefixes) > 1:
                student_prefix = prefixes[0]  # Use first detected prefix
        if student_prefix:
            print(f'  Auto-detected student prefix: {student_prefix}')

    # ================================================================
    # Phase 1: Ensure reference solutions exist (generate if needed)
    # ================================================================
    if not ref_dir:
        ref_dir = os.path.join(BASE_DIR, 'references', question_id)

    for task in tasks:
        task_id = task['id']
        task_ref_dir = os.path.join(ref_dir, task_id) if os.path.isdir(ref_dir) else ref_dir
        os.makedirs(task_ref_dir, exist_ok=True)

        # Check if refs already exist
        existing = [f for f in os.listdir(task_ref_dir) if f.endswith('.py')] if os.path.isdir(task_ref_dir) else []
        if len(existing) >= 2:
            task['reference_files'] = existing
            continue

        print(f'\n--- Phase 1: Generating references for {task_id} ---')
        p1_prompt = build_phase1_prompt(task_id, task, question_dir)
        print(f'  Prompt size: {len(p1_prompt)} chars')
        p1_response = call_deepseek(PHASE0_SYSTEM, p1_prompt)
        if not p1_response:
            print(f'  FAILED -- will proceed without references for {task_id}')
            task['reference_files'] = []
            continue

        try:
            ref_data = extract_json(p1_response)
            solutions = ref_data.get('solutions', [])
        except Exception as e:
            print(f'  Parse error: {e}')
            task['reference_files'] = []
            continue

        saved = []
        for sol in solutions:
            fname = sol.get('filename', f'{task_id.lower()}_v{chr(65+len(saved))}.py')
            fpath = os.path.join(task_ref_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(sol.get('code', ''))
            saved.append(fname)

        task['reference_files'] = saved
        print(f'  Generated {len(saved)} reference solutions: {saved}')

    # Batch -> Task mapping
    BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}

    # Load EXISTING FS from previous run (incremental mode)
    all_fs = []
    if has_existing:
        try:
            with open(existing_fs_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            all_fs = old_data.get('fs_registry', [])
            has_existing = True
            print(f'\n  Loaded {len(all_fs)} existing FS -- gap-only mode')
        except: pass

    # Rebuild ID counter from existing FS
    fs_id_counter = {}
    for fs in all_fs:
        crit = fs.get('criterion', '?')
        num = re.search(r'\d+', str(crit))
        num = num.group() if num else '0'
        fs_id_counter[num] = max(fs_id_counter.get(num, 0),
                                 int(fs.get('id', 'FS0.0').split('.')[-1] or 0))

    if has_existing:
        print('  Skipping Phase 2 -- only filling gaps below 100%')
    else:
        print(f'\n--- Phase 2: Generating FS ---')

    for batch, matched_task in sorted(BATCH_TASK_MAP.items()):
        task = next((t for t in tasks if t['id'] == matched_task), None)
        if not task:
            continue
        task_id = task['id']
        target_file = task.get('target_file', '')
        if not has_existing:
            print(f'\n--- Phase 2: Generating FS for {batch} -> {task_id} ({target_file}) ---')

        # Skip Phase 2 API calls if we already have FS
        if has_existing:
            continue

        # Collect ONLY this batch's submissions
        submissions = collect_submissions(submissions_dir, target_file,
                                          student_prefix=batch, max_students=30)
        if not submissions:
            print(f'  No submissions found, skipping')
            continue

        # Collect references
        ref_files = task.get('reference_files', [])
        ref_code = ''
        for rf in ref_files:
            for root, dirs, files in os.walk(ref_dir):
                if rf in files:
                    ref_code += f'### {rf}\n```python\n{read_file(os.path.join(root, rf))}\n```\n'
                    break

        print(f'  References: {len(ref_files)}, Submissions: {len(submissions)}')

        # Build prompt & call AI (with retry on JSON parse failure)
        prompt = build_fs_prompt(task_id, task, ref_code, submissions)
        print(f'  Calling API... ({len(prompt)} chars)')

        task_fs = []
        MAX_P2_RETRIES = 3
        for p2_attempt in range(1, MAX_P2_RETRIES + 1):
            response = call_deepseek(SYSTEM_PROMPT, prompt)
            if not response:
                print(f'  FAILED (attempt {p2_attempt}/{MAX_P2_RETRIES}) -- empty response')
                continue

            try:
                data = extract_json(response)
                task_fs = data.get('fs_registry', [])
                break  # Success
            except Exception as e:
                print(f'  Parse error (attempt {p2_attempt}/{MAX_P2_RETRIES}): {e}')
                if p2_attempt < MAX_P2_RETRIES:
                    # Try repair before retry
                    try:
                        fixed = _repair_json(response)
                        data = extract_json(fixed)
                        task_fs = data.get('fs_registry', [])
                        print(f'    Repair succeeded on attempt {p2_attempt}')
                        break
                    except Exception:
                        print(f'    Retrying...')
                        continue

        if not task_fs:
            print(f'  FAILED after {MAX_P2_RETRIES} attempts -- no FS for {batch}/{task_id}')
            continue

        # Assign IDs
        for fs in task_fs:
            fs.setdefault('task', task_id)
            fs.setdefault('files', [target_file])
            fs.setdefault('auto_generated', True)
            fs.pop('marks', None)
            crit = fs.get('criterion', '?')
            m = re.search(r'\d+', str(crit)); num = m.group() if m else '0'
            fs_id_counter.setdefault(num, 0)
            fs_id_counter[num] += 1
            fs['id'] = f'FS{num}.{fs_id_counter[num]}'

        all_fs.extend(task_fs)
        pos = sum(1 for f in task_fs if f.get('fs_type') == 'positive')
        neg = sum(1 for f in task_fs if f.get('fs_type') == 'negative')
        print(f'  Generated {len(task_fs)} FS ({pos}+, {neg}-)')

    # ================================================================
    # Phase 2.5: FCC coverage loop
    # ================================================================
    print(f'\n{"=" * 60}')
    print('  Phase 2.5: FCC Coverage Loop')
    print('=' * 60)

    from coverage import (
        run_coverage_check, find_gaps, build_supplement_prompt, format_coverage_report
    )

    MAX_FCC_ITERATIONS = 5
    MIN_GAP_SIZE = 1  # Fill ALL gaps

    # --- Batch -> Task mapping ---
    # q1 students only implemented Task1, q2 only Task2, q3 only Task3.
    # Only check FS against the batch that actually wrote code for that task.
    # Collect all submissions, tagged by batch
    all_subs_by_batch = {}
    for task in tasks:
        target_file = task.get('target_file', '')
        all_students = collect_submissions(
            submissions_dir, target_file, student_prefix=None, max_students=100
        )
        for s in all_students:
            sid = s['student']
            batch = sid[:3]  # 'q1-', 'q2-', 'q3-'
            if batch not in all_subs_by_batch:
                all_subs_by_batch[batch] = []
            # avoid duplicate students
            if not any(x['student'] == sid for x in all_subs_by_batch[batch]):
                all_subs_by_batch[batch].append(s)

    for fcc_round in range(1, MAX_FCC_ITERATIONS + 1):
        print(f'\n  --- Round {fcc_round} ---')
        # Show per-batch coverage (each batch only checked against its task's FS)
        for batch, subs in sorted(all_subs_by_batch.items()):
            matched_task = BATCH_TASK_MAP.get(batch)
            if not matched_task:
                continue
            batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
            print(f'\n  {batch} -> {matched_task} ({len(subs)} students)')
            print(format_coverage_report(batch_cov))

        # Save per-batch coverage
        for batch, subs in sorted(all_subs_by_batch.items()):
            matched_task = BATCH_TASK_MAP.get(batch)
            if not matched_task: continue
            batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
            cov_path = os.path.join(BASE_DIR, 'output', question_id,
                                    f'coverage_r{fcc_round}_{batch}.json')
            os.makedirs(os.path.dirname(cov_path), exist_ok=True)
            with open(cov_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'per_criterion': batch_cov['per_criterion'],
                    'matrix': batch_cov['matrix'],
                }, f, indent=2)

        # Find gaps per batch (each batch only checked against its task's FS)
        all_gaps = {}
        for batch, subs in all_subs_by_batch.items():
            matched_task = BATCH_TASK_MAP.get(batch)
            if not matched_task:
                continue
            task_fs = [fs for fs in all_fs if fs.get('task') == matched_task]
            batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
            batch_gaps = find_gaps(batch_cov, subs, task_fs, min_gap_size=MIN_GAP_SIZE)
            for c, g in batch_gaps.items():
                all_gaps[c] = g

        if not all_gaps:
            print(f'  No significant gaps (>={MIN_GAP_SIZE} students) -- converged.')
            break

        print(f'\n  Gaps to fill (>={MIN_GAP_SIZE} students per criterion):')
        for criterion, gap_students in sorted(all_gaps.items()):
            print(f'    {criterion}: {len(gap_students)} students uncovered')

        # Group gaps by task and fill with batched multi-criterion prompts.
        # This replaces the old per-criterion-per-part loop, reducing API calls
        # from ~20 to ~3 per round (one per task).
        from coverage import build_multi_criterion_supplement_prompt

        new_fs_count = 0
        discarded_fcc_fs: list[dict] = []

        # Group all_gaps by task
        gaps_by_task: dict[str, dict[str, list[dict]]] = defaultdict(dict)
        for criterion, gap_students in all_gaps.items():
            m = re.search(r'\d+', str(criterion))
            crit_num = m.group() if m else '1'
            parent_task = f'Task{crit_num}'
            gaps_by_task[parent_task][criterion] = gap_students

        for parent_task, task_gaps in sorted(gaps_by_task.items()):
            parent_file = 'unknown'
            for task in tasks:
                if task['id'] == parent_task:
                    parent_file = task.get('target_file', 'unknown')
                    break

            total_gap_pairs = sum(len(v) for v in task_gaps.values())
            print(f'\n    --- {parent_task}: {len(task_gaps)} gap criteria, '
                  f'{total_gap_pairs} student-criterion pairs ---')

            prompt = build_multi_criterion_supplement_prompt(
                task_gaps, all_fs, parent_task, parent_file,
                max_students_per_criterion=5
            )
            print(f'    Calling AI ({len(prompt)} chars)...', end=' ')

            response = None
            data = None
            for attempt in range(2):
                response = call_deepseek(SYSTEM_PROMPT, prompt)
                if not response:
                    break
                try:
                    data = extract_json(response)
                    break
                except Exception:
                    if attempt == 0:
                        try:
                            fixed = _repair_json(response)
                            data = extract_json(fixed)
                            break
                        except Exception:
                            continue

            if data is None:
                print('FAILED (parse error)')
                continue

            new_fs = data.get('supplement_fs', [])
            if not new_fs:
                print('no FS returned')
                continue

            validated = 0
            for fs in new_fs:
                regex = fs.get('regex')
                criterion = fs.get('criterion', '')
                if not regex:
                    continue

                # Validate against gap students for this criterion
                gap_students_for_crit = task_gaps.get(criterion, [])
                if not gap_students_for_crit:
                    gap_students_for_crit = list(task_gaps.values())[0] if task_gaps else []

                matched_any = False
                for gs in gap_students_for_crit[:8]:
                    try:
                        flags = 0
                        if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
                        if 'DOTALL' in fs.get('regex_flags', ''): flags |= re.DOTALL
                        if re.search(regex, gs['code'], flags):
                            matched_any = True
                            break
                    except:
                        pass

                if not matched_any:
                    discarded_fcc_fs.append({
                        'fs_name': fs.get('name', '?'),
                        'regex': fs.get('regex', 'null')[:120],
                        'reason': 'regex_did_not_match_gap_students',
                        'criterion': criterion,
                        'round': fcc_round,
                    })
                    continue

                fs.setdefault('task', parent_task)
                fs.setdefault('files', [parent_file])
                fs.setdefault('auto_generated', True)
                fs.pop('marks', None)
                m = re.search(r'\d+', str(criterion))
                num = m.group() if m else '0'
                fs_id_counter.setdefault(num, 0)
                fs_id_counter[num] += 1
                fs['id'] = f'FS{num}.{fs_id_counter[num]}'
                all_fs.append(fs)
                new_fs_count += 1
                validated += 1

            print(f'added {validated}/{len(new_fs)} FS')

            # Also save discarded for this task
            task_discarded = [d for d in discarded_fcc_fs if d['criterion'] in task_gaps]
            if task_discarded:
                disc_path = os.path.join(BASE_DIR, 'output', question_id,
                                         f'discarded_fs_fcc_r{fcc_round}_{parent_task}.json')
                os.makedirs(os.path.dirname(disc_path), exist_ok=True)
                with open(disc_path, 'w', encoding='utf-8') as f:
                    json.dump({'round': fcc_round, 'discarded': task_discarded}, f, indent=2)

        if new_fs_count == 0:
            print(f'  No new FS generated -- converged.')
            break

        print(f'  Round {fcc_round} added {new_fs_count} FS total')

        # P2-2: save discarded FCC FS log
        if discarded_fcc_fs:
            fcc_disc_path = os.path.join(BASE_DIR, 'output', question_id,
                                         f'discarded_fs_fcc_r{fcc_round}.json')
            os.makedirs(os.path.dirname(fcc_disc_path), exist_ok=True)
            with open(fcc_disc_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'round': fcc_round,
                    'discarded_count': len(discarded_fcc_fs),
                    'discarded': discarded_fcc_fs,
                }, f, indent=2, ensure_ascii=False)
            print(f'    Logged {len(discarded_fcc_fs)} discarded FCC FS to {fcc_disc_path}')

    # ================================================================
    # Phase 2.9: Verification & Auto-Fix (TAFFIES FCC-aligned)
    # ================================================================
    print(f'\n{"=" * 60}')
    print('  Phase 2.9: Verification')
    print('=' * 60)

    # Collect all submissions flat (used by spot-check and comment detection)
    all_subs_flat: list[dict] = []
    seen_sids: set[str] = set()
    for batch_subs in all_subs_by_batch.values():
        for s in batch_subs:
            if s['student'] not in seen_sids:
                seen_sids.add(s['student'])
                all_subs_flat.append(s)

    # Collect reference code
    all_ref_code = ''
    for task in tasks:
        ref_files = task.get('reference_files', [])
        for rf in ref_files:
            for root, dirs, files in os.walk(ref_dir):
                if rf in files:
                    all_ref_code += read_file(os.path.join(root, rf)) + '\n'

    # --- 1. Comment false-positive check (file-level) ---
    print('\n  --- Comment False-Positive Check ---')
    from comment_stripper import PythonCommentStripper
    _stripper = PythonCommentStripper()
    removed_comment_fs: list[str] = []
    for fs in all_fs:
        regex = fs.get('regex')
        if not regex:
            continue
        flags = 0
        if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
        if 'DOTALL' in fs.get('regex_flags', ''): flags |= re.DOTALL
        # Check against first 10 students
        comment_only_count = 0
        for sub in all_subs_flat[:10]:
            code = sub.get('code', '')
            if not code.strip():
                continue
            try:
                match_full = re.search(regex, code, flags)
            except re.error:
                continue
            if not match_full:
                continue
            # Check if match still exists after stripping comments
            stripped_code = '\n'.join(
                text for _ln, text, is_sig in _stripper.strip(code) if is_sig
            )
            try:
                match_stripped = re.search(regex, stripped_code, flags)
            except re.error:
                match_stripped = None
            if not match_stripped:
                comment_only_count += 1
        if comment_only_count >= 2:
            removed_comment_fs.append(fs['id'])

    if removed_comment_fs:
        print(f'  Removing {len(removed_comment_fs)} comment-only FS')
    else:
        print('  PASS: No comment-only matches detected.')

    # --- 2. Negative FS vs. reference cross-check ---
    # Upgrade: negative FS without (?!...) or (?<!...) that match reference
    # are FALSE NEGATIVES by construction — their regex is a pure positive
    # match that would flag correct code. These are now REMOVED, not just
    # warned. Negative FS WITH negative assertions that still match reference
    # are kept with _warn_ref_match (may be legitimate edge cases).
    print('\n  --- Negative FS Reference Cross-Check ---')
    negative_fs = [fs for fs in all_fs if fs.get('fs_type') == 'negative']
    bad_negatives: list[str] = []        # has negation assertion — keep, warn
    false_negative_ids: list[str] = []   # no negation assertion — remove
    if all_ref_code:
        for fs in negative_fs:
            regex = fs.get('regex')
            if not regex:
                continue
            flags = 0
            fs_flags = fs.get('regex_flags', '')
            if 'IGNORECASE' in fs_flags: flags |= re.IGNORECASE
            if 'DOTALL' in fs_flags: flags |= re.DOTALL
            try:
                if re.search(regex, all_ref_code, flags):
                    has_neg_assertion = '(?!' in regex or '(?<!' in regex
                    if not has_neg_assertion:
                        # Pure positive match on a negative FS = false negative
                        false_negative_ids.append(fs['id'])
                    else:
                        bad_negatives.append(fs['id'])
            except re.error:
                pass
    if false_negative_ids:
        print(f'  REMOVING {len(false_negative_ids)} false-negative FS '
              f'(no assertion, match reference):')
        for fs_id in false_negative_ids[:10]:
            print(f'    {fs_id}')
    if bad_negatives:
        print(f'  WARNING: {len(bad_negatives)} negative FS match reference '
              f'(has assertion, kept with _warn_ref_match):')
        for fs_id in bad_negatives[:5]:
            print(f'    {fs_id}')
    if not false_negative_ids and not bad_negatives:
        print('  PASS: All negative FS correctly do not match reference code.')

    # --- 3. AI spot-check ---
    ai_issues: list[dict] = []
    print('\n  --- AI Spot-Check ---')
    # Build sample pairs
    import random
    pairs: list[dict] = []
    for sub in all_subs_flat[:20]:
        code = sub.get('code', '')
        if not code.strip():
            continue
        for fs in all_fs:
            regex = fs.get('regex')
            if not regex:
                continue
            flags = 0
            if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
            try:
                match = re.search(regex, code, flags)
                if match:
                    pairs.append({
                        'fs_id': fs.get('id', '?'),
                        'criterion': fs.get('criterion', '?'),
                        'fs_type': fs.get('fs_type', '?'),
                        'student': sub['student'],
                        'matched_text': match.group()[:200],
                        'regex': regex,
                    })
            except re.error:
                pass

    if pairs:
        sample = random.sample(pairs, min(5, len(pairs)))
        sample_text = '\n\n'.join(
            f"FS: {p['fs_id']} (criterion: {p['criterion']}, type: {p['fs_type']})\n"
            f"Regex: {p['regex']}\n"
            f"Student: {p['student']}\n"
            f"Matched: `{p['matched_text'][:150]}`"
            for i, p in enumerate(sample)
        )
        spot_prompt = f"""## AI Spot-Check
Verify whether these FS matches are correct.

{sample_text}

For each spot check, determine:
1. Is the matched code GENUINELY related to the criterion?
2. Is the classification (positive/negative) correct?
3. Would the feedback be appropriate for this code?

Output ONLY JSON:
{{"issues": [
  {{"fs_id": "...", "student": "...", "problem": "...", "severity": "high|medium|low"}}
]}}"""
        spot_response = call_deepseek(SYSTEM_PROMPT, spot_prompt)
        if spot_response:
            try:
                spot_data = extract_json(spot_response)
                ai_issues = spot_data.get('issues', [])
                if ai_issues:
                    print(f'  AI flagged {len(ai_issues)} potential issues')
            except Exception:
                print('  AI spot-check: could not parse response.')
        else:
            print('  AI spot-check: API call failed.')
    else:
        print('  AI spot-check: no match pairs available to sample.')

    # --- 4. Auto-Fix: remove problematic FS ---
    # Removes:
    #   - Comment-only FS (match comments not code)
    #   - False-negative FS (negative with no assertion, matches reference)
    #   - AI-flagged high-severity issues
    # Negative FS WITH assertions that match reference are KEPT with
    # _warn_ref_match — they may match legitimate student mistakes that
    # happen to share substrings with reference code.
    print('\n  --- Auto-Fix: Removing Problematic FS ---')
    remove_ids: set[str] = set(removed_comment_fs)
    remove_ids.update(false_negative_ids)
    for issue in ai_issues:
        if issue.get('severity') == 'high':
            remove_ids.add(issue.get('fs_id', ''))

    # Mark (but don't delete) negative FS that match reference
    for fid in bad_negatives:
        for fs in all_fs:
            if fs.get('id') == fid:
                fs['_warn_ref_match'] = True
                break

    all_fs = [fs for fs in all_fs if fs.get('id', '') not in remove_ids]
    removed_comment = len([fid for fid in remove_ids if fid in removed_comment_fs])
    removed_fn = len([fid for fid in remove_ids if fid in false_negative_ids])
    removed_ai = len(remove_ids) - removed_comment - removed_fn
    print(f'  Removed: {len(remove_ids)} FS ({removed_comment} comment-only, '
          f'{removed_fn} false-negative, {removed_ai} AI-flagged), '
          f'Kept: {len(all_fs)} FS')
    if bad_negatives:
        print(f'  Warning (kept): {len(bad_negatives)} negative FS match reference '
              f'(have assertion — may be valid, review recommended)')
    for fid in sorted(remove_ids)[:10]:
        if fid in removed_comment_fs:
            reason = 'comment-only'
        elif fid in false_negative_ids:
            reason = 'false-negative (no assertion)'
        else:
            reason = 'AI-flagged-high'
        print(f'    REMOVED {fid}: {reason}')

    # ================================================================
    # 4.5: FS Quality Gate — deterministic post-generation checks
    # These run WITHOUT AI calls. They detect and fix regex patterns
    # that would cause scoring errors: duplicates, contradictions,
    # broad positives, structural failures, and feedback mismatches.
    # ================================================================
    quality_report: dict[str, Any] = {
        'false_negatives_removed': list(false_negative_ids),
        'duplicates_merged': {},
        'contradictory_criteria': [],
        'fs_pair_overlaps': [],
        'broad_positives_flagged': [],
        'narrow_positives': [],
        'structurally_broken': [],
        'feedback_regex_mismatches': [],
        'template_matches': [],
        'literal_fallback_fs': [],
        'variable_names_generalized': [],
        'scoring_readiness': {'reliable': [], 'unreliable': [], 'needs_review': []},
    }

    # --- 4.5pre: Variable Name Generalisation ---
    # AI generates regexes with hardcoded variable names because it sees
    # concrete student code (e.g. "stats = cursor.fetchall()") and copies
    # the literal identifiers. This step uses tokenize on actual matched
    # code to identify which tokens ARE variable names, then replaces
    # them with \w+ in the regex. Runs BEFORE other checks since it
    # modifies regex content.
    print('\n  --- Quality Gate: Variable Name Generalisation ---')
    import tokenize as _tokenize, io as _io

    PY_KEYWORDS = frozenset({
        'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'try',
        'except', 'finally', 'with', 'as', 'import', 'from', 'class',
        'pass', 'break', 'continue', 'yield', 'raise', 'assert', 'del',
        'global', 'nonlocal', 'lambda', 'and', 'or', 'not', 'in', 'is',
        'True', 'False', 'None', 'self', 'cls',
    })
    SQL_KEYWORDS = frozenset({
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
        'ON', 'GROUP', 'BY', 'ORDER', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'AS', 'IN', 'OR', 'IGNORE', 'INTO', 'VALUES', 'SET', 'CREATE',
        'TABLE', 'EXISTS', 'NOT', 'AND', 'LIKE', 'BETWEEN', 'HAVING',
        'DISTINCT', 'UNION', 'ALL', 'LIMIT', 'OFFSET', 'INNER', 'OUTER',
        'LEFT', 'RIGHT', 'CROSS', 'FULL', 'NATURAL', 'USING', 'ASC',
        'DESC', 'NULL', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'INDEX',
        'UNIQUE', 'CHECK', 'DEFAULT', 'CASCADE', 'OR', 'REPLACE',
        'select', 'insert', 'update', 'delete', 'from', 'where', 'join',
        'on', 'group', 'by', 'order', 'count', 'sum', 'avg', 'min', 'max',
    })
    KNOWN_APIS = frozenset({
        'app', 'flask', 'Flask', 'request', 'session', 'g', 'redirect',
        'url_for', 'render_template', 'flash', 'make_response', 'jsonify',
        'cursor', 'conn', 'connection', 'db', 'sqlite3', 'sqlite',
        'csv', 'DictReader', 'reader', 'writer', 'DictWriter',
        'Path', 'open', 'file', 'os', 'sys', 'json', 're', 'datetime',
        'timedelta', 'date', 'Enum', 'Base', 'Exception', 'ValueError',
        'TypeError', 'KeyError', 'IndexError', 'FileNotFoundError',
        'OperationalError', 'IntegrityError', 'ProgrammingError',
        'fetchall', 'fetchone', 'fetchmany', 'execute', 'executemany',
        'commit', 'rollback', 'close', 'connect', 'cursor_factory',
        'row_factory', 'rowcount', 'lastrowid', 'description',
        'get', 'post', 'put', 'delete', 'patch', 'route', 'errorhandler',
        'before_request', 'after_request', 'teardown_request',
        'send_file', 'send_from_directory', 'abort',
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
        'bytes', 'bytearray', 'frozenset', 'range', 'enumerate', 'zip',
        'map', 'filter', 'sorted', 'reversed', 'iter', 'next', 'len',
        'print', 'input', 'isinstance', 'issubclass', 'hasattr', 'getattr',
        'setattr', 'delattr', 'type', 'super', 'object', 'property',
        'staticmethod', 'classmethod', 'any', 'all', 'abs', 'round',
        'min', 'max', 'sum', 'pow', 'divmod', 'chr', 'ord', 'hex', 'oct',
        'bin', 'format', 'repr', 'ascii', 'eval', 'exec', 'compile',
        '__name__', '__main__', '__file__', '__init__', '__str__',
        '__repr__', '__dict__', '__class__', '__doc__', '__module__',
        'Playlist', 'Track', 'Genre', 'PlaylistTrack', 'PlaylistId',
        'TrackId', 'GenreId', 'Name', 'Milliseconds', 'UnitPrice',
        'Composer', 'AlbumId', 'MediaTypeId', 'Bytes', 'BillingCountry',
        'BillingCity', 'BillingState', 'BillingAddress', 'BillingPostalCode',
        'Total', 'InvoiceId', 'InvoiceLineId', 'CustomerId', 'EmployeeId',
        'playlist', 'track', 'genre', 'playlist_track', 'playlists',
        'tracks', 'genres', 'iMusic', 'statistics',
        'methods', 'delimiter', 'newline', 'encoding', 'errors',
        'debug', 'port', 'host', 'secret_key',
    })
    BLOCKLIST = PY_KEYWORDS | SQL_KEYWORDS | KNOWN_APIS

    # --- Add template function names to BLOCKLIST ---
    # Prevents function names like rename_playlist, create_playlist etc.
    # from being replaced with \w+, which would make negative FS match
    # every function in the file (including correct ones from other tasks).
    template_func_names: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(os.path.join(BASE_DIR, 'question')):
        for fn in filenames:
            if fn.endswith('.py'):
                code = read_file(os.path.join(dirpath, fn))
                template_func_names.update(re.findall(r'def\s+(\w+)\s*\(', code))
    BLOCKLIST = BLOCKLIST | template_func_names
    if template_func_names:
        print(f'  Added {len(template_func_names)} template function names to BLOCKLIST: '
              f'{sorted(template_func_names)}')

    # Also collect function def names from student code (these are NEVER variables)
    all_func_def_names: set[str] = set()
    for sub in (all_subs_flat[:10] if all_subs_flat else []):
        all_func_def_names.update(re.findall(r'def\s+(\w+)\s*\(', sub.get('code', '')))
    # Path, DB_FILE, route, etc. are also not variables — they're constants/configs
    CONSTANT_LIKE = frozenset({
        'DB_FILE', 'BASE_DIR', 'UPLOAD_FOLDER', 'Path', 'app',
        'playlist_tracks_file', 'update_playlist_tracks',
    })
    BLOCKLIST = BLOCKLIST | all_func_def_names | CONSTANT_LIKE

    generalized_count = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex or len(regex) < 10:
            continue

        # Find student code that matches this FS
        fs_task = fs.get('task', '')
        task_samples = []
        for batch, subs in all_subs_by_batch.items():
            if BATCH_TASK_MAP.get(batch) == fs_task:
                task_samples = subs
                break
        if not task_samples:
            task_samples = all_subs_flat

        # Skip if regex already uses \w+ extensively (already generalized)
        if regex.count('\\w+') >= 3 or regex.count('\\\\w\\+') >= 3:
            continue

        matched_snippets: list[str] = []
        flags = 0
        if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
        if 'DOTALL' in fs.get('regex_flags', ''): flags |= re.DOTALL
        if 'MULTILINE' in fs.get('regex_flags', ''): flags |= re.MULTILINE
        try:
            compiled = re.compile(regex, flags)
        except re.error:
            continue

        # Reduced sample sizes for performance (was 15/5, now 5/2)
        for sub in task_samples[:5]:
            code = sub.get('code', '')
            if not code.strip():
                continue
            try:
                match = compiled.search(code)
                if match:
                    matched_snippets.append(match.group())
                    if len(matched_snippets) >= 2:
                        break
            except re.error:
                continue

        # Strategy A: if we have >=1 matched sample, extract variable names
        replaceable: set[str] = set()

        if len(matched_snippets) >= 1:
            # --- Strategy A: tokenize-based extraction ---
            var_candidates: dict[str, int] = {}
            for snippet in matched_snippets:
                seen_in_snippet: set[str] = set()
                try:
                    tokens = _tokenize.generate_tokens(
                        _io.StringIO(snippet).readline)
                    for tok in tokens:
                        if tok.type == _tokenize.NAME and len(tok.string) >= 2:
                            seen_in_snippet.add(tok.string)
                except (_tokenize.TokenError, IndentationError,
                        SyntaxError, Exception):
                    continue
                for name in seen_in_snippet:
                    var_candidates[name] = var_candidates.get(name, 0) + 1
            for name, count in var_candidates.items():
                if count >= 2 and name not in BLOCKLIST:
                    replaceable.add(name)
        else:
            # --- Strategy B: regex-text analysis ---
            # Extract bare identifier-like tokens from the regex string.
            # These are sequences of [a-zA-Z_][a-zA-Z0-9_]+ that are NOT:
            #   - preceded by \ (part of \s, \w, etc.)
            #   - inside [...] character classes
            #   - inside the regex pattern '\\w+' itself
            #   - in any blocklist
            # Strip regex constructs to find literal identifiers
            cleaned = regex
            # Remove known regex patterns that contain identifiers
            cleaned = re.sub(r'\\[swdSWD]\s*[\*\+?]?(?:\{[^}]*\})?', ' ', cleaned)
            cleaned = re.sub(r'\\[.?*+()\[\]{}|^$]', ' ', cleaned)
            cleaned = re.sub(r'\[[^\]]*\]', ' ', cleaned)
            cleaned = re.sub(r'\([^)]*\)', ' ', cleaned)  # groups
            # Now extract identifier-like tokens
            raw_names = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b', cleaned))
            for name in raw_names:
                if name not in BLOCKLIST:
                    # Verify this isn't part of a regex escape by checking
                    # the original regex: ensure it appears as a bare word
                    if re.search(r'(?<!\\)\b' + re.escape(name) + r'\b', regex):
                        replaceable.add(name)

        if not replaceable:
            continue

        # CRITICAL: Remove function definition names from replaceable.
        # A name that appears as "def name(" in student code is a function
        # name, NOT a variable. Replacing it with \w+ makes the regex match
        # EVERY function, causing false negatives/positives.
        func_names_in_samples: set[str] = set()
        for sub in (task_samples[:5] if task_samples else all_subs_flat[:5]):
            func_names_in_samples.update(
                re.findall(r'def\s+(\w+)\s*\(', sub.get('code', ''))
            )
        if func_names_in_samples:
            removed_funcs = replaceable & func_names_in_samples
            if removed_funcs:
                replaceable -= removed_funcs
                # Only log if something was actually removed
                pass  # silent — these were wrongly identified as variables

        if not replaceable:
            continue

        # Replace hardcoded variable names with \w+ in the regex
        # Sort by length (longest first) to avoid partial replacements
        # Use negative lookbehind for \ to avoid matching inside \s, \w, etc.
        modified_regex = regex
        for name in sorted(replaceable, key=len, reverse=True):
            # Only replace if the name appears as a standalone identifier
            # (not preceded by \) and surrounded by word boundaries
            pattern = r'(?<!\\)\b' + re.escape(name) + r'\b'
            replacement = r'\\w+'
            new_regex = re.sub(pattern, replacement, modified_regex)
            if new_regex != modified_regex:
                modified_regex = new_regex

        if modified_regex != regex:
            fs['regex'] = modified_regex
            fs['_generalized_vars'] = sorted(replaceable)
            generalized_count += 1
            quality_report['variable_names_generalized'].append({
                'fs_id': fs.get('id', '?'),
                'criterion': fs.get('criterion', '?'),
                'replaced': sorted(replaceable),
            })

    if generalized_count:
        print(f'  GENERALIZED {generalized_count} FS: replaced hardcoded '
              f'variable names with \\\\w+')
    else:
        print('  NOTE: No variable names to generalise '
              '(all FS already use \\\\w+ or have too few matches)')

    # --- 4.5a: Duplicate FS Merge ---
    # Merge FS with identical regex (after whitespace normalisation)
    # within the same criterion. Keeps the most descriptive name.
    print('\n  --- Quality Gate: Duplicate FS Merge ---')
    from collections import defaultdict as _defaultdict
    dup_groups: dict[tuple[str, str], list[dict]] = _defaultdict(list)
    for fs in all_fs:
        # Aggressive normalisation: collapse all whitespace patterns, collapse
        # all digit patterns, remove cosmetic differences that don't change
        # what the regex actually matches.
        norm = re.sub(r'\\s\+|\\s\*|\\s\{1,\}|\\s\?', r'\\s*', fs.get('regex', ''))
        norm = re.sub(r'\\d\+|\\d\*|\\d\{1,\}|\\d\?', r'\\d*', norm)
        norm = re.sub(r'\\w\+|\\w\*|\\w\{1,\}|\\w\?', r'\\w*', norm)
        norm = re.sub(r'\s+', '', norm)  # strip ALL whitespace
        key = (fs.get('criterion', '?'), norm)
        dup_groups[key].append(fs)

    dup_merge_count = 0
    for (criterion, _norm), group in dup_groups.items():
        if len(group) < 2:
            continue
        # Keep the FS with the longest (most descriptive) name
        group.sort(key=lambda f: len(f.get('name', '')), reverse=True)
        keeper = group[0]
        for dup in group[1:]:
            keeper_id = keeper.get('id', '?')
            dup_id = dup.get('id', '?')
            dup['_duplicate_of'] = keeper_id
            # Merge feedback if the duplicate adds new information
            if len(dup.get('feedback', '')) > len(keeper.get('feedback', '')):
                keeper['feedback'] = dup['feedback']
            quality_report['duplicates_merged'][dup_id] = keeper_id
            dup_merge_count += 1

    if dup_merge_count:
        print(f'  Merged {dup_merge_count} duplicate FS '
              f'(kept most descriptive name per group)')
        # Note: duplicates are soft-marked (_duplicate_of), not deleted,
        # to preserve coverage. Scoring logic should skip _duplicate_of FS.
    else:
        print('  PASS: No duplicate FS found.')

    # --- 4.5b: Positive-Negative Contradiction Detection ---
    # A criterion is UNRELIABLE if it has both positive FS and
    # negative FS whose regex matches reference code (including
    # those we just removed — their very existence signals the AI
    # couldn't distinguish good from bad for this criterion).
    print('\n  --- Quality Gate: Pos-Neg Contradiction Detection ---')
    criteria_with_false_neg: set[str] = set()
    # Check removed false negatives
    all_bad_for_criterion: dict[str, list[str]] = _defaultdict(list)
    for fid in false_negative_ids:
        for fs in all_fs:
            if fs.get('id') == fid:
                all_bad_for_criterion[fs.get('criterion', '?')].append(fid)
                break
    # Also check remaining bad_negatives
    for fid in bad_negatives:
        for fs in all_fs:
            if fs.get('id') == fid:
                all_bad_for_criterion[fs.get('criterion', '?')].append(fid)
                break

    contradictory: list[str] = []
    for criterion, bad_ids in sorted(all_bad_for_criterion.items()):
        has_positive = any(
            fs.get('criterion') == criterion and fs.get('fs_type') == 'positive'
            for fs in all_fs
        )
        if has_positive:
            contradictory.append(criterion)
            quality_report['contradictory_criteria'].append({
                'criterion': criterion,
                'bad_negative_ids': bad_ids,
                'issue': 'Positive FS exist but negative FS match correct '
                         'reference code — scoring for this criterion is unreliable.',
            })

    if contradictory:
        print(f'  UNRELIABLE: {len(contradictory)} criteria have pos-neg '
              f'contradictions: {", ".join(contradictory)}')
    else:
        print('  PASS: No pos-neg contradictions detected.')

    # --- 4.5b2: FS-Pair Level Pos-Neg Overlap Detection ---
    # For each criterion, test each (positive, negative) FS pair against
    # reference code. If both match → the same correct code triggers both
    # a reward and a penalty → direct scoring contradiction at FS level.
    print('\n  --- Quality Gate: FS-Pair Overlap Detection ---')
    fs_pair_overlaps: list[dict] = []
    criteria = sorted(set(fs.get('criterion', '?') for fs in all_fs
                          if fs.get('criterion', '').startswith('RQ')))
    for crit in criteria:
        pos_fs = [fs for fs in all_fs
                  if fs.get('criterion') == crit and fs.get('fs_type') == 'positive']
        neg_fs = [fs for fs in all_fs
                  if fs.get('criterion') == crit and fs.get('fs_type') == 'negative']
        for pfs in pos_fs:
            for nfs in neg_fs:
                pregex = pfs.get('regex', '')
                nregex = nfs.get('regex', '')
                if not pregex or not nregex:
                    continue
                # Test: does reference code match BOTH regexes?
                pflags = 0
                nflags = 0
                if 'IGNORECASE' in pfs.get('regex_flags', ''): pflags |= re.IGNORECASE
                if 'DOTALL' in pfs.get('regex_flags', ''): pflags |= re.DOTALL
                if 'MULTILINE' in pfs.get('regex_flags', ''): pflags |= re.MULTILINE
                if 'IGNORECASE' in nfs.get('regex_flags', ''): nflags |= re.IGNORECASE
                if 'DOTALL' in nfs.get('regex_flags', ''): nflags |= re.DOTALL
                if 'MULTILINE' in nfs.get('regex_flags', ''): nflags |= re.MULTILINE
                try:
                    p_match = re.search(pregex, all_ref_code, pflags) if all_ref_code else None
                    n_match = re.search(nregex, all_ref_code, nflags) if all_ref_code else None
                except re.error:
                    continue
                if p_match and n_match:
                    fs_pair_overlaps.append({
                        'criterion': crit,
                        'positive_fs': pfs.get('id', '?'),
                        'negative_fs': nfs.get('id', '?'),
                        'positive_matched': p_match.group()[:80],
                        'negative_matched': n_match.group()[:80],
                    })
                    pfs['_warn_overlaps_with_negative'] = True
                    nfs['_warn_overlaps_with_positive'] = True

    if fs_pair_overlaps:
        print(f'  FS-LEVEL OVERLAP: {len(fs_pair_overlaps)} (pos,neg) pairs '
              f'both match reference code:')
        for ov in fs_pair_overlaps[:8]:
            print(f'    {ov["criterion"]}: +{ov["positive_fs"]} vs '
                  f'-{ov["negative_fs"]}')
        quality_report['fs_pair_overlaps'] = fs_pair_overlaps
    else:
        print('  PASS: No FS-pair overlaps detected.')

    # --- 4.5b3: Auto-Fix Negative FS Matching Reference Code ---
    # Negative FS that match reference solutions are FALSE PENALTIES.
    # Attempt deterministic fixes before giving up.
    print('\n  --- Quality Gate: Negative FS Auto-Fix ---')
    neg_fix_count = 0
    neg_unfixable: list[str] = []
    for fs in all_fs:
        if not (fs.get('_warn_ref_match') and fs.get('fs_type') == 'negative'):
            continue
        regex = fs.get('regex', '')
        fid = fs.get('id', '?')
        fixed = None

        # Strategy 1: If regex uses def\s+\w+ (generalized function name),
        # try to find the intended function name from the FS name or feedback
        if re.search(r'def\\s\+\\w\+', regex):
            # Extract function names mentioned in the FS name/description/feedback
            mentioned_funcs = set()
            for field in ['name', 'description', 'feedback']:
                text = fs.get(field, '')
                mentioned_funcs.update(re.findall(
                    r'\b(update_playlist_tracks|rename_playlist|create_playlist|'
                    r'delete_playlist|add_tracks_by_genre|remove_tracks_by_genre|'
                    r'statistics|get_all_genres|get_statistics|playlists|'
                    r'get_all_playlists)\b', text
                ))
            if mentioned_funcs:
                # Replace \w+ after def with the specific function name
                func_alt = '|'.join(sorted(mentioned_funcs))
                fixed = re.sub(
                    r'def\\s\+\\w\+',
                    f'def\\\\s+(?:{func_alt})',
                    regex
                )
                # Also fix the generalized vars record
                fix_name = next(iter(mentioned_funcs))
                if fixed != regex:
                    fs['_auto_fixed_func_name'] = True

        # Strategy 2: For negative FS that are structurally too broad
        # (match any function), try narrowing by adding the function name
        if not fixed and re.search(r'def\\s\+\\w\+\\s\*\\\(', regex):
            # Too generic — can't fix automatically
            neg_unfixable.append(fid)
            continue

        if fixed and fixed != regex:
            # Verify the fixed regex no longer matches reference code
            try:
                flags = 0
                if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
                if 'DOTALL' in fs.get('regex_flags', ''): flags |= re.DOTALL
                if 'MULTILINE' in fs.get('regex_flags', ''): flags |= re.MULTILINE
                if not re.search(fixed, all_ref_code, flags):
                    fs['regex'] = fixed
                    fs.pop('_warn_ref_match', None)
                    fs['_auto_fixed_negative'] = True
                    neg_fix_count += 1
                    print(f'  FIXED {fid}: narrowed function scope for negative FS')
                else:
                    neg_unfixable.append(fid)
            except re.error:
                neg_unfixable.append(fid)
        elif not fixed:
            neg_unfixable.append(fid)

    if neg_fix_count:
        print(f'  Auto-fixed {neg_fix_count} negative FS (narrowed function scope)')
    if neg_unfixable:
        print(f'  UNFIXABLE ({len(neg_unfixable)}): negative FS too broad, '
              f'will keep _warn_ref_match and reduced weight')
        quality_report['unfixable_negatives'] = neg_unfixable
    if not neg_fix_count and not neg_unfixable:
        print('  PASS: No negative FS needed fixing.')

    # --- 4.5c: Broad Positive FS Detection ---
    # Positive FS whose regex is too broad — would give credit for
    # incomplete or sub-optimal code.
    print('\n  --- Quality Gate: Broad Positive FS Detection ---')
    BROAD_POSITIVE_PATTERNS = [
        # return True|False without try-except context
        (r'return\\s\+True\|return\\s\+False', 'matches any return statement without checking try-except'),
        # GET-only route marked positive when POST is also required
        (r"methods\\s\*=\\s\*\[\\s\*\['\"]GET['\"]\\s\*\]", 'GET-only route marked positive (POST required)'),
        # SELECT * marked positive when specific columns expected
        (r'SELECT\\s\+\\\*\\s\+FROM', 'SELECT * marked positive (specific columns expected)'),
        # ORDER BY ? ? — parameterized column/direction does NOT work in SQLite
        # (placeholders become literal strings, not identifiers)
        (r'ORDER\\s\+BY\\s\+\\\?\\s\+\\\?', 'ORDER BY ? ? — parameterized ORDER BY does not work in SQLite'),
    ]
    broad_positives: list[str] = []
    for fs in all_fs:
        if fs.get('fs_type') != 'positive':
            continue
        regex = fs.get('regex', '')
        for pattern, desc in BROAD_POSITIVE_PATTERNS:
            try:
                if re.search(pattern, regex):
                    fid = fs.get('id', '?')
                    fs['_warn_broad_positive'] = True
                    broad_positives.append(fid)
                    quality_report['broad_positives_flagged'].append({
                        'fs_id': fid,
                        'criterion': fs.get('criterion', '?'),
                        'pattern': desc,
                    })
                    break
            except re.error:
                pass

    if broad_positives:
        print(f'  FLAGGED {len(broad_positives)} broad positive FS '
              f'(marked _warn_broad_positive):')
        for fid in broad_positives[:5]:
            print(f'    {fid}')
    else:
        print('  PASS: No broad positive FS detected.')

    # --- 4.5d: Regex Structural Validation ---
    # Check for: broken regex syntax, type-annotation incompatibility,
    # regex that matches nothing in any student sample.
    print('\n  --- Quality Gate: Regex Structural Validation ---')
    broken_regex: list[str] = []
    type_annotation_issues: list[str] = []
    matches_nothing: list[str] = []

    for fs in all_fs:
        regex = fs.get('regex', '')
        fid = fs.get('id', '?')
        if not regex:
            broken_regex.append(fid)
            continue

        # 1. Pre-compile check: variable-width lookbehind
        # Python's re module requires fixed-width lookbehind patterns.
        # Patterns like (?<!@app\.route[^\n]*\n) use [^\n]* which is
        # variable-width and will raise re.error at compile time.
        if re.search(r'\(\?<![^)]*(?:\*|\+|(?:\{[^}]*,\s*[^}]*\}))[^)]*\)', regex):
            broken_regex.append(fid)
            quality_report['structurally_broken'].append({
                'fs_id': fid,
                'issue': 'VARIABLE_WIDTH_LOOKBEHIND: Python re requires '
                         'fixed-width lookbehind patterns',
            })
            continue

        # 2. Compile check (catches any remaining syntax errors)
        try:
            flags = 0
            if 'IGNORECASE' in fs.get('regex_flags', ''): flags |= re.IGNORECASE
            if 'DOTALL' in fs.get('regex_flags', ''): flags |= re.DOTALL
            if 'MULTILINE' in fs.get('regex_flags', ''): flags |= re.MULTILINE
            compiled = re.compile(regex, flags)
        except re.error as e:
            broken_regex.append(fid)
            quality_report['structurally_broken'].append({
                'fs_id': fid, 'issue': f're.error: {e}',
            })
            continue

        # 3. Type annotation incompatibility check
        # Pattern "\)\s*:" fails on Python 3 signatures like
        # "def f(path: Path) -> bool:" because ")" is followed by " -> bool:"
        if re.search(r'\\\)\\s\*:', regex) and '->' not in regex:
            type_annotation_issues.append(fid)
            fs['_warn_type_annotation'] = True
            quality_report['structurally_broken'].append({
                'fs_id': fid,
                'issue': 'TYPE_ANNOTATION_INCOMPATIBLE: regex has "\)\\s*:" '
                         'which fails on "def f(...) -> type:" signatures',
            })

        # 4. Matches-nothing check: test against task-scoped student samples
        # Scope by FS task to avoid false alarms (e.g. Task 2 FS tested
        # against Task 1 students who never wrote that code).
        fs_task = fs.get('task', '')
        task_samples = []
        if fs_task and fs_task in all_subs_by_batch:
            # Find matching batch for this task
            for batch, subs in all_subs_by_batch.items():
                if BATCH_TASK_MAP.get(batch) == fs_task:
                    task_samples = subs[:5]
                    break
        if not task_samples:
            task_samples = all_subs_flat[:5]  # fallback
        if task_samples:
            matched_any = False
            for sub in task_samples:
                code = sub.get('code', '')
                if not code.strip():
                    continue
                try:
                    if compiled.search(code):
                        matched_any = True
                        break
                except re.error:
                    pass
            if not matched_any:
                matches_nothing.append(fid)
                fs['_warn_matches_nothing'] = True
                quality_report['structurally_broken'].append({
                    'fs_id': fid,
                    'issue': 'MATCHES_NOTHING: regex did not match any of '
                             'the first 5 relevant student samples',
                })

    if broken_regex:
        print(f'  BROKEN ({len(broken_regex)}): regex fails to compile — '
              f'adding to remove_ids')
        remove_ids.update(broken_regex)
    if type_annotation_issues:
        print(f'  TYPE_ANNOTATION ({len(type_annotation_issues)}): '
              f'regex has "\\)\\s*:" incompatible with "-> type:" signatures')
    if matches_nothing:
        print(f'  MATCHES_NOTHING ({len(matches_nothing)}): '
              f'regex matches zero student samples')
    # Upgrade: positive FS that match zero students are UNRELIABLE for scoring
    narrow_positives: list[str] = []
    for fid in matches_nothing:
        for fs in all_fs:
            if fs.get('id') == fid and fs.get('fs_type') == 'positive':
                meaningful = len(re.sub(r'\\[sSwWdD]', '', fs.get('regex', '')))
                if meaningful > 30:
                    narrow_positives.append(fid)
                    fs['_warn_narrow_positive'] = True
    if narrow_positives:
        print(f'  NARROW_POSITIVE ({len(narrow_positives)}): positive FS '
              f'match zero students — will never award credit')
        quality_report['narrow_positives'] = narrow_positives

    # --- 4.5d-extra: Semantic regex quality checks ---
    # Detect patterns that indicate AI prompt-following failures.
    generic_func_matches: list[str] = []
    pure_positive_negatives: list[str] = []
    for fs in all_fs:
        regex = fs.get('regex', '')
        fid = fs.get('id', '?')
        if not regex:
            continue

        # Check 1: Negative FS using def\s+\w+ — matches ALL functions
        if (fs.get('fs_type') == 'negative'
                and re.search(r'def\\s\+\\w\+', regex)
                and not re.search(r'def\\s\+\(\?:', regex)):  # allow explicit alternation
            generic_func_matches.append(fid)
            fs['_warn_generic_function_match'] = True
            quality_report['structurally_broken'].append({
                'fs_id': fid,
                'issue': 'GENERIC_FUNCTION: def\\s+\\w+ in negative FS '
                         'matches EVERY function, not just the target',
            })

        # Check 2: Negative FS with pure positive match (no assertion)
        # TYPE B negative FS (detecting "present bad pattern") uses pure positive
        # regex correctly. Only flag if the regex matches REFERENCE code
        # (meaning the "bad pattern" appears in correct solutions — it's not actually bad).
        if (fs.get('fs_type') == 'negative'
                and '(?!' not in regex
                and '(?<!' not in regex
                and 'def\\s+' not in regex[:20]):  # def-prefix negative is OK
            # Heuristic: if the regex starts with a plain keyword/function name
            # without any negative assertion, it MAY be a pure positive match
            pure_match = re.match(
                r'^(?:INSERT|SELECT|UPDATE|DELETE|CREATE|DROP|ALTER|'
                r'csv|sqlite3|import|from|@app|return|print|flash)',
                regex
            )
            if pure_match:
                # Type B check: does this regex match reference code?
                # If it does NOT match reference, it's a valid Type B negative FS
                # (detecting a genuinely bad pattern absent from correct solutions).
                matches_ref = False
                if all_ref_code:
                    try:
                        matches_ref = bool(re.search(regex, all_ref_code, re.IGNORECASE | re.DOTALL))
                    except re.error:
                        matches_ref = False
                if matches_ref:
                    # True problem: "bad pattern" that exists in reference → wrong
                    pure_positive_negatives.append(fid)
                    fs['_warn_pure_positive_negative'] = True
                    quality_report['structurally_broken'].append({
                        'fs_id': fid,
                        'issue': 'PURE_POSITIVE_NEGATIVE: negative FS matches '
                                 'reference code without negative assertion — '
                                 'the "bad pattern" exists in correct solutions',
                    })
                # else: valid Type B negative FS — pure positive regex, doesn't match ref

    if generic_func_matches:
        print(f'  GENERIC_FUNC ({len(generic_func_matches)}): negative FS '
              f'use def\\s+\\w+ — matches ALL functions:')
        for fid in generic_func_matches[:5]:
            print(f'    {fid}')
        quality_report['generic_function_matches'] = generic_func_matches
    if pure_positive_negatives:
        print(f'  PURE_POS_NEG ({len(pure_positive_negatives)}): negative FS '
              f'with no negative assertion — will match correct code:')
        for fid in pure_positive_negatives[:5]:
            print(f'    {fid}')
        quality_report['pure_positive_negatives'] = pure_positive_negatives

    if not broken_regex and not type_annotation_issues and not matches_nothing:
        print('  PASS: All regex structurally valid.')

    # --- 4.5d2: Source-Code Escape Sequence Fix ---
    # AI-generated regexes often use \t, \n, \r to match delimiter='\t'
    # etc. in Python source code. But in regex, \t matches a literal TAB
    # character (ASCII 0x09), while student .py files contain the TWO
    # characters backslash + t. This fix replaces regex \t with \\t
    # (matching literal backslash-t in source code) when it appears in
    # string-literal / delimiter contexts.
    # Also fixes type annotation incompatibility: \)\s*: -> \)\s*(?:->\s*\w+\s*)?\s*:
    print('\n  --- Quality Gate: Source-Code Escape Fix ---')
    escape_fix_count = 0
    type_anno_fix_count = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue
        modified = regex

        # Fix 1: Replace \t that should match source-code '\t' (two chars)
        # Only fix \t that appears in delimiter/split/string contexts,
        # NOT \t that is part of \t in a character class or other pattern.
        # Pattern: \t preceded by a quote char or delimiter keyword
        if re.search(r"(?:delimiter|split|replace|['\"])\s*[=:]*\s*.*\\t", modified):
            # Replace regex \t (tab char) with \\t (literal backslash-t)
            # Careful: in the regex string, '\t' is two chars: \ and t
            # We need to replace it with '\\t' which is three chars: \, \, t
            # But we only want to do this for \t that's NOT part of \s, \w, etc.
            # Strategy: find \t preceded by a printable char (not \)
            modified = re.sub(r'(?<!\\)\\t', r'\\\\t', modified)
        if modified != regex:
            escape_fix_count += 1

        # Fix 2: Type annotation incompatibility
        # Pattern "\)\s*:" fails on "def f(...) -> bool:"
        if re.search(r'\\\)\\s\*:', modified) and '->' not in modified:
            modified = re.sub(r'\\\)\\s\*:', r'\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:', modified)
            if modified != regex:
                type_anno_fix_count += 1

        # Fix 3: Variable-width lookbehind — try to rewrite as (?:^|\n) prefix
        if re.search(r'\(\?<![^)]*(?:\*|\+|(?:\{[^}]*,\s*[^}]*\}))[^)]*\)', modified):
            # These can't be auto-fixed safely; mark for removal
            fs['_warn_unfixable_lookbehind'] = True

        if modified != regex:
            fs['regex'] = modified
            fs['_source_escape_fixed'] = True

    if escape_fix_count:
        print(f'  FIXED \\t ESCAPE in {escape_fix_count} FS: '
              f'\\t -> \\\\t for source-code matching')
    if type_anno_fix_count:
        print(f'  FIXED TYPE ANNOTATION in {type_anno_fix_count} FS: '
              f'\\)\\s*: -> \\)\\s*(?:->\\s*\\w+\\s*)?\\s*:')
    if not escape_fix_count and not type_anno_fix_count:
        print('  PASS: No source-code escape issues detected.')

    # --- 4.5e: Regex-Feedback Consistency Check ---
    # Simple heuristic: if feedback mentions a specific table/function
    # name that does NOT appear in the regex, the feedback may be
    # describing a problem the regex doesn't actually detect.
    print('\n  --- Quality Gate: Regex-Feedback Consistency ---')
    mismatch_count = 0
    for fs in all_fs:
        feedback = fs.get('feedback', '')
        regex = fs.get('regex', '')
        if not feedback or not regex:
            continue
        # Extract CamelCase and snake_case identifiers from feedback
        feedback_ids = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', feedback))
        feedback_ids.update(re.findall(r'\b[a-z]+_[a-z]+(?:_[a-z]+)*\b', feedback))
        # Filter out common English words
        english_words = {'the', 'and', 'for', 'you', 'your', 'this', 'that',
                         'with', 'from', 'when', 'what', 'how', 'not', 'but',
                         'can', 'will', 'has', 'are', 'was', 'use', 'used',
                         'using', 'does', 'also', 'all', 'its', 'his', 'her',
                         'one', 'two', 'any', 'out', 'has', 'had', 'been',
                         'have', 'they', 'them', 'their', 'would', 'could',
                         'should', 'may', 'might', 'must', 'need', 'make',
                         'good', 'bad', 'well', 'way', 'just', 'more', 'some',
                         'each', 'very', 'much', 'many', 'few', 'new', 'old',
                         'now', 'then', 'here', 'there', 'which', 'into',
                         'over', 'only', 'other', 'after', 'before', 'between',
                         'same', 'such', 'both', 'still', 'while', 'during',
                         'though', 'through', 'without', 'within', 'along',
                         'also', 'even', 'yet', 'too', 'either', 'neither',
                         'whether', 'rather', 'quite', 'almost', 'enough'}
        feedback_ids -= english_words
        # Check if these identifiers appear in the regex
        for ident in feedback_ids:
            if len(ident) < 4:
                continue
            # Try to find the identifier (case-insensitive) in regex
            ident_pattern = ident.replace('_', r'[_\s]*')
            try:
                if not re.search(ident_pattern, regex, re.IGNORECASE):
                    mismatch_count += 1
                    quality_report['feedback_regex_mismatches'].append({
                        'fs_id': fs.get('id', '?'),
                        'identifier_in_feedback': ident,
                        'issue': f'Feedback mentions "{ident}" but regex '
                                 f'does not contain a matching pattern',
                    })
                    fs['_warn_feedback_mismatch'] = True
                    break  # One mismatch per FS is enough
            except re.error:
                pass

    if mismatch_count:
        print(f'  FLAGGED {mismatch_count} FS with potential '
              f'feedback-regex mismatch (marked _warn_feedback_mismatch)')
    else:
        print('  PASS: No feedback-regex mismatches detected.')

    # --- Re-apply removal after Quality Gate ---
    if broken_regex:
        all_fs = [fs for fs in all_fs if fs.get('id', '') not in remove_ids]
        print(f'\n  After Quality Gate removals: {len(all_fs)} FS remaining')

    # --- 4.5f: Template Code Match Detection ---
    # FS must only match student-written code, NOT the starter template.
    # If a regex matches unchanged template code, it gives credit for
    # code the student didn't write.
    print('\n  --- Quality Gate: Template Code Match Detection ---')
    template_code = ''
    for dirpath, _dirnames, filenames in os.walk(os.path.join(question_dir, 'code')):
        for fn in filenames:
            if fn.endswith('.py'):
                template_code += read_file(os.path.join(dirpath, fn)) + '\n'
    if not template_code:
        # Fallback: walk entire question_dir
        for dirpath, _dirnames, filenames in os.walk(question_dir):
            for fn in filenames:
                if fn.endswith('.py'):
                    template_code += read_file(os.path.join(dirpath, fn)) + '\n'

    template_matches: list[str] = []
    if template_code:
        for fs in all_fs:
            regex = fs.get('regex', '')
            if not regex:
                continue
            flags = 0
            fs_flags = fs.get('regex_flags', '')
            if 'IGNORECASE' in fs_flags: flags |= re.IGNORECASE
            if 'DOTALL' in fs_flags: flags |= re.DOTALL
            if 'MULTILINE' in fs_flags: flags |= re.MULTILINE
            try:
                if re.search(regex, template_code, flags):
                    fid = fs.get('id', '?')
                    fs['_warn_matches_template'] = True
                    template_matches.append(fid)
            except re.error:
                pass

    if template_matches:
        print(f'  FLAGGED {len(template_matches)} FS matching template code '
              f'(marked _warn_matches_template):')
        for fid in template_matches[:10]:
            print(f'    {fid}')
        quality_report['template_matches'] = template_matches
    else:
        print('  PASS: No FS match template/boilerplate code.')

    # --- 4.5g: Literal/Fallback FS Detection ---
    # FCC hard fallback uses re.escape() on unique code lines. These are
    # literal matches that only work for one specific student — they have
    # no generalisation value and should be excluded from auto-scoring.
    print('\n  --- Quality Gate: Literal/Fallback FS Detection ---')
    literal_fs: list[str] = []
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue
        # Heuristic: count backslash-escaped non-word characters
        # A re.escape() pattern has many \., \(, \), etc.
        total = len(regex)
        if total < 20:
            continue
        escaped = len(re.findall(r'\\[^\w]', regex))
        ratio = escaped / total if total > 0 else 0
        # High ratio of escaped chars = literal pattern from re.escape()
        if ratio > 0.25 and total > 40:
            fid = fs.get('id', '?')
            fs['_warn_literal_fallback'] = True
            literal_fs.append(fid)

    if literal_fs:
        print(f'  FLAGGED {len(literal_fs)} literal/fallback FS '
              f'(marked _warn_literal_fallback):')
        for fid in literal_fs[:10]:
            print(f'    {fid}')
        quality_report['literal_fallback_fs'] = literal_fs
    else:
        print('  PASS: No literal/fallback FS detected.')

    # --- 4.5h: Per-FS Scoring Weight Assignment ---
    # FS with quality warnings should carry reduced weight in auto-scoring.
    # This prevents flagged FS from having equal impact to clean FS.
    WARNING_WEIGHT_MAP = {
        '_warn_ref_match': 0.5,
        '_warn_broad_positive': 0.3,
        '_warn_narrow_positive': 0.3,
        '_warn_matches_nothing': 0.0,
        '_warn_type_annotation': 0.5,
        '_warn_feedback_mismatch': 0.7,
        '_warn_matches_template': 0.0,
        '_warn_literal_fallback': 0.0,
        '_warn_overlaps_with_negative': 0.5,
        '_warn_overlaps_with_positive': 0.5,
        '_duplicate_of': 0.0,
    }
    for fs in all_fs:
        weight = 1.0
        for warn_key, warn_penalty in WARNING_WEIGHT_MAP.items():
            if fs.get(warn_key):
                weight = min(weight, warn_penalty)
        fs['_scoring_weight'] = weight

    weighted_count = sum(1 for fs in all_fs if fs.get('_scoring_weight', 1.0) < 1.0)
    zero_weight = sum(1 for fs in all_fs if fs.get('_scoring_weight', 1.0) == 0.0)
    if weighted_count:
        print(f'\n  Scoring weights: {weighted_count} FS reduced '
              f'({zero_weight} at 0.0 — exclude from auto-scoring)')

    # --- Build scoring readiness summary ---
    all_criteria = sorted(set(
        fs.get('criterion', '?') for fs in all_fs
        if fs.get('criterion', '').startswith('RQ')
    ))
    for crit in all_criteria:
        if crit in contradictory:
            quality_report['scoring_readiness']['unreliable'].append(crit)
        elif crit in all_bad_for_criterion:
            quality_report['scoring_readiness']['needs_review'].append(crit)
        else:
            quality_report['scoring_readiness']['reliable'].append(crit)

    # --- 5. Post-Fix FCC check: if gaps opened, fill via AI supplement ---
    if remove_ids:
        print(f'\n  --- Post-Fix FCC Gap Check ---')
        postfix_new_fs = 0
        for batch, subs in sorted(all_subs_by_batch.items()):
            matched_task = BATCH_TASK_MAP.get(batch)
            if not matched_task:
                continue
            batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
            task_fs = [fs for fs in all_fs if fs.get('task') == matched_task]
            batch_gaps = find_gaps(batch_cov, subs, task_fs, min_gap_size=1)
            if batch_gaps:
                total = sum(len(v) for v in batch_gaps.values())
                print(f'  {batch}: {total} gap(s) after fix -- filling via AI...')
                for criterion, gap_students in sorted(batch_gaps.items()):
                    prompt = build_supplement_prompt(
                        criterion, gap_students[:10], all_fs, matched_task,
                        next((t.get('target_file', '') for t in tasks if t['id'] == matched_task), '')
                    )
                    resp = call_deepseek(SYSTEM_PROMPT, prompt)
                    if resp:
                        try:
                            data = extract_json(resp)
                            new_fs = data.get('supplement_fs', [])
                            for fs in new_fs:
                                fs['task'] = matched_task
                                fs['criterion'] = criterion
                                fs['auto_generated'] = True
                                m = re.search(r'\d+', str(criterion))
                                num = m.group() if m else '0'
                                fs_id_counter.setdefault(num, 0)
                                fs_id_counter[num] += 1
                                fs['id'] = f'FS{num}.{fs_id_counter[num]}'
                            all_fs.extend(new_fs)
                            postfix_new_fs += len(new_fs)
                        except Exception:
                            pass
                print(f'    Added {postfix_new_fs} FS to fill post-fix gaps')

    # ================================================================
    # Phase 2.10: Finalization (TAFFIES FCC report)
    # ================================================================
    print(f'\n{"=" * 60}')
    print('  Phase 2.10: TAFFIES FCC Finalization')
    print('=' * 60)

    # Run final FCC check and report per-criterion coverage
    final_gaps: dict[str, list] = {}
    for batch, subs in sorted(all_subs_by_batch.items()):
        matched_task = BATCH_TASK_MAP.get(batch)
        if not matched_task:
            continue
        batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
        task_fs = [fs for fs in all_fs if fs.get('task') == matched_task]
        for criterion, gap_students in find_gaps(batch_cov, subs, task_fs, min_gap_size=1).items():
            final_gaps[f'{batch}/{criterion}'] = gap_students

    # Report: rubric criteria coverage (the TAFFIES FCC metric)
    print(f'\n  --- TAFFIES FCC Report ---')
    rubric_criteria = set(fs.get('criterion', '?') for fs in all_fs if fs.get('criterion', '').startswith('RQ'))
    total_rubric_students = 0
    covered_rubric_students = 0
    for batch, subs in sorted(all_subs_by_batch.items()):
        matched_task = BATCH_TASK_MAP.get(batch)
        if not matched_task:
            continue
        batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
        for criterion in sorted(rubric_criteria):
            info = batch_cov['per_criterion'].get(criterion, {})
            if info:
                print(f'  {batch} / {criterion}: {info["covered"]}/{info["total"]} students ({info["coverage_pct"]}%)')
                total_rubric_students += info['total']
                covered_rubric_students += info['covered']

    total_rubric_pct = round(100 * covered_rubric_students / total_rubric_students, 1) if total_rubric_students else 0
    print(f'\n  Overall rubric coverage: {covered_rubric_students}/{total_rubric_students} ({total_rubric_pct}%)')

    total_gap_students = sum(len(v) for v in final_gaps.values())
    if final_gaps:
        print(f'\n  Remaining gaps: {len(final_gaps)} criteria, {total_gap_students} student-criterion pairs')
        for key, gap_students in sorted(final_gaps.items(), key=lambda x: -len(x[1])):
            print(f'    {key}: {len(gap_students)} students')
    else:
        print(f'\n  All rubric criteria fully covered — no gaps.')

    # ================================================================
    # Save final coverage per batch
    # ================================================================
    for batch, subs in sorted(all_subs_by_batch.items()):
        matched_task = BATCH_TASK_MAP.get(batch)
        if not matched_task:
            continue
        batch_cov = run_coverage_check(all_fs, subs, task_filter=matched_task)
        cov_path = os.path.join(BASE_DIR, 'output', question_id,
                                f'coverage_final_{batch}.json')
        os.makedirs(os.path.dirname(cov_path), exist_ok=True)
        with open(cov_path, 'w', encoding='utf-8') as f:
            json.dump({
                'per_criterion': batch_cov['per_criterion'],
                'matrix': batch_cov['matrix'],
            }, f, indent=2)

    # ================================================================
    # Save output
    # ================================================================
    out_dir = os.path.join(BASE_DIR, 'output', question_id)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'fs_registry.json')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'question': question_name,
            'model': DEEPSEEK_MODEL,
            'total_fs': len(all_fs),
            'fs_registry': all_fs,
        }, f, indent=2, ensure_ascii=False, default=str)

    # Write quality report alongside fs_registry.json
    qr_path = os.path.join(out_dir, 'quality_report.json')
    quality_report['generated_at'] = datetime.now().isoformat()
    quality_report['total_fs_after_quality_gate'] = len(all_fs)
    with open(qr_path, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
    print(f'  Quality report: {qr_path}')
    reliable = quality_report['scoring_readiness']['reliable']
    unreliable = quality_report['scoring_readiness']['unreliable']
    needs_review = quality_report['scoring_readiness']['needs_review']
    print(f'  Scoring readiness: {len(reliable)} reliable, '
          f'{len(unreliable)} unreliable, {len(needs_review)} needs review')

    print(f'\n{"=" * 60}')
    print(f'  DONE -- {len(all_fs)} FS total')
    pos = sum(1 for f in all_fs if f.get('fs_type') == 'positive')
    neg = sum(1 for f in all_fs if f.get('fs_type') == 'negative')
    pending = sum(1 for f in all_fs if f.get('fs_type') == 'pending_judgment')
    judged = sum(1 for f in all_fs if f.get('judged_by_ai'))
    print(f'  Positive: {pos}, Negative: {neg}', end='')
    if pending:
        print(f', Pending: {pending}', end='')
    if judged:
        print(f', AI-judged: {judged}', end='')
    print()
    print(f'  Output: {json_path}')

    return all_fs


# ============================================================
# Standalone exports for main.py orchestration
# ============================================================

def phase0_analyze_question(question_dir: str, cache_path: str = '') -> dict | None:
    """AI analyzes question folder. Uses rubric cache for reproducibility."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('tasks'):
            print(f'  Loaded cached rubric ({len(cached["tasks"])} tasks)')
            return cached

    print('\n--- Phase 0: AI analyzing question folder ---')
    p0_prompt = build_phase0_prompt(question_dir)
    print(f'  Prompt size: {len(p0_prompt)} chars')

    result = None
    for attempt in range(3):
        resp = call_deepseek(PHASE0_SYSTEM, p0_prompt)
        if not resp:
            continue
        try:
            result = extract_json(resp)
            break
        except Exception:
            if attempt < 2:
                try:
                    result = extract_json(_repair_json(resp))
                    break
                except Exception:
                    continue

    if not result:
        print('  FAILED — cannot proceed without question analysis')
        return None

    tasks = result.get('tasks', [])
    if not tasks:
        print('  No tasks detected')
        return None

    print(f'  Detected: {result.get("question_name", "?")} — {len(tasks)} tasks')
    for t in tasks:
        print(f'    {t["id"]}: {t.get("target_file", "?")} -> {t.get("target_functions", [])}')

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'  Cached rubric to {cache_path}')

    return result


def phase1_ensure_references(tasks: list[dict], question_dir: str,
                              ref_dir: str) -> None:
    """Ensure per-criterion reference variants exist, generating via AI if needed.

    Generates 3-5 diverse correct implementation snippets per criterion.
    Stores as criterion_variants.json per task, plus .py files for each variant.
    """
    for task in tasks:
        task_id = task['id']
        task_ref_dir = os.path.join(ref_dir, task_id) if os.path.isdir(ref_dir) else ref_dir
        os.makedirs(task_ref_dir, exist_ok=True)

        # Check for existing per-criterion variants
        variants_file = os.path.join(task_ref_dir, 'criterion_variants.json')
        existing_py = [f for f in os.listdir(task_ref_dir)
                       if f.endswith('.py') and f != 'criterion_variants.json'] \
                      if os.path.isdir(task_ref_dir) else []

        if os.path.exists(variants_file) and len(existing_py) >= 4:
            with open(variants_file, 'r', encoding='utf-8') as f:
                task['criterion_variants'] = json.load(f)
            task['reference_files'] = existing_py
            continue

        print(f'\n--- Phase 1: Generating per-criterion variants for {task_id} ---')
        prompt = build_phase1_prompt(task_id, task, question_dir)
        resp = call_deepseek(PHASE0_SYSTEM, prompt)
        if not resp:
            task['reference_files'] = existing_py or []
            continue
        try:
            ref_data = extract_json(resp)
        except Exception:
            task['reference_files'] = existing_py or []
            continue

        # New format: per-criterion implementations
        crit_impls = ref_data.get('criterion_implementations', {})
        if not crit_impls:
            # Fallback: old format with solutions
            saved = []
            for sol in ref_data.get('solutions', []):
                fname = sol.get('filename', f'{task_id.lower()}_v{chr(65 + len(saved))}.py')
                filepath = os.path.join(task_ref_dir, fname)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(sol.get('code', ''))
                saved.append(fname)
            task['reference_files'] = saved
            task['criterion_variants'] = {}
            print(f'  Generated {len(saved)} reference solutions (old format)')
            continue

        # Save per-criterion variants
        saved = []
        all_variants = {}
        for crit_id, variants in crit_impls.items():
            all_variants[crit_id] = []
            for v in variants:
                vname = f'{crit_id}_v{v.get("variant", chr(65 + len(all_variants[crit_id])))}.py'
                filepath = os.path.join(task_ref_dir, vname)
                code = v.get('code', '')
                if code:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(code)
                    saved.append(vname)
                    all_variants[crit_id].append({
                        'variant': v.get('variant', '?'),
                        'approach': v.get('approach', ''),
                        'file': vname,
                    })

        # Save variants index
        with open(variants_file, 'w', encoding='utf-8') as f:
            json.dump(all_variants, f, indent=2)

        task['criterion_variants'] = all_variants
        task['reference_files'] = saved
        total_variants = sum(len(v) for v in all_variants.values())
        print(f'  Generated {total_variants} variants across {len(all_variants)} criteria')


def phase2_generate_fs(tasks: list[dict], submissions_dir: str,
                        ref_dir: str, all_subs_by_batch: dict,
                        template_code: str = '') -> list[dict]:
    """Generate FS for each task via AI. Returns combined FS list."""
    all_fs: list[dict] = []
    fs_id_counter: dict[str, int] = {}
    BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}

    for batch, matched_task in sorted(BATCH_TASK_MAP.items()):
        task = next((t for t in tasks if t['id'] == matched_task), None)
        if not task:
            continue
        task_id = task['id']
        target_file = task.get('target_file', 'iMusic.py')

        print(f'\n--- Phase 2: Generating FS for {batch} -> {task_id} ({target_file}) ---')

        submissions = collect_submissions(submissions_dir, target_file,
                                           student_prefix=batch, max_students=30)
        if not submissions:
            print('  No submissions found, skipping')
            continue

        # Build reference code from per-criterion variants
        ref_parts = []
        crit_variants = task.get('criterion_variants', {})
        if crit_variants:
            for crit_id, variants in crit_variants.items():
                for v in variants:
                    fname = v.get('file', '')
                    for root, _, files in os.walk(ref_dir):
                        if fname in files:
                            code = read_file(os.path.join(root, fname))
                            ref_parts.append(
                                f'### {crit_id} — {v.get("approach", "reference implementation")}\n'
                                f'```python\n{code}\n```\n'
                            )
                            break
        ref_code = '\n'.join(ref_parts)

        # Fallback: old-style reference files
        if not ref_code:
            for rf in task.get('reference_files', []):
                for root, _, files in os.walk(ref_dir):
                    if rf in files:
                        ref_code += f'### {rf}\n```python\n{read_file(os.path.join(root, rf))}\n```\n'
                        break

        n_variants = sum(len(v) for v in crit_variants.values()) if crit_variants else len(task.get('reference_files', []))
        print(f'  Ref variants: {n_variants}, Subs: {len(submissions)}')
        prompt = build_fs_prompt(task_id, task, ref_code, submissions, template_code)
        print(f'  Calling API ({len(prompt)} chars)...')

        result = None
        for attempt in range(3):
            resp = call_deepseek(SYSTEM_PROMPT, prompt)
            if not resp:
                continue
            try:
                result = extract_json(resp)
                break
            except Exception:
                if attempt < 2:
                    try:
                        result = extract_json(_repair_json(resp))
                        break
                    except Exception:
                        continue

        if not result:
            print('  FAILED')
            continue

        task_fs = result.get('fs_registry', [])
        for fs in task_fs:
            fs.setdefault('task', task_id)
            fs.setdefault('files', [target_file])
            fs.setdefault('auto_generated', True)
            fs.pop('marks', None)
            crit = fs.get('criterion', '?')
            m = re.search(r'\d+', str(crit))
            num = m.group() if m else '0'
            fs_id_counter.setdefault(num, 0)
            fs_id_counter[num] += 1
            fs['id'] = f'FS{num}.{fs_id_counter[num]}'

        all_fs.extend(task_fs)
        pos = sum(1 for f in task_fs if f.get('fs_type') == 'positive')
        neg = sum(1 for f in task_fs if f.get('fs_type') == 'negative')
        print(f'  Generated {len(task_fs)} FS ({pos}+, {neg}-)')

    return all_fs


def phase2_generate_fs_batched(
    tasks: list[dict],
    submissions_dir: str,
    ref_dir: str,
    template_code: str = '',
    all_readmes: dict | None = None,
    batch_size: int = 10,
    max_prompt_chars: int = 50000,
) -> list[dict]:
    """Generate FS for each task via AI using BATCH PROCESSING with README ground truth.

    V3: Independent batches + post-generation dedup (no incremental FS sharing).
    - Each batch generates FS independently (no previous FS in prompt)
    - Prompt size capped to avoid DeepSeek 64K context overflow
    - After all batches complete, deduplicate across batches

    Args:
        tasks: Task configs from Phase 0.
        submissions_dir: Path to submission/ (CW format).
        ref_dir: Path to reference solutions.
        template_code: Template code (for reference only).
        all_readmes: Output of ground_truth.load_all_readmes().
        batch_size: Students per batch (default 10). Reduced if prompt too large.
        max_prompt_chars: Hard limit on prompt size (default 50K, DeepSeek 64K limit).

    Returns:
        Combined, deduplicated list of FS dictionaries.
    """
    all_fs: list[dict] = []
    fs_id_counter: dict[str, int] = {}

    TASK_NUM_MAP = {1: 'Task1', 2: 'Task2', 3: 'Task3'}

    # Load ground truth patterns if available
    readme_patterns_by_task = {}
    if all_readmes:
        from ground_truth import get_task_patterns
        for tn in [1, 2, 3]:
            readme_patterns_by_task[tn] = get_task_patterns(all_readmes, TASK_NUM_MAP[tn])

    for task_num in [1, 2, 3]:
        task_id = TASK_NUM_MAP[task_num]
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            continue

        target_file = f'task{task_num}.py'
        submissions = collect_submissions_by_task(submissions_dir, task_num, max_students=None)
        if not submissions:
            print(f'\n--- Phase 2 [{task_id}]: No submissions found ---')
            continue

        # Build reference code and README text (shared across all batches)
        ref_code = _build_ref_code(task, ref_dir)
        task_patterns = readme_patterns_by_task.get(task_num, {})
        readme_text = ''
        if task_patterns:
            from ground_truth import format_patterns_for_prompt
            readme_text = format_patterns_for_prompt(task_patterns, task_id)

        # Determine batch size: start with configured size, reduce if prompt too large
        effective_batch_size = batch_size
        while effective_batch_size >= 6:
            # Estimate prompt with one batch
            test_batch = submissions[:effective_batch_size]
            test_prompt = build_fs_prompt(
                task_id, task, ref_code, test_batch,
                template_code=template_code,
                readme_patterns=readme_text,
                previous_fs=None,  # No previous FS!
            )
            if len(test_prompt) <= max_prompt_chars:
                break
            effective_batch_size -= 2
            print(f'  [{task_id}] Prompt {len(test_prompt)} chars > {max_prompt_chars}, '
                  f'reducing batch to {effective_batch_size}')

        # Split into batches
        batches = []
        for i in range(0, len(submissions), effective_batch_size):
            batch = submissions[i:i + effective_batch_size]
            if batch:
                batches.append(batch)

        print(f'\n--- Phase 2 [{task_id}]: {len(submissions)} students, '
              f'{len(batches)} batches of {effective_batch_size} '
              f'(max prompt {max_prompt_chars // 1000}K) ---')
        if readme_text:
            print(f'  README: {len(task_patterns.get("good", []))} good, '
                  f'{len(task_patterns.get("bad", []))} bad patterns')

        # Process each batch INDEPENDENTLY (no previous FS)
        task_fs: list[dict] = []
        for batch_idx, batch in enumerate(batches):
            batch_label = f'Batch {batch_idx + 1}/{len(batches)}'
            sids = [s['student'] for s in batch]

            prompt = build_fs_prompt(
                task_id, task, ref_code, batch,
                template_code=template_code,
                readme_patterns=readme_text,
                previous_fs=None,  # No incremental FS — post-generation dedup instead
                batch_label=batch_label,
            )
            print(f'  {batch_label}: {sids[0]}-{sids[-1]} ({len(batch)} students, '
                  f'{len(prompt)} chars)')

            result = None
            for attempt in range(3):
                resp = call_deepseek(SYSTEM_PROMPT, prompt)
                if not resp:
                    continue
                try:
                    result = extract_json(resp)
                    break
                except Exception:
                    if attempt < 2:
                        try:
                            result = extract_json(_repair_json(resp))
                            break
                        except Exception:
                            continue

            if not result:
                print(f'  {batch_label} FAILED after 3 attempts')
                continue

            batch_fs = result.get('fs_registry', [])
            for fs in batch_fs:
                fs.setdefault('task', task_id)
                fs.setdefault('files', [target_file])
                fs.setdefault('auto_generated', True)
                fs.setdefault('_batch', batch_idx + 1)
                fs.pop('marks', None)
                crit = fs.get('criterion', '?')
                m = re.search(r'\d+', str(crit))
                num = m.group() if m else '0'
                fs_id_counter.setdefault(num, 0)
                fs_id_counter[num] += 1
                fs['id'] = f'FS{num}.{fs_id_counter[num]}'

            task_fs.extend(batch_fs)
            pos = sum(1 for f in batch_fs if f.get('fs_type') == 'positive')
            neg = sum(1 for f in batch_fs if f.get('fs_type') == 'negative')
            print(f'  {batch_label}: {len(batch_fs)} FS ({pos}+, {neg}-)')

        # Post-generation dedup: remove near-duplicate FS across batches
        before_dedup = len(task_fs)
        task_fs = _deduplicate_fs(task_fs)
        after_dedup = len(task_fs)
        if before_dedup != after_dedup:
            print(f'  [{task_id}] Dedup: {before_dedup} → {after_dedup} FS '
                  f'({before_dedup - after_dedup} duplicates merged)')

        all_fs.extend(task_fs)
        total_pos = sum(1 for f in task_fs if f.get('fs_type') == 'positive')
        total_neg = sum(1 for f in task_fs if f.get('fs_type') == 'negative')
        print(f'  [{task_id}] Final: {after_dedup} FS ({total_pos}+, {total_neg}-)')

    return all_fs


def _deduplicate_fs(fs_list: list[dict]) -> list[dict]:
    """Post-generation FS deduplication: merge near-identical regexes across batches.

    Groups FS by (criterion, fs_type), normalizes regexes, and keeps the best
    (longest feedback) from each cluster of near-identical patterns.
    """
    from collections import defaultdict

    def _normalize_regex(regex: str) -> str:
        """Normalize regex for dedup comparison."""
        # Collapse all whitespace patterns
        norm = re.sub(r'\\s\+|\\s\*|\\s\{1,\}', r'\\s*', regex)
        norm = re.sub(r'\\d\+|\\d\*|\\d\{1,\}', r'\\d*', norm)
        norm = re.sub(r'\\w\+|\\w\*|\\w\{1,\}', r'\\w*', norm)
        # Collapse literal whitespace
        norm = re.sub(r'\s+', '', norm)
        return norm

    # Group by (criterion, fs_type, normalized_regex)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for fs in fs_list:
        key = (fs.get('criterion', '?'), fs.get('fs_type', '?'),
               _normalize_regex(fs.get('regex', '')))
        groups[key].append(fs)

    # Keep the best FS from each group
    result = []
    for (criterion, fs_type, _), group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Pick keeper: longest feedback (most descriptive)
            group.sort(key=lambda f: len(f.get('feedback', '')), reverse=True)
            keeper = group[0]
            for dup in group[1:]:
                dup['_duplicate_of'] = keeper.get('id', '?')
            result.append(keeper)

    return result


def _build_ref_code(task: dict, ref_dir: str) -> str:
    """Build reference code string for a task from per-criterion variants or files."""
    ref_parts = []
    crit_variants = task.get('criterion_variants', {})
    if crit_variants:
        for crit_id, variants in crit_variants.items():
            for v in variants:
                fname = v.get('file', '')
                for root, _, files in os.walk(ref_dir):
                    if fname in files:
                        code = read_file(os.path.join(root, fname))
                        ref_parts.append(
                            f'### {crit_id} — {v.get("approach", "reference")}\n'
                            f'```python\n{code}\n```\n'
                        )
                        break
    if not ref_parts:
        for rf in task.get('reference_files', []):
            for root, _, files in os.walk(ref_dir):
                if rf in files:
                    ref_parts.append(
                        f'### {rf}\n```python\n{read_file(os.path.join(root, rf))}\n```\n'
                    )
                    break
    return '\n'.join(ref_parts)


# ============================================================
# Conflict Detection (deterministic, no AI)
# ============================================================

def detect_conflicts(fs_list: list[dict], all_subs_by_batch: dict) -> dict:
    """Detect students matched by BOTH positive AND negative FS for same criterion.
    Auto-mitigates: reduces negative FS weight to 0.5 when conflict found.

    Returns conflict report.
    """
    from coverage import run_coverage_check

    # Normalize: handle both {batch: {sid: code}} and {batch: [{student, code}, ...]}
    sub_list = []
    for batch, students in all_subs_by_batch.items():
        if isinstance(students, dict):
            for sid, code in students.items():
                sub_list.append({'student': sid, 'code': code})
        elif isinstance(students, list):
            sub_list.extend(students)

    cov = run_coverage_check(fs_list, sub_list)
    matrix = cov['matrix']
    fs_hits = cov['fs_hits']

    fs_lookup = {f['id']: f for f in fs_list}

    conflicts = []
    mitigated = 0

    for sid, crits in matrix.items():
        for crit in crits:
            pos_fs_ids = []
            neg_fs_ids = []
            for fid, students in fs_hits.items():
                if sid not in students: continue
                f = fs_lookup.get(fid)
                if not f or f.get('criterion') != crit: continue
                if f.get('_scoring_weight', 1.0) == 0: continue
                if f.get('fs_type') == 'positive': pos_fs_ids.append(fid)
                elif f.get('fs_type') == 'negative': neg_fs_ids.append(fid)

            if pos_fs_ids and neg_fs_ids:
                conflicts.append({
                    'student': sid, 'criterion': crit,
                    'positive_fs': pos_fs_ids,
                    'negative_fs': neg_fs_ids,
                })
                for fid in neg_fs_ids:
                    f = fs_lookup.get(fid)
                    if f and f.get('_scoring_weight', 1.0) == 1.0:
                        f['_scoring_weight'] = 0.5
                        f['_conflict_mitigated'] = True
                        mitigated += 1

    print(f'\n  Conflict Detection: {len(conflicts)} student-criterion conflicts')
    print(f'  Auto-mitigated: {mitigated} negative FS (weight 1.0 -> 0.5)')

    return {'conflicts': conflicts, 'mitigated_count': mitigated}


# ============================================================
# Quality-Biased Coverage Check (deterministic, no AI)
# ============================================================

# NOTE: FULL_CRITERIA_MAP and BATCH_TASK are derived dynamically from rubric cache.
# The static maps below are fallbacks only — actual criteria come from rubric_cache.json.
BATCH_TASK = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}
# FULL_CRITERIA_MAP is built at runtime in functions that need it


def check_quality_coverage(fs_list: list[dict], all_subs_by_batch: dict) -> dict:
    """Check FS coverage stratified by student quality level.

    Reports:
    - Positive FS missing for high-quality students (correct code not rewarded)
    - Negative FS missing for low-quality students (wrong code not penalized)
    - Negative FS hitting high-quality students (potential false negatives)
    """
    from coverage import run_coverage_check

    # Normalize: handle both {batch: {sid: code}} and {batch: [{student, code}, ...]}
    sub_list = []
    for batch, students in all_subs_by_batch.items():
        if isinstance(students, dict):
            for sid, code in students.items():
                sub_list.append({'student': sid, 'code': code})
        elif isinstance(students, list):
            sub_list.extend(students)

    cov = run_coverage_check(fs_list, sub_list)
    matrix = cov['matrix']
    fs_hits = cov['fs_hits']
    fs_lookup = {f['id']: f for f in fs_list}

    high_quality = {'excellent', 'full-marks'}
    low_quality = {'poor', 'weak'}

    neg_on_high = []       # negative FS matching excellent students
    no_pos_on_high = []    # excellent students with no positive FS
    no_neg_on_low = []     # poor students with no negative FS

    for sid in matrix:
        quality = sid.split('-')[1] if '-' in sid else '?'
        batch = sid[:3]
        task = BATCH_TASK.get(batch, '')
        # Build criteria list dynamically from FS registry (not hardcoded)
        criteria = list(set(fs.get('criterion', '')
                            for fs in fs_list
                            if fs.get('criterion', '').startswith('RQ')
                            and fs.get('task') == task))

        for crit in criteria:
            pos_hits = [fid for fid, students in fs_hits.items()
                        if sid in students
                        and fs_lookup.get(fid, {}).get('criterion') == crit
                        and fs_lookup.get(fid, {}).get('fs_type') == 'positive'
                        and fs_lookup.get(fid, {}).get('_scoring_weight', 1.0) > 0]
            neg_hits = [fid for fid, students in fs_hits.items()
                        if sid in students
                        and fs_lookup.get(fid, {}).get('criterion') == crit
                        and fs_lookup.get(fid, {}).get('fs_type') == 'negative'
                        and fs_lookup.get(fid, {}).get('_scoring_weight', 1.0) > 0]

            if quality in high_quality:
                if neg_hits:
                    neg_on_high.append((sid, crit, neg_hits))
                if not pos_hits:
                    no_pos_on_high.append((sid, crit))

            if quality in low_quality and not neg_hits:
                no_neg_on_low.append((sid, crit))

    print(f'\n  Quality-Biased Coverage:')
    print(f'    Neg FS on excellent students: {len(neg_on_high)} (potential false neg)')
    print(f'    Excellent missing positive:   {len(no_pos_on_high)} (correct unrewarded)')
    print(f'    Poor/weak missing negative:   {len(no_neg_on_low)} (wrong unpenalized)')

    return {
        'neg_on_high': neg_on_high,
        'no_pos_on_high': no_pos_on_high,
        'no_neg_on_low': no_neg_on_low,
    }


def fcc_supplement_loop(all_fs: list[dict], all_subs_by_batch: dict,
                         tasks: list[dict], max_rounds: int = 5,
                         ref_code: str = '', template_code: str = '') -> int:
    """Progressive 3-phase FCC supplement: quick→deep→last-resort.

    Round 1-2: 5 samples/criterion, T=0.3
    Round 3+: ALL gap students shown, T=0.5, targeted patterns
    Returns total number of FS added.
    """
    from collections import defaultdict as _dd

    BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}
    fs_id_counter: dict[str, int] = {}
    for fs in all_fs:
        crit = fs.get('criterion', '?')
        try:
            seq = int(fs.get('id', 'FS0.0').split('.')[-1] or 0)
        except (ValueError, IndexError):
            seq = 0
        fs_id_counter[crit] = max(fs_id_counter.get(crit, 0), seq)

    total_added = 0

    for rnd in range(1, max_rounds + 1):
        print(f'\n  --- FCC Round {rnd} ---')

        # Show per-batch coverage
        for batch, subs in sorted(all_subs_by_batch.items()):
            mt = BATCH_TASK_MAP.get(batch)
            if not mt:
                continue
            cov = run_coverage_check(all_fs, subs, task_filter=mt)
            print(f'\n  {batch} -> {mt} ({len(subs)} students)')
            print(format_coverage_report(cov))

        # Collect all gaps
        all_gaps = {}
        for batch, subs in all_subs_by_batch.items():
            mt = BATCH_TASK_MAP.get(batch)
            if not mt:
                continue
            task_fs = [fs for fs in all_fs if fs.get('task') == mt]
            cov = run_coverage_check(all_fs, subs, task_filter=mt)
            for c, g in find_gaps(cov, subs, task_fs, min_gap_size=1).items():
                all_gaps[c] = g

        if not all_gaps:
            print('  No gaps — converged!')
            break

        total = sum(len(v) for v in all_gaps.values())
        print(f'\n  Gaps: {len(all_gaps)} criteria, {total} pairs')
        for c, g in sorted(all_gaps.items(), key=lambda x: -len(x[1])):
            print(f'    {c}: {len(g)} students')

        # Group by task
        gaps_by_task = _dd(dict)
        for criterion, gap_students in all_gaps.items():
            m = re.search(r'\d+', str(criterion))
            tn = f'Task{m.group()}' if m else 'Task1'
            gaps_by_task[tn][criterion] = gap_students

        # Progressive strategy
        samples_per = 5 if rnd <= 2 else 999  # Show ALL in round 3+
        temp = 0.3 if rnd <= 2 else 0.5
        extra = ""
        if rnd >= 3:
            extra = (
                "\n\nLAST RESORT — gaps survived multiple rounds. Study EACH "
                "student's code carefully. Consider: variable whitelists, "
                "f-string ORDER BY without validation, .format() ORDER BY, "
                "pre-fetch validation (SELECT INTO set), multi-line render_template, "
                "hardcoded returns, truthiness checks (if x and y), "
                "equals checks (if x == 'Y'). Generate one FS per pattern."
            )

        new_count = 0
        for parent_task, task_gaps in sorted(gaps_by_task.items()):
            # Build supplement prompt
            task_fs = [fs for fs in all_fs if fs.get('task') == parent_task]
            existing_text = '\n'.join(
                f"- {fs.get('id','?')}: {fs.get('name','')[:60]} | {fs.get('regex','')[:80]}"
                for fs in task_fs[:25]
            )
            sections = []
            for criterion, gap_students in sorted(task_gaps.items()):
                sample = gap_students[:samples_per]
                sections.append(
                    f"## {criterion}: {len(gap_students)} uncovered\n"
                    + '\n\n'.join(f"### {g['student']}\n```python\n{g['code'][:1200]}\n```"
                                  for g in sample)
                )

            prompt = f"""Analyze why these students are uncovered AND generate targeted FS.

## Rubric Criteria for {parent_task}
{json.dumps([c for t in tasks if t['id'] == parent_task for c in t.get('rubric_criteria', [])], indent=2)}

## Reference Solutions (correct implementations — NEGATIVE FS MUST NOT match these)
```python
{ref_code[:3000] if ref_code else '(none)'}
```

## Template Starter Code (DO NOT match — unchanged from starter)
```python
{template_code[:2000] if template_code else '(none)'}
```

## Existing FS for {parent_task} (DO NOT duplicate patterns)
{existing_text if existing_text else '(none)'}

## Uncovered Students
{chr(10).join(sections)}
{extra}

## Instructions
For each criterion with uncovered students, CHECKLIST-BASED JUDGMENT:
1. Read the criterion's good_patterns and bad_patterns above
2. For each uncovered student, check each pattern as a YES/NO checklist
3. **EXTRACT naming variants from uncovered students' code:**
   Look at what table/column names these students ACTUALLY use.
   Common variants to match: PlaylistTrack\|playlist_track\|playlist_tracks,
   Playlist\|playlists, Track\|tracks\|track, GenreId\|genre_id, etc.
   Your regex MUST include the variants seen in uncovered students' code.
4. Generate 3-6 FS per criterion:
   - regex: \\\\w+ for identifiers, \\\\s+ for whitespace, [^\\n]* for line-internal
   - Table/column names: use (?:variant1\|variant2) to match student variants
   - Positive FS: MUST contain specific implementation detail NOT in the template
   - Positive FS: MUST use broad table/column matching to cover ALL students
   - Negative FS — Type A (missing good): use (?!...) across entire function
   - Negative FS — Type B (present bad): pure positive regex IS correct, match bad pattern directly
   - Negative FS: MUST NOT match reference code
   - Each FS: non-empty feedback (2-3 sentences)
5. Validate mentally: negative FS does NOT match reference; positive FS does NOT match template

Output ONLY JSON: {{"supplement_fs": [...]}}"""

            print(f'\n    {parent_task}: {len(task_gaps)} criteria, '
                  f'{sum(len(v) for v in task_gaps.values())} gaps (r{rnd}, T={temp})')
            resp = call_deepseek(SYSTEM_PROMPT, prompt, temperature=temp)
            if not resp:
                print('    FAILED')
                continue

            try:
                result = extract_json(resp)
            except Exception:
                try:
                    result = extract_json(_repair_json(resp))
                except Exception:
                    print('    Parse error')
                    continue

            new_fs_list = result.get('supplement_fs', [])
            added = 0
            for fs in new_fs_list:
                regex = fs.get('regex')
                criterion = fs.get('criterion', '')
                if not regex:
                    continue
                try:
                    flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
                    re.compile(regex, flags)
                except re.error:
                    continue
                gap_subs = task_gaps.get(criterion, [])
                if not any(re.search(regex, g['code'], flags)
                          for g in gap_subs[:15]):
                    continue

                # Cross-check: negative FS must not match reference code
                if fs.get('fs_type') == 'negative' and ref_code:
                    try:
                        if re.search(regex, ref_code, flags):
                            fs['_warn_ref_match'] = True
                    except re.error:
                        pass
                # Cross-check: positive FS must not match template code
                if fs.get('fs_type') == 'positive' and template_code:
                    try:
                        if re.search(regex, template_code, flags):
                            fs['_warn_matches_template'] = True
                    except re.error:
                        pass

                fs.setdefault('task', parent_task)
                # Get target_file from task config
                pt = next((t for t in tasks if t['id'] == parent_task), None)
                fs.setdefault('files', [pt.get('target_file', 'main.py') if pt else 'main.py'])
                fs['source'] = 'fcc'
                fs['auto_generated'] = True
                fs_id_counter.setdefault(criterion, 0)
                fs_id_counter[criterion] += 1
                fs['id'] = f'FS{criterion.replace("RQ", "").replace("_", "")}.{fs_id_counter[criterion]}'
                all_fs.append(fs)
                added += 1
                new_count += 1
            print(f'    +{added}/{len(new_fs_list)} FS')

        if new_count == 0 and rnd >= 3:
            # ---- ESCALATION: per-criterion dedicated prompts ----
            # When bulk supplement fails, target each criterion individually
            # with ALL gap students shown. This mirrors the manual process
            # that achieved 100%.
            print('\n  --- ESCALATION: per-criterion targeted supplement ---')
            esc_added = 0
            for criterion, gap_students in sorted(all_gaps.items(),
                                                    key=lambda x: -len(x[1])):
                if len(gap_students) == 0:
                    continue
                m = re.search(r'\d+', str(criterion))
                pt = f'Task{m.group()}' if m else 'Task1'
                task_fs = [fs for fs in all_fs if fs.get('task') == pt]

                # Dedicated prompt for ONE criterion, showing ALL gap students
                prompt = f"""Generate FS for {criterion} — {len(gap_students)} students uncovered.

## Existing FS for {criterion}
{chr(10).join(f"- {fs.get('id','?')}: {fs.get('name','')[:60]} | {fs.get('regex','')[:80]}" for fs in task_fs if fs.get('criterion') == criterion)}

## ALL Uncovered Students — study EACH one carefully
{chr(10).join(f"### {g['student']}\n```python\n{g['code'][:1500]}\n```" for g in gap_students)}

## Task
Generate 3-8 FS that SPECIFICALLY match the patterns in these students' code.
Look for: variable whitelists, f-string ORDER BY, .format() ORDER BY, pre-fetch
validation, multi-line render_template, hardcoded returns, pass stubs.
Each regex MUST match at least one student above.
Output ONLY JSON: {{"supplement_fs": [...]}}"""

                print(f'    {criterion}: {len(gap_students)} gaps, '
                      f'calling AI ({len(prompt)} chars)...')
                resp = call_deepseek(SYSTEM_PROMPT, prompt, temperature=0.5)
                if not resp:
                    continue
                try:
                    result = extract_json(resp)
                except Exception:
                    try:
                        result = extract_json(_repair_json(resp))
                    except Exception:
                        continue

                for fs in result.get('supplement_fs', []):
                    regex = fs.get('regex')
                    if not regex:
                        continue
                    try:
                        flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
                        re.compile(regex, flags)
                    except re.error:
                        continue
                    if not any(re.search(regex, g['code'], flags)
                              for g in gap_students[:15]):
                        continue
                    # Cross-check against reference/template
                    if fs.get('fs_type') == 'negative' and ref_code:
                        try:
                            if re.search(regex, ref_code, flags):
                                fs['_warn_ref_match'] = True
                        except re.error:
                            pass
                    if fs.get('fs_type') == 'positive' and template_code:
                        try:
                            if re.search(regex, template_code, flags):
                                fs['_warn_matches_template'] = True
                        except re.error:
                            pass
                    fs.setdefault('task', pt)
                    tf = next((t.get('target_file', 'main.py') for t in tasks if t['id'] == pt), 'main.py')
                    fs.setdefault('files', [tf])
                    fs['source'] = 'fcc'
                    fs['auto_generated'] = True
                    fs.setdefault('criterion', criterion)
                    fs_id_counter.setdefault(criterion, 0)
                    fs_id_counter[criterion] += 1
                    fs['id'] = f'FS{criterion.replace("RQ", "").replace("_", "")}.{fs_id_counter[criterion]}'
                    all_fs.append(fs)
                    esc_added += 1
                print(f'    +{len(result.get("supplement_fs", []))} FS for {criterion}')

            if esc_added > 0:
                total_added += esc_added
                print(f'  Escalation added {esc_added} FS, total {len(all_fs)}')
                # Continue to next round to re-check coverage
                continue
            else:
                print('  No FS after escalation — truly converged.')
                break

        total_added += new_count
        print(f'  Round {rnd}: +{new_count} FS, total {len(all_fs)}')

    return total_added


def apply_quality_gates(all_fs, all_subs_by_batch, all_subs_flat,
                         all_ref_code, template_code, question_dir=''):
    """Standalone quality gates: generalization, dedup, escape fix, weights.

    This is a thin export of the quality gate logic embedded in run_pipeline().
    Call it directly when orchestrating from main.py.
    """
    quality_report = {'structurally_broken': []}
    print('\n  --- Quality Gate: Variable Name Generalisation ---')
    import tokenize as _tokenize, io as _io

    PY_KEYWORDS = frozenset({
        'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'try',
        'except', 'finally', 'with', 'as', 'import', 'from', 'class',
        'pass', 'break', 'continue', 'yield', 'raise', 'assert', 'del',
        'global', 'nonlocal', 'lambda', 'and', 'or', 'not', 'in', 'is',
        'True', 'False', 'None', 'self', 'cls',
    })
    SQL_KEYWORDS = frozenset({
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
        'ON', 'GROUP', 'BY', 'ORDER', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'AS', 'IN', 'OR', 'IGNORE', 'INTO', 'VALUES', 'SET', 'CREATE',
        'TABLE', 'EXISTS', 'NOT', 'AND', 'LIKE', 'BETWEEN', 'HAVING',
        'DISTINCT', 'UNION', 'ALL', 'LIMIT', 'OFFSET', 'INNER', 'OUTER',
        'LEFT', 'RIGHT', 'CROSS', 'FULL', 'NATURAL', 'USING', 'ASC',
        'DESC', 'NULL', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'INDEX',
        'UNIQUE', 'CHECK', 'DEFAULT', 'CASCADE', 'OR', 'REPLACE',
        'select', 'insert', 'update', 'delete', 'from', 'where', 'join',
        'on', 'group', 'by', 'order', 'count', 'sum', 'avg', 'min', 'max',
    })
    KNOWN_APIS = frozenset({
        'app', 'flask', 'Flask', 'request', 'session', 'g', 'redirect',
        'url_for', 'render_template', 'flash', 'make_response', 'jsonify',
        'cursor', 'conn', 'connection', 'db', 'sqlite3', 'sqlite',
        'csv', 'DictReader', 'reader', 'writer', 'DictWriter',
        'Path', 'open', 'file', 'os', 'sys', 'json', 're', 'datetime',
        'timedelta', 'date', 'Enum', 'Base', 'Exception', 'ValueError',
        'TypeError', 'KeyError', 'IndexError', 'FileNotFoundError',
        'OperationalError', 'IntegrityError', 'ProgrammingError',
        'fetchall', 'fetchone', 'fetchmany', 'execute', 'executemany',
        'commit', 'rollback', 'close', 'connect', 'cursor_factory',
        'row_factory', 'rowcount', 'lastrowid', 'description',
        'get', 'post', 'put', 'delete', 'patch', 'route', 'errorhandler',
        'before_request', 'after_request', 'teardown_request',
        'send_file', 'send_from_directory', 'abort',
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
        'bytes', 'bytearray', 'frozenset', 'range', 'enumerate', 'zip',
        'map', 'filter', 'sorted', 'reversed', 'iter', 'next', 'len',
        'print', 'input', 'isinstance', 'issubclass', 'hasattr', 'getattr',
        'setattr', 'delattr', 'type', 'super', 'object', 'property',
        'staticmethod', 'classmethod', 'any', 'all', 'abs', 'round',
        'min', 'max', 'sum', 'pow', 'divmod', 'chr', 'ord', 'hex', 'oct',
        'bin', 'format', 'repr', 'ascii', 'eval', 'exec', 'compile',
        '__name__', '__main__', '__file__', '__init__', '__str__',
        '__repr__', '__dict__', '__class__', '__doc__', '__module__',
        'Playlist', 'Track', 'Genre', 'PlaylistTrack', 'PlaylistId',
        'TrackId', 'GenreId', 'Name', 'Milliseconds', 'UnitPrice',
        'Composer', 'AlbumId', 'MediaTypeId', 'Bytes', 'BillingCountry',
        'BillingCity', 'BillingState', 'BillingAddress', 'BillingPostalCode',
        'Total', 'InvoiceId', 'InvoiceLineId', 'CustomerId', 'EmployeeId',
        'playlist', 'track', 'genre', 'playlist_track', 'playlists',
        'tracks', 'genres', 'iMusic', 'statistics',
        'methods', 'delimiter', 'newline', 'encoding', 'errors',
        'debug', 'port', 'host', 'secret_key',
    })
    BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}

    # Add template function names and constants to BLOCKLIST
    template_funcs = set()
    for dp, _, fn in os.walk(os.path.join(BASE_DIR, 'question')):
        for f in fn:
            if f.endswith('.py'):
                template_funcs.update(re.findall(r'def\s+(\w+)\s*\(',
                                                  read_file(os.path.join(dp, f))))
    all_func_names = set()
    for sub in (all_subs_flat[:10] if all_subs_flat else []):
        all_func_names.update(re.findall(r'def\s+(\w+)\s*\(', sub.get('code', '')))
    CONSTANT_LIKE = frozenset({
        'DB_FILE', 'BASE_DIR', 'UPLOAD_FOLDER', 'Path', 'app',
        'playlist_tracks_file', 'update_playlist_tracks',
    })
    BLOCKLIST = PY_KEYWORDS | SQL_KEYWORDS | KNOWN_APIS | template_funcs | all_func_names | CONSTANT_LIKE

    generalized_count = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex or len(regex) < 10:
            continue
        fs_task = fs.get('task', '')
        task_samples = []
        for batch, subs in all_subs_by_batch.items():
            if BATCH_TASK_MAP.get(batch) == fs_task:
                task_samples = subs
                break
        if not task_samples:
            task_samples = all_subs_flat

        matched_snippets = []
        flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
        try:
            compiled = re.compile(regex, flags)
        except re.error:
            continue
        for sub in task_samples[:15]:
            try:
                m = compiled.search(sub.get('code', ''))
                if m:
                    matched_snippets.append(m.group())
                    if len(matched_snippets) >= 5:
                        break
            except re.error:
                continue

        replaceable = set()
        if len(matched_snippets) >= 2:
            var_candidates = {}
            for snippet in matched_snippets:
                seen = set()
                try:
                    for tok in _tokenize.generate_tokens(_io.StringIO(snippet).readline):
                        if tok.type == _tokenize.NAME and len(tok.string) >= 2:
                            seen.add(tok.string)
                except Exception:
                    continue
                for name in seen:
                    var_candidates[name] = var_candidates.get(name, 0) + 1
            for name, count in var_candidates.items():
                if count >= 2 and name not in BLOCKLIST:
                    replaceable.add(name)

        func_names = set()
        for sub in (task_samples[:5] if task_samples else all_subs_flat[:5]):
            func_names.update(re.findall(r'def\s+(\w+)\s*\(', sub.get('code', '')))
        replaceable -= func_names
        if not replaceable:
            continue

        modified_regex = regex
        for name in sorted(replaceable, key=len, reverse=True):
            modified_regex = re.sub(r'(?<!\\)\b' + re.escape(name) + r'\b',
                                     r'\\w+', modified_regex)
        if modified_regex != regex:
            fs['regex'] = modified_regex
            fs['_generalized_vars'] = sorted(replaceable)
            generalized_count += 1

    print(f'  GENERALIZED {generalized_count} FS' if generalized_count
          else '  No variable names to generalise')

    # Duplicate merge
    print('\n  --- Quality Gate: Duplicate FS Merge ---')
    dup_groups = defaultdict(list)
    for fs in all_fs:
        norm = re.sub(r'\\s\+|\\s\*|\\s\{1,\}|\\s\?', r'\\s*', fs.get('regex', ''))
        norm = re.sub(r'\\d\+|\\d\*|\\d\{1,\}|\\d\?', r'\\d*', norm)
        norm = re.sub(r'\\w\+|\\w\*|\\w\{1,\}|\\w\?', r'\\w*', norm)
        norm = re.sub(r'\s+', '', norm)
        key = (fs.get('criterion', '?'), norm)
        dup_groups[key].append(fs)
    dup_count = 0
    for _, group in dup_groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda f: len(f.get('name', '')), reverse=True)
        keeper = group[0]
        for dup in group[1:]:
            dup['_duplicate_of'] = keeper.get('id', '?')
            dup_count += 1
    print(f'  Merged {dup_count} duplicates' if dup_count else '  No duplicates')

    # Source escape fix
    print('\n  --- Quality Gate: Source-Code Escape Fix ---')
    escape_fix_count = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue
        modified = regex
        if re.search(r"(?:delimiter|split|replace|['\"])\s*[=:]*\s*.*\\t", modified):
            modified = re.sub(r'(?<!\\)\\t', r'\\\\t', modified)
        if re.search(r'\\\)\\s\*:', modified) and '->' not in modified:
            modified = re.sub(r'\\\)\\s\*:', r'\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:', modified)
        if modified != regex:
            fs['regex'] = modified
            fs['_source_escape_fixed'] = True
            escape_fix_count += 1
    print(f'  FIXED {escape_fix_count} escape issues' if escape_fix_count
          else '  No escape issues')

    # Regex structural validation
    print('\n  --- Quality Gate: Regex Structural Validation ---')
    broken_regex = []
    matches_nothing = []
    remove_ids = set()
    for fs in all_fs:
        regex = fs.get('regex', '')
        fid = fs.get('id', '?')
        if not regex:
            broken_regex.append(fid)
            continue
        if re.search(r'\(\?<![^)]*(?:\*|\+|(?:\{[^}]*,\s*[^}]*\}))[^)]*\)', regex):
            broken_regex.append(fid)
            continue
        try:
            flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
            compiled = re.compile(regex, flags)
        except re.error:
            broken_regex.append(fid)
            continue
        fs_task = fs.get('task', '')
        task_samples = []
        for batch, subs in all_subs_by_batch.items():
            if BATCH_TASK_MAP.get(batch) == fs_task:
                task_samples = subs[:5]
                break
        if not task_samples:
            task_samples = all_subs_flat[:5]
        if task_samples and not any(compiled.search(s.get('code', ''))
                                     for s in task_samples if s.get('code', '').strip()):
            matches_nothing.append(fid)
            fs['_warn_matches_nothing'] = True

    if broken_regex:
        print(f'  BROKEN ({len(broken_regex)}): removing')
        remove_ids.update(broken_regex)
    if matches_nothing:
        print(f'  MATCHES_NOTHING ({len(matches_nothing)}): '
              f'KEPT (deletion moved to late-stage gate with larger sample)')
        for fid in matches_nothing:
            quality_report['structurally_broken'].append({
                'fs_id': fid,
                'issue': 'MATCHES_NOTHING_EARLY: matched 0/5 task samples — '
                         'kept for late-stage re-evaluation with larger sample',
            })
    if remove_ids:
        all_fs[:] = [fs for fs in all_fs if fs.get('id', '') not in remove_ids]
        print(f'  After removals: {len(all_fs)} FS')

    # ---- Negative Assertion Position Fix (deterministic) ----
    print('\n  --- Quality Gate: Negative Assertion Position Fix ---')
    try:
        from rule_engine import fix_negative_assertion_position
        assertion_fixed = 0
        for fs in all_fs:
            if fs.get('fs_type') != 'negative':
                continue
            rx = fs.get('regex', '')
            if not rx or '(?!' not in rx:
                continue
            fixed = fix_negative_assertion_position(rx)
            if fixed and fixed != rx:
                try:
                    re.compile(fixed, _parse_flags(fs.get('regex_flags', 'IGNORECASE')))
                    # Verify it doesn't match reference now (shouldn't after fix)
                    fs['regex'] = fixed
                    assertion_fixed += 1
                except re.error:
                    pass
        if assertion_fixed:
            print(f'  FIXED {assertion_fixed} negative assertion positions')
        else:
            print('  No assertion position issues')
    except ImportError:
        print('  Skipped (rule_engine.py not found)')

    # ---- Reference & Template Cross-Check (prevents scoring misjudgments) ----
    print('\n  --- Quality Gate: Reference & Template Cross-Check ---')
    ref_false_negatives = 0
    tpl_false_positives = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue
        flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
        try:
            compiled = re.compile(regex, flags)
        except re.error:
            continue

        # Negative FS must not match reference code (false penalty)
        if fs.get('fs_type') == 'negative' and all_ref_code:
            try:
                if compiled.search(all_ref_code):
                    fs['_warn_ref_match'] = True
                    ref_false_negatives += 1
            except re.error:
                pass

        # Positive FS must not match template code (false reward)
        if fs.get('fs_type') == 'positive' and template_code:
            try:
                if compiled.search(template_code):
                    fs['_warn_matches_template'] = True
                    tpl_false_positives += 1
            except re.error:
                pass

    if ref_false_negatives:
        print(f'  FLAGGED {ref_false_negatives} false negatives '
              f'(negative FS match reference → weight 0.5)')
    if tpl_false_positives:
        print(f'  FLAGGED {tpl_false_positives} false positives '
              f'(positive FS match template → weight 0.0)')
    if not ref_false_negatives and not tpl_false_positives:
        print('  PASS: No ref/template cross-contamination')

    # ---- Broad Negative FS Detection ----
    # Negative FS matching >60% of students provide no discrimination
    print('\n  --- Quality Gate: Broad Negative FS Detection ---')
    broad_neg_count = 0
    for fs in all_fs:
        if fs.get('fs_type') != 'negative':
            continue
        rx = fs.get('regex', '')
        if not rx:
            continue
        fs_task = fs.get('task', '')
        task_subs = [s for batch, subs in all_subs_by_batch.items()
                     if BATCH_TASK_MAP.get(batch) == fs_task
                     for s in (subs if isinstance(subs, list) else subs.values())]
        if not task_subs:
            continue
        matches = 0
        for sub in task_subs:
            code = sub.get('code', '') if isinstance(sub, dict) else str(sub)
            try:
                if re.search(rx, code, _parse_flags(fs.get('regex_flags', 'IGNORECASE'))):
                    matches += 1
            except re.error:
                pass
        if matches > len(task_subs) * 0.4:
            fs['_warn_broad_negative'] = True
            broad_neg_count += 1
    if broad_neg_count:
        print(f'  FLAGGED {broad_neg_count} overly-broad negative FS (weight 0.0)')
    else:
        print('  No overly-broad negative FS')

    # Assign scoring weights (incorporates ref/template flags from above)
    WARNING_WEIGHT_MAP = {
        '_warn_ref_match': 0.5, '_warn_broad_positive': 0.3,
        '_warn_narrow_positive': 0.3, '_warn_matches_nothing': 0.0,
        '_warn_type_annotation': 0.5, '_warn_feedback_mismatch': 0.7,
        '_warn_matches_template': 0.0, '_warn_literal_fallback': 0.0,
        '_warn_overlaps_with_negative': 0.5, '_warn_overlaps_with_positive': 0.5,
        '_duplicate_of': 0.0,
        '_warn_broad_negative': 0.0,
    }
    for fs in all_fs:
        weight = 1.0
        for wk, wp in WARNING_WEIGHT_MAP.items():
            if fs.get(wk):
                weight = min(weight, wp)
        fs['_scoring_weight'] = weight

    w0 = sum(1 for fs in all_fs if fs.get('_scoring_weight', 1.0) == 0.0)
    w1 = sum(1 for fs in all_fs if fs.get('_scoring_weight', 1.0) == 1.0)
    print(f'  Scoring: {w1} full-weight, {w0} excluded')


# Re-export coverage functions for convenience
from coverage import run_coverage_check, find_gaps, format_coverage_report, build_multi_criterion_supplement_prompt
from coverage import _parse_flags as _coverage_parse_flags
_parse_flags = _coverage_parse_flags


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        # Usage: python ai_pipeline.py <question_dir> <submissions_dir> [question_id]
        q_dir = sys.argv[1]
        s_dir = sys.argv[2]
        qid = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.abspath(q_dir))
        refs = os.path.join(BASE_DIR, 'references', qid) if os.path.isdir(
            os.path.join(BASE_DIR, 'references', qid)) else ''
        run_pipeline(q_dir, s_dir, qid, ref_dir=refs)
    elif len(sys.argv) == 2:
        # Backward compat: python ai_pipeline.py <config.yaml>
        config = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
        q_dir = os.path.join(BASE_DIR, config.get('question_dir', 'question'))
        s_dir = os.path.join(BASE_DIR, config.get('submissions_path', 'submission'))
        refs = os.path.join(BASE_DIR, config.get('references_path', ''))
        run_pipeline(q_dir, s_dir, config['question_id'],
                     question_name=config.get('question_name', ''),
                     ref_dir=refs,
                     student_prefix=config.get('student_prefix'))
    else:
        print('Usage: python ai_pipeline.py <question_dir> <submissions_dir> [question_id]')
        print('   or: python ai_pipeline.py <config.yaml>')
        sys.exit(1)
