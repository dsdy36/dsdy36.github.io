"""
Ground Truth Module — README.md Processing for FS Generation
=============================================================
Parses CW-generated README.md files to extract known good/bad patterns.
Provides per-task, per-student pattern lookups for AI FS generation.
Also verifies README-code consistency.
"""
import os
import re
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, 'ground_truth')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submission')

# Criterion ID → Task mapping
CRITERION_TASK_MAP = {
    'RQ1_1': 'Task1', 'RQ1_2': 'Task1', 'RQ1_3': 'Task1', 'RQ1_4': 'Task1',
    'RQ2_1': 'Task2', 'RQ2_2': 'Task2', 'RQ2_3': 'Task2', 'RQ2_4': 'Task2',
    'RQ3_1': 'Task3', 'RQ3_2': 'Task3', 'RQ3_3': 'Task3',
    'RQ3_4': 'Task3', 'RQ3_5': 'Task3', 'RQ3_6': 'Task3',
}


def parse_readme(readme_text: str) -> dict[str, dict[str, list[str]]]:
    """Parse a single README.md into structured good/bad patterns per criterion.

    Returns:
        {criterion_id: {'good': [...], 'bad': [...]}}
        'bad' includes both Error pattern and Mistake to include sections.
    """
    patterns: dict[str, dict[str, list[str]]] = defaultdict(lambda: {'good': [], 'bad': [], 'mistake': []})

    # Find all criterion sections: ### RQx_y: Name (marks)
    sections = re.split(r'\n### (RQ\d+_\d+):', readme_text)
    # sections[0] = header before first criterion, then alternating criterion_id, content
    for i in range(1, len(sections), 2):
        if i + 1 > len(sections):
            break
        crit_id = sections[i].strip()
        content = sections[i + 1]

        # Extract Correct approach (plain text format)
        good_match = re.search(
            r'\*\*Correct approach:?\*\*\s*\n((?:\s*-.*\n?)*)',
            content
        )
        if good_match:
            for line in good_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- ['):
                    # Remove the variant letter tag: "- [A] description" -> "description"
                    desc = re.sub(r'^-\s*\[[A-Z]\]\s*', '', line).strip()
                    if desc and desc not in patterns[crit_id]['good']:
                        patterns[crit_id]['good'].append(desc)

        # Extract Error pattern (plain text format)
        bad_match = re.search(
            r'\*\*Error pattern:?\*\*\s*\n((?:\s*-.*\n?)*)',
            content
        )
        if bad_match:
            for line in bad_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- ['):
                    desc = re.sub(r'^-\s*\[[A-Z]\]\s*', '', line).strip()
                    if desc and desc not in patterns[crit_id]['bad']:
                        patterns[crit_id]['bad'].append(desc)

        # Extract Mistake to include — stored separately from Error patterns.
        # Mistake patterns may appear in reference code (e.g., INSERT OR IGNORE).
        # They should only be flagged when used WITHOUT required companion checks.
        mistake_match = re.search(
            r'\*\*Mistake to include:?\*\*\s*\n((?:\s*-.*\n?)*)',
            content
        )
        if mistake_match:
            for line in mistake_match.group(1).strip().split('\n'):
                line = line.strip()
                if line.startswith('- ['):
                    desc = re.sub(r'^-\s*\[[A-Z]\]\s*', '', line).strip()
                    if desc and desc not in patterns[crit_id]['mistake']:
                        patterns[crit_id]['mistake'].append(desc)

    return dict(patterns)


def load_all_readmes(source_dir: str = '') -> dict[str, dict]:
    """Load and parse all README.md files from student directories.

    Auto-detects: tries submission/ first (CW format), then ground_truth/.

    Returns:
        {student_id: {
            'quality_tier': str,
            'correct_count': int,
            'error_count': int,
            'criteria': {criterion_id: {'good': [...], 'bad': [...]}},
        }}
    """
    if source_dir:
        search_dir = source_dir
    elif os.path.isdir(SUBMISSION_DIR):
        # Check if submission dir has README files (CW format)
        sample = os.listdir(SUBMISSION_DIR)[:1]
        if sample and os.path.exists(os.path.join(SUBMISSION_DIR, sample[0], 'README.md')):
            search_dir = SUBMISSION_DIR
        else:
            search_dir = GROUND_TRUTH_DIR
    else:
        search_dir = GROUND_TRUTH_DIR

    all_data = {}
    if not os.path.isdir(search_dir):
        return all_data

    for sid in sorted(os.listdir(search_dir)):
        student_dir = os.path.join(search_dir, sid)
        if not os.path.isdir(student_dir):
            continue
        readme_path = os.path.join(student_dir, 'README.md')
        if not os.path.exists(readme_path):
            continue

        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()

        criteria = parse_readme(text)

        # Extract metadata from header table
        quality = 'medium'
        correct_count = 0
        error_count = 0
        qm = re.search(r'\|\s*Quality Tier\s*\|\s*\*{0,2}(\w+)\*{0,2}\s*\|', text)
        if qm:
            quality = qm.group(1)
        cm = re.search(r'\|\s*Correct Patterns\s*\|\s*(\d+)\s*\|', text)
        if cm:
            correct_count = int(cm.group(1))
        em = re.search(r'\|\s*Error Patterns\s*\|\s*(\d+)\s*\|', text)
        if em:
            error_count = int(em.group(1))

        all_data[sid] = {
            'quality_tier': quality,
            'correct_count': correct_count,
            'error_count': error_count,
            'criteria': criteria,
        }

    return all_data


def build_pattern_inventory(all_readmes: dict[str, dict]) -> dict[str, dict[str, list[str]]]:
    """Aggregate patterns across all students into a deduplicated inventory per task.

    Returns:
        {task_id: {'good': [pattern_desc, ...], 'bad': [pattern_desc, ...]}}
    """
    inventory: dict[str, dict[str, set[str]]] = defaultdict(lambda: {'good': set(), 'bad': set()})

    for sid, data in all_readmes.items():
        for crit_id, patterns in data.get('criteria', {}).items():
            task = CRITERION_TASK_MAP.get(crit_id, 'Unknown')
            for g in patterns.get('good', []):
                inventory[task]['good'].add(g)
            for b in patterns.get('bad', []):
                inventory[task]['bad'].add(b)

    # Convert sets to sorted lists
    return {
        task: {
            'good': sorted(patterns['good']),
            'bad': sorted(patterns['bad']),
        }
        for task, patterns in inventory.items()
    }


def get_task_patterns(all_readmes: dict[str, dict], task_id: str) -> dict[str, list[str]]:
    """Get deduplicated good/bad patterns for a specific task."""
    inventory = build_pattern_inventory(all_readmes)
    return inventory.get(task_id, {'good': [], 'bad': []})


def get_student_task_patterns(student_id: str, task_id: str,
                               all_readmes: dict[str, dict]) -> dict[str, list[str]]:
    """Get good/bad patterns for a specific student × task.

    Filters the student's README criteria to only those belonging to the given task.
    """
    data = all_readmes.get(student_id, {})
    good = []
    bad = []
    for crit_id, patterns in data.get('criteria', {}).items():
        if CRITERION_TASK_MAP.get(crit_id) == task_id:
            good.extend(patterns.get('good', []))
            bad.extend(patterns.get('bad', []))
    return {'good': sorted(set(good)), 'bad': sorted(set(bad))}


def format_patterns_for_prompt(task_patterns: dict[str, list[str]], task_id: str) -> str:
    """Format task-level patterns as a prompt section for AI.

    Args:
        task_patterns: Output of get_task_patterns().
        task_id: e.g., 'Task1'.
    """
    good = task_patterns.get('good', [])
    bad = task_patterns.get('bad', [])

    lines = [f'## GROUND TRUTH — Known patterns for {task_id}']
    lines.append('The following patterns are KNOWN to exist in these student submissions.')
    lines.append('Your job: generate FS that match these specific patterns.')

    if good:
        lines.append('\n### Patterns that SHOULD be matched by POSITIVE FS:')
        for i, g in enumerate(good, 1):
            lines.append(f'  {i}. {g}')

    if bad:
        lines.append('\n### Patterns that SHOULD be matched by NEGATIVE FS:')
        for i, b in enumerate(bad, 1):
            lines.append(f'  {i}. {b}')

    if not good and not bad:
        lines.append('\n(No ground truth patterns found — generate FS from code analysis only.)')

    lines.append(f'\nIMPORTANT: Generate at least one FS for EACH pattern listed above.')
    return '\n'.join(lines)


# ─── README-Code Consistency Verification ─────────────────────────

# Simple keyword-based checks for the most common patterns
# Maps pattern description keywords → code checks
_CONSISTENCY_CHECKS = [
    # (description_keyword, code_must_contain, code_must_not_contain, check_type)
    # check_type: 'good' means code SHOULD have it, 'bad' means code SHOULD have it

    # Task 1 good patterns
    ('csv.DictReader', 'csv.DictReader', None, 'good'),
    ('csv.reader', 'csv.reader', None, 'good'),
    ("delimiter=", 'delimiter', None, 'good'),
    ('conn.close()', '.close()', None, 'good'),
    ('INSERT OR IGNORE', 'INSERT OR IGNORE', None, 'good'),
    ('parameterized', '?', None, 'good'),
    ('.commit()', '.commit()', None, 'good'),

    # Task 1 bad patterns
    ('pandas', 'import pandas', None, 'bad'),
    ('Hardcode the file path', None, None, 'bad'),  # hard to verify automatically
    ("split each line by", '.split(', None, 'bad'),
    ('Manual.*pars', '.split(', None, 'bad'),  # heuristic

    # Task 2 good patterns
    ('"All"', '"All"', None, 'good'),
    ('GenreId.*0', 'GenreId', None, 'good'),  # weak signal

    # Task 3 good patterns
    ('NOT IN', 'NOT IN', None, 'good'),

    # Task 3 bad patterns
    ('f-string', 'f"', None, 'bad'),
    ('% formatting', '%', None, 'bad'),
    ('string concatenation.*INSERT', '+', None, 'bad'),
]


def verify_student_readme(student_id: str, task_codes: dict[int, str],
                          all_readmes: dict[str, dict]) -> dict:
    """Verify that a student's README claims match their actual code.

    Args:
        student_id: e.g., 'S001'.
        task_codes: {1: code, 2: code, 3: code} — raw task files.
        all_readmes: Output of load_all_readmes().

    Returns:
        {student_id: {'mismatches': [...], 'verified': bool}}
        Each mismatch: {'criterion': str, 'type': 'good'|'bad', 'description': str,
                        'issue': 'code_missing_expected'|'code_has_unexpected'}
    """
    data = all_readmes.get(student_id, {})
    mismatches = []

    for crit_id, patterns in data.get('criteria', {}).items():
        task_num = int(crit_id[2])  # RQ1_1 → 1
        code = task_codes.get(task_num, '')

        for g in patterns.get('good', []):
            for keyword, must_have, _, check_type in _CONSISTENCY_CHECKS:
                if check_type == 'good' and keyword.lower() in g.lower():
                    if must_have and must_have not in code:
                        mismatches.append({
                            'criterion': crit_id,
                            'type': 'good',
                            'description': g[:80],
                            'issue': 'code_missing_expected',
                            'detail': f'README claims "{keyword}" but code does not contain "{must_have}"',
                        })
                        break  # one mismatch per pattern

        for b in patterns.get('bad', []):
            for keyword, must_have, _, check_type in _CONSISTENCY_CHECKS:
                if check_type == 'bad' and keyword.lower() in b.lower():
                    if must_have and must_have not in code:
                        mismatches.append({
                            'criterion': crit_id,
                            'type': 'bad',
                            'description': b[:80],
                            'issue': 'code_missing_expected',
                            'detail': f'README claims "{keyword}" but pattern not found in code',
                        })
                        break

    return {
        'student_id': student_id,
        'mismatches': mismatches,
        'verified': len(mismatches) == 0,
    }


def verify_all_readmes(all_readmes: dict[str, dict],
                       submissions_source: str = '') -> dict[str, dict]:
    """Verify README-code consistency for all students.

    Args:
        all_readmes: Output of load_all_readmes().
        submissions_source: Path to CW submissions dir (task1/2/3.py per student).
                           If empty, skips verification.

    Returns:
        {student_id: verification_result, ...}
    """
    results = {}
    if not submissions_source or not os.path.isdir(submissions_source):
        return results

    for sid, data in all_readmes.items():
        student_dir = os.path.join(submissions_source, sid)
        if not os.path.isdir(student_dir):
            continue

        task_codes = {}
        for tn in [1, 2, 3]:
            tf = os.path.join(student_dir, f'task{tn}.py')
            if os.path.exists(tf):
                with open(tf, 'r', encoding='utf-8') as f:
                    task_codes[tn] = f.read()

        results[sid] = verify_student_readme(sid, task_codes, all_readmes)

    return results


def print_verification_report(results: dict[str, dict]):
    """Print a summary of README-code verification results."""
    total = len(results)
    verified = sum(1 for r in results.values() if r['verified'])
    failed = total - verified
    total_mismatches = sum(len(r['mismatches']) for r in results.values())

    print(f'\n  README-Code Verification: {verified}/{total} passed, '
          f'{failed} with mismatches ({total_mismatches} total)')

    if failed > 0:
        print(f'\n  Students with mismatches:')
        for sid, r in sorted(results.items()):
            if not r['verified']:
                print(f'    {sid}: {len(r["mismatches"])} mismatches')
                for m in r['mismatches'][:3]:
                    print(f'      [{m["criterion"]}] {m["type"]}: {m["detail"]}')


if __name__ == '__main__':
    # Quick test
    all_data = load_all_readmes()
    print(f'Loaded {len(all_data)} README files')

    inventory = build_pattern_inventory(all_data)
    for task in ['Task1', 'Task2', 'Task3']:
        inv = inventory.get(task, {})
        print(f'\n{task}: {len(inv.get("good", []))} good, '
              f'{len(inv.get("bad", []))} bad patterns')

    # Verify consistency
    cw_dir = os.path.join(BASE_DIR, '..', 'CW-generater', 'submissions_imusic_v5')
    results = verify_all_readmes(all_data, cw_dir)
    print_verification_report(results)
