"""
TAFFIES-Aligned FS Generator
=============================
Generates narrow FS from clustered student code patterns.
Follows the TAFFIES paper methodology (§3.2):

  1. Look at actual student code FIRST
  2. Find SPECIFIC patterns (good or bad) — not categories
  3. Write one narrow FS per specific pattern
  4. FCC iteration for remaining uncovered students
  5. Validate against reference + template

ALL FS follow the TAFFIES definition: "签名—反馈" pairing where the
signature detects a SPECIFIC pattern found in actual student code.
NO broad Type A patterns, NO rubric-inferred categories.

Usage:
    from taffies_fs_generator import generate_taffies_fs
    all_fs = generate_taffies_fs(tasks, task_subs, all_readmes, ref_code, template_code)
"""

import json
import os
import re
import textwrap
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

from ai_pipeline import call_deepseek, extract_json, _repair_json, read_file, collect_submissions_by_task
from coverage import run_coverage_check, find_gaps, _parse_flags
from ground_truth import load_all_readmes


# ============================================================
# System prompts — AI generates Python check functions
# ============================================================

NEGATIVE_FS_SYSTEM = """You are an automated grading assistant using the TAFFIES
(Tailored Automated Feedback Framework) methodology.

Your job: write ONE Python check function that detects a SPECIFIC mistake
pattern found in actual student code.

The function signature is ALWAYS:
    def check(code: str) -> tuple:
        # Return (True, "evidence: found f-string SQL in execute()") if the bad pattern IS found
        # Return (False, "") if not found
        ...

CRITICAL RULES — VIOLATING ANY IS AN ERROR:

1. FUNCTION SCOPING (MANDATORY):
   The `code` parameter contains ONLY the relevant student-written function(s).
   You MUST search within the target function, NOT the entire file.
   - If using ast: find the target FunctionDef node FIRST, then ast.walk() THAT NODE.
     Example: for node in ast.walk(tree): if isinstance(node, ast.FunctionDef) and node.name == 'TARGET_FUNC': ...
   - If using re.search: the code is already scoped to the target function. Search it directly.
   - NEVER use ast.walk(tree) on the root — always filter to the target function first.

2. DETECT PRESENCE OF BAD CODE — NEVER detect ABSENCE of good code.
   WRONG: return ('commit()' not in code, "")              <- Type A, detects absence
   WRONG: return (not re.search(...), "")                  <- Type A, detects absence
   RIGHT: return ('pd.read_csv' in code, "used pandas")    <- Type B, detects presence
   Your check function MUST return True when bad code IS FOUND.
   If you find yourself writing "not in", "return not", STOP — you are writing Type A.

3. NARROW SCOPE: ONE specific mistake. Not a category.
4. You may use: re.search, ast.parse, ast.walk, string methods.
5. You must NOT use: os, subprocess, open, exec, eval, __import__
6. Handle SyntaxError from ast.parse gracefully — return (False, "").
7. SPECIFIC FEEDBACK: Name exact mistake and exact fix.
8. RETURN EVIDENCE: The second element of the tuple MUST be a short string
   showing WHAT matched (e.g., the matched code snippet, function name, or pattern name).
   This is the proof that the pattern was found. Max 200 characters.

Examples of GOOD check functions:

For "used pandas instead of csv module" (scoped — searches only update_playlist_tracks):
```python
def check(code: str) -> tuple:
    # code is the update_playlist_tracks function body
    if 'import pandas' in code:
        return (True, "import pandas")
    if 'from pandas' in code:
        return (True, "from pandas import")
    if 'pd.read_csv' in code:
        return (True, "pd.read_csv")
    return (False, "")
```

For "f-string used in SQL query inside cursor.execute()" (scoped via ast):
```python
def check(code: str) -> tuple:
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr') and node.func.attr == 'execute':
                    for arg in node.args:
                        if isinstance(arg, ast.JoinedStr):
                            return (True, "f-string in cursor.execute()")
        return (False, "")
    except SyntaxError:
        return (False, "")
```

For "missing conn.commit()" (Type B — detects INSERT without subsequent commit):
```python
def check(code: str) -> tuple:
    import ast
    try:
        tree = ast.parse(code)
        has_insert = False
        has_commit = False
        insert_evidence = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr') and node.func.attr == 'execute':
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and 'INSERT' in arg.value:
                            has_insert = True
                            insert_evidence = arg.value[:80]
                if hasattr(node.func, 'attr') and node.func.attr == 'commit':
                    has_commit = True
        if has_insert and not has_commit:
            return (True, f"INSERT without commit: {insert_evidence}")
        return (False, "")
    except SyntaxError:
        return (False, "")
```

Output ONLY valid JSON:
{{"check_function": "def check(code):\\n    ...", "name": "...", "feedback": "...", "marks": -2}}"""


POSITIVE_FS_SYSTEM = """You are an automated grading assistant using the TAFFIES
(Tailored Automated Feedback Framework) methodology.

Your job: write ONE Python check function that detects a SPECIFIC correct
implementation pattern found in actual student code.

The function signature is ALWAYS:
    def check(code: str) -> tuple:
        # Return (True, "evidence: found csv.DictReader with delimiter") if the correct pattern IS found
        # Return (False, "") if not found
        ...

CRITICAL RULES — VIOLATING ANY IS AN ERROR:

1. FUNCTION SCOPING (MANDATORY):
   The `code` parameter contains ONLY the relevant student-written function(s).
   You MUST search within the target function, NOT the entire file.
   - If using ast: find the target FunctionDef node FIRST, then ast.walk() THAT NODE.
   - If using re.search: the code is already scoped. Search it directly.
   - NEVER use ast.walk(tree) on the root without filtering to the target function.

2. DETECT PRESENCE OF A SPECIFIC CORRECT PATTERN — not "has any code."
   WRONG: return (len(code) > 50, "")                   <- matches any code
   RIGHT: return ('csv.DictReader' in code, "csv.DictReader")  <- matches specific pattern

3. NARROW SCOPE: ONE specific correct approach, not a category.

4. MUST NOT MATCH TEMPLATE/STUB CODE:
   - The prompt will show you which functions are template-provided vs student-written
   - Do NOT match template-provided functions (their names are listed in the prompt)
   - Do not match functions that are pass-only stubs
   - Do not match imports or Flask boilerplate
   - If code is just 'pass', return (False, "")

5. NEVER use "not in" to detect ABSENCE — only detect PRESENCE.

6. You may use: re.search, ast.parse, ast.walk, string methods.
7. You must NOT use: os, subprocess, open, exec, eval, __import__
8. Handle SyntaxError from ast.parse gracefully — return (False, "").
9. SPECIFIC FEEDBACK: Name exact technique and WHY it's correct.
10. RETURN EVIDENCE: The second element of the tuple MUST be a short string
   showing WHAT matched (e.g., the matched code snippet, function name, or pattern name).
   This is the proof that the pattern was found. Max 200 characters.

Examples of GOOD check functions:

For "csv.DictReader with delimiter='\\t'" (correctly scoped):
```python
def check(code: str) -> tuple:
    # code is the target function body
    if len(code.strip()) < 20:
        return (False, "")  # stub
    if 'csv.DictReader' in code and 'delimiter=' in code and '\\\\t' in code:
        return (True, "csv.DictReader with tab delimiter")
    return (False, "")
```

For "parameterized INSERT with ? placeholders" (correctly scoped):
```python
def check(code: str) -> tuple:
    import re
    if len(code.strip()) < 20:
        return (False, "")
    m = re.search(r'\\.execute\\s*\\(\\s*[\"\\x27].*INSERT.*\\?.*[\"\\x27]', code, re.I | re.DOTALL)
    if m:
        return (True, m.group(0)[:100])
    return (False, "")
```

Output ONLY valid JSON:
{{"check_function": "def check(code):\\n    ...", "name": "...", "feedback": "...", "marks": 2}}"""


# ============================================================
# Behavioral enrichment for FS prompts
# ============================================================

def _build_behavior_context(
    criterion: str, cluster: dict, behavioral_fingerprints: dict | None = None
) -> str:
    """Build a behavioral observation block for the FS prompt.

    When we have runtime test results for this criterion, we tell the AI
    exactly what behavior was observed — this is ground truth that helps
    the AI write more precise check functions.

    Returns empty string if no behavioral data available.
    """
    if not behavioral_fingerprints:
        return ''

    rep_sid = cluster.get('representative_student', '')
    fp = behavioral_fingerprints.get(rep_sid, {}).get(criterion, {})
    if not fp:
        return ''

    # Build behavior summary
    vuln = fp.get('vulnerable')
    used_safe = fp.get('used_safe_default')
    has_fstring = fp.get('has_fstring_sql')
    has_params = fp.get('has_params')
    has_order = fp.get('has_order_by_name')
    sorted_ok = fp.get('sorted_correctly')
    exception_caught = fp.get('exception_caught')
    used_default = fp.get('used_safe_default')
    query_count = fp.get('query_count', 0)
    exit_code = fp.get('exit_code', 0)

    parts = ['\n### RUNTIME BEHAVIOR OBSERVATION (GROUND TRUTH)']
    parts.append('When executed with test inputs, this code showed:')

    if vuln is True:
        parts.append(f'  - Test result: VULNERABLE — the bad pattern WAS confirmed')
        if criterion == 'RQ2_3':
            parts.append('  - SQL injection with "1; DROP TABLE Playlist--" SUCCEEDED')
            parts.append('  - The sort_column was interpolated directly into SQL without validation')
        elif criterion == 'RQ2_1':
            parts.append('  - SQL queries use f-string/concatenation without parameterization')
            if has_fstring:
                parts.append('  - f-string SQL detected in execute() call')
        elif criterion == 'RQ1_3':
            parts.append('  - IntegrityError was NOT caught — no try/except')
    elif vuln is False:
        parts.append(f'  - Test result: SAFE — the code correctly handles the criterion')
        if criterion == 'RQ2_3':
            parts.append('  - SQL injection was BLOCKED by allowlist validation')
            if used_safe:
                parts.append('  - Code defaults to safe column when invalid input is given')
        elif criterion == 'RQ2_1':
            parts.append('  - SQL queries use parameterized ? placeholders')
            if has_params:
                parts.append('  - Parameters are properly passed to execute()')
        elif criterion == 'RQ3_1':
            if has_order:
                parts.append('  - Query includes ORDER BY Name')
            if sorted_ok:
                parts.append('  - Results are correctly sorted by name')
        elif criterion == 'RQ1_3':
            if exception_caught:
                parts.append('  - IntegrityError is caught with try/except')

    if exit_code != 0:
        parts.append(f'  - Exit code: {exit_code} (code may have errors)')

    parts.append(f'  - SQL queries generated: {query_count}')
    parts.append('\nThis is GROUND TRUTH from actual code execution.')
    parts.append('Use this to write a precise check function targeting the EXACT pattern.')

    return '\n'.join(parts)


# ============================================================
# Prompt builders
# ============================================================

def _find_criterion(criterion: str, rubric_criteria: list[dict]) -> dict:
    for rc in rubric_criteria:
        if rc.get('id') == criterion:
            return rc
    return {}


def _get_template_context(template_code: str) -> str:
    """Extract template function names dynamically from template code."""
    if not template_code:
        return ''
    func_names = set(re.findall(r'def\s+(\w+)\s*\(', template_code))
    student_funcs = STUDENT_FUNCTIONS
    template_funcs = sorted(func_names - student_funcs)
    if not template_funcs:
        return ''
    return (
        '### Template-provided functions (pre-written, NOT student code):\n'
        f'  {", ".join(template_funcs)}\n'
        'These are in the starter template. Do NOT generate FS that match these.\n'
        'Only match patterns in student-written functions.\n'
    )


def build_negative_fs_prompt(criterion: str, rubric_criteria: list[dict],
                              pattern_label: str,
                              all_student_codes: list[tuple[str, str]],
                              template_context: str = '',
                              is_mistake: bool = False,
                              behavioral_context: str = '') -> str:
    crit = _find_criterion(criterion, rubric_criteria)
    target_func = CRIT_FUNC.get(criterion, ['update_playlist_tracks'])
    target_str = ', '.join(target_func)
    mistake_note = ''
    if is_mistake:
        mistake_note = (
            '\n### [Mistake to include — NOT always wrong]\n'
            'This pattern may appear in correct code too (e.g., reference solutions).\n'
            'Only flag it when the student uses this WITHOUT the required companion check.\n'
            'Example: flag INSERT OR IGNORE only when there is NO prior SELECT COUNT validation.\n'
        )

    # Build multi-student code blocks
    n_students = len(all_student_codes)
    code_blocks = []
    for i, (sid, code) in enumerate(all_student_codes):
        label = " (representative)" if i == 0 else ""
        code_blocks.append(f"#### {sid}{label}:\n```python\n{code}\n```")
    all_code_text = '\n\n'.join(code_blocks)

    return f"""## Criterion: {criterion} -- {crit.get('name', criterion)}
{mistake_note}
### Correct approach (rubric):
{chr(10).join(f'- {gp}' for gp in crit.get('good_patterns', []))}

### Target student-written function(s): {target_str}
{template_context}
### ALL students ({n_students} total) who made this specific mistake:
**"{pattern_label}"**
{behavioral_context}
### Their code (ONLY {target_str} function body, COMPLETE — no truncation):

{all_code_text}

## Task
Write ONE Python check function that detects THIS specific mistake across ALL {n_students} students shown above.

Requirements:
- def check(code: str) -> tuple:
- `code` IS the {target_str} function body — search it directly, no file-level walk
- Return (True, "evidence") ONLY when this specific mistake is found
- Return (False, "") otherwise — do NOT detect ABSENCE
- Template functions listed above are NOT in this code — ignore them
- Feedback names the specific mistake and gives concrete fix
- marks: negative integer representing point deduction (e.g., -2 for a moderate mistake, -5 for severe)

Output ONLY JSON:
{{"check_function": "def check(code):\\n    ...", "name": "...", "feedback": "...", "marks": -2}}"""


def build_positive_fs_prompt(criterion: str, rubric_criteria: list[dict],
                              pattern_label: str,
                              all_student_codes: list[tuple[str, str]],
                              template_context: str = '',
                              behavioral_context: str = '') -> str:
    crit = _find_criterion(criterion, rubric_criteria)
    target_func = CRIT_FUNC.get(criterion, ['update_playlist_tracks'])
    target_str = ', '.join(target_func)

    # Build multi-student code blocks
    n_students = len(all_student_codes)
    code_blocks = []
    for i, (sid, code) in enumerate(all_student_codes):
        label = " (representative)" if i == 0 else ""
        code_blocks.append(f"#### {sid}{label}:\n```python\n{code}\n```")
    all_code_text = '\n\n'.join(code_blocks)

    return f"""## Criterion: {criterion} -- {crit.get('name', criterion)}

### What the rubric requires:
{chr(10).join(f'- {gp}' for gp in crit.get('good_patterns', []))}

### Target student-written function(s): {target_str}
{template_context}
### ALL students ({n_students} total) who correctly implemented:
**"{pattern_label}"**
{behavioral_context}
### Their code (ONLY {target_str} function body, COMPLETE — no truncation):

{all_code_text}

## Task
Write ONE Python check function that detects THIS specific correct pattern across ALL {n_students} students shown above.

Requirements:
- def check(code: str) -> tuple:
- `code` IS the {target_str} function body — search it directly
- Return (True, "evidence") ONLY when this specific correct pattern is found
- Return (False, "") otherwise
- Do NOT match pass-only stubs (check len(code) > 50)
- Template functions listed above are NOT in this code — ignore them
- Feedback names the specific technique and explains WHY it's correct
- marks: positive integer representing point award (e.g., 2 for a minor technique, 5 for major)

Output ONLY JSON:
{{"check_function": "def check(code):\\n    ...", "name": "...", "feedback": "...", "marks": 2}}"""


# ============================================================
# Clustering
# ============================================================

def _normalize_label(label: str) -> str:
    label = re.sub(r'^\[[A-Z]\]\s*', '', label)
    label = re.sub(r'\s+', ' ', label).strip().lower()
    label = label.replace('f-string', 'fstring').replace('f string', 'fstring')
    label = label.replace('parameterised', 'parameterized')
    return label


# Only these functions are student-written. All others (upload_route, index,
# main, page_not_found, and Flask route functions) are template-provided.
STUDENT_FUNCTIONS = frozenset({
    'update_playlist_tracks',
    'get_all_genres', 'get_statistics',
    'get_all_playlists', 'create_playlist', 'rename_playlist',
    'delete_playlist', 'add_tracks_by_genre', 'remove_tracks_by_genre',
})


def _call_check_fn(fn, code: str) -> tuple:
    """Call a check function, handling both (bool, str) and legacy bool returns.

    Returns (matched: bool, evidence: str). Compatible with:
      - New check functions returning (bool, str)
      - Legacy check functions returning bool
      - Any truthy/falsey value
    """
    try:
        result = fn(code)
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1])
        return bool(result), ""
    except Exception:
        return False, ""

# Criterion → which student functions are relevant
CRIT_FUNC = {
    'RQ1_1': ['update_playlist_tracks'], 'RQ1_2': ['update_playlist_tracks'],
    'RQ1_3': ['update_playlist_tracks'], 'RQ1_4': ['update_playlist_tracks'],
    'RQ2_1': ['get_all_genres'], 'RQ2_2': ['get_all_genres'],
    'RQ2_3': ['get_statistics'], 'RQ2_4': ['get_statistics'],
    'RQ3_1': ['get_all_playlists'], 'RQ3_2': ['create_playlist'],
    'RQ3_3': ['rename_playlist'], 'RQ3_4': ['delete_playlist'],
    'RQ3_5': ['add_tracks_by_genre'], 'RQ3_6': ['remove_tracks_by_genre'],
}


def _extract_student_functions(code: str, criterion: str = '') -> str:
    """Extract ONLY student-written functions from code.
    Strips all template-provided functions (upload_route, index, main, Flask routes).
    Returns concatenated, COMPLETE function bodies. No truncation.

    If criterion is specified, only extracts the functions relevant to that criterion.
    Otherwise extracts ALL student functions.
    """
    target = CRIT_FUNC.get(criterion, []) if criterion else list(STUDENT_FUNCTIONS)
    if not target:
        target = list(STUDENT_FUNCTIONS)

    parts = []
    for fname in target:
        # Match: optional decorator + def func_name(params): + entire body until next def
        pattern = (
            r'(?:@[^\n]+\n\s*)?def\s+' + re.escape(fname) +
            r'\s*\([^)]*\)\s*(?:->\s*\w+\s*)?\s*:.*?'
            r'(?=\n(?:@[^\n]+\n\s*)?def\s+\w+\s*\(|\Z)'
        )
        m = re.search(pattern, code, re.DOTALL)
        if m:
            body = m.group().strip()
            if len(body) > 20:
                parts.append(body)
    return '\n\n'.join(parts)


def _pick_representative(sids: list[str], sub_lookup: dict[str, str],
                        criterion: str = '') -> tuple[str | None, str]:
    """Pick representative student and extract ONLY student-written functions.
    Sends COMPLETE function bodies (no truncation), template code excluded.
    """
    for sid in sorted(sids):
        code = sub_lookup.get(sid, '')
        if not code or len(code.strip()) < 30:
            continue

        extracted = _extract_student_functions(code, criterion)
        if extracted and len(extracted.strip()) > 30:
            return sid, extracted

        # Fallback: if extraction produced nothing, try all student functions
        extracted = _extract_student_functions(code, '')
        if extracted and len(extracted.strip()) > 30:
            return sid, extracted

    return None, ''


def cluster_by_pattern(
    all_readmes: dict, task_subs: dict[int, list[dict]],
    criterion: str, pattern_type: str = 'bad',
    min_cluster_size: int = 2,
) -> list[dict]:
    """Cluster students by their specific good or bad pattern for a criterion.

    Args:
        pattern_type: 'good' or 'bad'
        min_cluster_size: Only return clusters with >= this many students.
                          Smaller clusters are left for FCC iteration.

    Returns:
        [{label, students, count, representative_student, representative_code}, ...]
    """
    task_num = int(re.search(r'(\d)', criterion).group(1))
    subs = task_subs.get(task_num, [])
    sub_lookup = {s['student']: s.get('code', '') for s in subs}

    pattern_groups: dict[str, list[str]] = defaultdict(list)

    for sid, rdata in all_readmes.items():
        criteria = rdata.get('criteria', {})
        if isinstance(criteria, str):
            try: criteria = eval(criteria)
            except Exception: continue
        if not isinstance(criteria, dict):
            continue

        crit_data = criteria.get(criterion, {})
        pattern_list = crit_data.get(pattern_type, [])
        for desc in pattern_list:
            if desc.strip():
                key = _normalize_label(desc.strip())
                pattern_groups[key].append(sid)

    clusters = []
    skipped = 0
    for label, sids in sorted(pattern_groups.items(), key=lambda x: -len(x[1])):
        if len(sids) < min_cluster_size:
            skipped += 1
            continue
        rep_sid, rep_code = _pick_representative(sids, sub_lookup, criterion)
        if not rep_sid:
            continue
        clusters.append({
            'label': label, 'students': sorted(sids), 'count': len(sids),
            'representative_student': rep_sid, 'representative_code': rep_code,
            'pattern_type': pattern_type,  # 'bad', 'mistake', or 'good'
        })
    if skipped:
        print(f'    ({skipped} clusters < {min_cluster_size} students — left for FCC)')
    return clusters


# ============================================================
# FS Generation (per cluster)
# ============================================================

def _call_ai_for_fs(system_prompt: str, user_prompt: str,
                    model_override: str | None) -> dict | None:
    for attempt in range(3):
        resp = call_deepseek(system_prompt, user_prompt, temperature=0.3,
                            model_override=model_override)
        if not resp or not resp.strip():
            continue
        try:
            return extract_json(resp)
        except Exception:
            try:
                return extract_json(_repair_json(resp))
            except Exception:
                continue
    return None


def _validate_and_build_fs(data: dict, criterion: str, task_num: int,
                           fs_type: str, cluster: dict) -> dict | None:
    """Validate AI output and build FS entry.
    Handles both check_function and regex signature types.
    Supports both (bool, str) tuple returns and legacy bool returns.
    """
    check_fn_body = data.get('check_function', '').strip()
    regex = data.get('regex', '').strip()
    evidence = ""  # matched evidence string

    # Prefer check_function over regex
    if check_fn_body:
        # AI outputs the COMPLETE function including def line. Use as-is.
        fn_source = check_fn_body.strip()
        try:
            import ast as _ast
            _ast.parse(fn_source)
        except SyntaxError as e:
            print(f'      FAILED: syntax error: {e}')
            return None
        try:
            local_ns = {}
            exec(fn_source, {'__builtins__': __builtins__}, local_ns)
            check_fn = local_ns.get('check')
            if not callable(check_fn):
                print(f'      FAILED: no callable check()')
                return None
        except Exception as e:
            print(f'      FAILED: runtime: {e}')
            return None
        try:
            matched, evidence = _call_check_fn(check_fn, cluster['representative_code'])
            if not matched:
                print(f'      FAILED: returned False for representative')
                return None
        except Exception as e:
            print(f'      FAILED: exception: {e}')
            return None

        sign = {'check_function': check_fn_body, 'signature_type': 'check_function'}
    elif regex:
        # Regex fallback
        try:
            re.compile(regex, re.IGNORECASE | re.DOTALL)
        except re.error:
            return None
        try:
            m = re.search(regex, cluster['representative_code'], re.IGNORECASE | re.DOTALL)
            if not m:
                return None
            evidence = m.group(0)[:200]
        except re.error:
            return None
        sign = {'regex': regex, 'signature_type': 'regex'}
    else:
        return None

    # Determine marks: use AI-provided value or default
    marks = data.get('marks')
    if marks is None:
        marks = 1 if fs_type == 'positive' else -1

    return {
        'name': data.get('name', f'FS: {cluster["label"][:60]}'),
        'fs_type': fs_type, 'criterion': criterion,
        **sign,
        'regex_flags': 'IGNORECASE',
        'feedback': data.get('feedback', ''),
        'marks': int(marks),
        'evidence': evidence,
        'task': f'Task{task_num}', 'files': [f'task{task_num}.py'],
        'auto_generated': True,
        'source': f'taffies_{fs_type}',
        'source_detail': cluster['label'][:80],
        '_cluster_label': cluster['label'],
        '_cluster_size': cluster['count'],
        '_representative_student': cluster['representative_student'],
    }


def generate_fs_for_clusters(
    clusters: list[dict], criterion: str, rubric_criteria: list[dict],
    fs_type: str, model_override: str | None = None,
    template_context: str = '',
    behavioral_fingerprints: dict | None = None,
    sub_lookup: dict[str, str] | None = None,
) -> list[dict]:
    """Generate one narrow FS per cluster.

    Args:
        behavioral_fingerprints: {student_id: {criterion: {vulnerable, ...}}}
            If provided, behavioral observations are included in the AI prompt.
        sub_lookup: {student_id: full_code} lookup OR {task_num: {student_id: code}}.
            If provided, ALL students' code in the cluster is sent to AI.
            If None, falls back to only the representative student's code.
    """
    task_num = int(re.search(r'(\d)', criterion).group(1))
    system_prompt = POSITIVE_FS_SYSTEM if fs_type == 'positive' else NEGATIVE_FS_SYSTEM
    build_fn = build_positive_fs_prompt if fs_type == 'positive' else build_negative_fs_prompt

    fs_list = []
    for i, cluster in enumerate(clusters):
        label = cluster['label']
        cluster_is_mistake = (fs_type == 'negative' and cluster.get('pattern_type') == 'mistake')

        # Build behavioral context for this cluster
        behavior_ctx = _build_behavior_context(
            criterion, cluster, behavioral_fingerprints
        )

        tag = ''
        if 'BEHAVIOR: VULNERABLE' in label:
            tag = ' [RUNTIME: VULNERABLE]'
        elif 'BEHAVIOR: SAFE' in label:
            tag = ' [RUNTIME: SAFE]'

        total_students = cluster['count']
        print(f'    [{criterion}] {fs_type} cluster {i+1}/{len(clusters)}: '
              f'"{label[:60]}" ({total_students} students, rep={cluster["representative_student"]}){tag}')

        # Collect ALL student codes in this cluster (NOT just the representative)
        all_codes = []
        if sub_lookup:
            # Handle both flat {sid: code} and nested {task_num: {sid: code}}
            lookup = sub_lookup
            if task_num in sub_lookup and isinstance(sub_lookup.get(task_num), dict):
                lookup = sub_lookup[task_num]
            for sid in cluster['students']:
                code = lookup.get(sid, '')
                if code:
                    extracted = _extract_student_functions(code, criterion)
                    if extracted and len(extracted.strip()) > 30:
                        all_codes.append((sid, extracted))

        if not all_codes:
            # Fallback: use only representative code
            all_codes = [(cluster['representative_student'],
                          cluster.get('representative_code', ''))]

        # Warn if prompt might be large
        total_code_chars = sum(len(c) for _, c in all_codes)
        if total_code_chars > 40000:
            print(f'      WARNING: {len(all_codes)} students, {total_code_chars} chars — may approach context limit')

        prompt_kwargs = {
            'template_context': template_context,
            'behavioral_context': behavior_ctx,
        }
        if cluster_is_mistake:
            prompt_kwargs['is_mistake'] = True

        prompt = build_fn(criterion, rubric_criteria, label,
                         all_codes,  # ← ALL students, not just representative
                         **prompt_kwargs)
        data = _call_ai_for_fs(system_prompt, prompt, model_override)
        if not data:
            print(f'      FAILED: API/parse error')
            continue

        fs = _validate_and_build_fs(data, criterion, task_num, fs_type, cluster)
        if fs:
            fs_list.append(fs)
            print(f'      [OK] {fs["name"][:60]}')
        else:
            print(f'      FAILED: validation (compile/match)')

    return fs_list


# ============================================================
# FCC iteration
# ============================================================

TASK_NUM_MAP = {1: 'Task1', 2: 'Task2', 3: 'Task3'}


def fcc_fill_gaps(
    all_fs: list[dict], task_subs: dict[int, list[dict]],
    all_readmes: dict, rubric_criteria: list[dict],
    max_rounds: int = 3, model_override: str | None = None,
) -> list[dict]:
    """FCC: find uncovered students, generate narrow FS for their specific pattern."""
    for fcc_round in range(1, max_rounds + 1):
        print(f'\n  --- FCC Round {fcc_round} ---')

        all_gaps = {}
        for task_num in [1, 2, 3]:
            task_id = TASK_NUM_MAP[task_num]
            subs = task_subs.get(task_num, [])
            if not subs: continue
            task_fs = [f for f in all_fs if f.get('task') == task_id
                      and f.get('_scoring_weight', 1.0) > 0]
            cov = run_coverage_check(all_fs, subs, task_filter=task_id)
            gaps = find_gaps(cov, subs, task_fs, min_gap_size=1)
            for crit, gap_students in gaps.items():
                if gap_students: all_gaps[crit] = gap_students

        if not all_gaps:
            print('  No gaps — converged.')
            break

        total_gaps = sum(len(v) for v in all_gaps.values())
        print(f'  {len(all_gaps)} criteria with gaps, {total_gaps} student-criterion pairs')

        sub_lookup = {}
        for tn in [1, 2, 3]:
            for s in task_subs.get(tn, []):
                sub_lookup[s['student']] = s.get('code', '')

        added = 0
        for criterion, gap_students in sorted(all_gaps.items(), key=lambda x: -len(x[1])):
            task_num = int(re.search(r'(\d)', criterion).group(1))
            task_id = TASK_NUM_MAP[task_num]

            # PRIORITY: if criterion has ANY bad/mistake patterns in ground truth AND
            # has 0 active negative FS, MUST generate negative FS for gap students.
            crit_has_bad = False
            for sid, rdata in all_readmes.items():
                criteria = rdata.get('criteria', {})
                if isinstance(criteria, str):
                    try: criteria = eval(criteria)
                    except Exception: continue
                if isinstance(criteria, dict):
                    pats = criteria.get(criterion, {})
                    if pats.get('bad', []) or pats.get('mistake', []):
                        crit_has_bad = True
                        break

            # Count existing negative FS for this criterion
            existing_neg = sum(1 for f in all_fs
                             if f.get('criterion') == criterion
                             and f.get('fs_type') == 'negative'
                             and f.get('_scoring_weight', 1.0) > 0)

            for gs in gap_students[:3]:
                code = sub_lookup.get(gs['student'], '')
                if not code or len(code.strip()) < 30: continue

                # Extract student-written functions only, send COMPLETE
                extracted = _extract_student_functions(code, criterion)
                if extracted:
                    code = extracted

                rdata = all_readmes.get(gs['student'], {})
                criteria_raw = rdata.get('criteria', '{}')
                if isinstance(criteria_raw, str):
                    try: criteria_raw = eval(criteria_raw)
                    except Exception: criteria_raw = {}
                gt = criteria_raw if isinstance(criteria_raw, dict) else {}
                crit_gt = gt.get(criterion, {})
                bad_list = crit_gt.get('bad', [])
                mistake_list = crit_gt.get('mistake', [])
                good_list = crit_gt.get('good', [])
                all_bad = bad_list + mistake_list

                # FCC priority logic:
                # 1. If criterion has bad patterns AND 0 negative FS → MUST generate negative
                #    Use RUBRIC bad_patterns as label (specific) not README label (may be vague)
                # 2. If this student has bad patterns → negative
                # 3. If this student has good patterns → positive
                if crit_has_bad and existing_neg == 0:
                    # Use rubric bad_patterns — always specific and actionable
                    crit_rubric = _find_criterion(criterion, rubric_criteria)
                    rubric_bad = crit_rubric.get('bad_patterns', [])
                    if rubric_bad:
                        fs_type = 'negative'
                        label = rubric_bad[0]  # e.g., "Using SQLAlchemy or other database libraries"
                    else:
                        fs_type = 'negative'
                        label = all_bad[0] if all_bad else f'Uncovered error in {gs["student"]}'
                    build_fn = build_negative_fs_prompt
                    sys_prompt = NEGATIVE_FS_SYSTEM
                    print(f'    [{criterion}] FORCING negative FS (rubric: {label[:60]})')
                elif bad_list:
                    fs_type = 'negative'
                    label = bad_list[0]
                    build_fn = build_negative_fs_prompt
                    sys_prompt = NEGATIVE_FS_SYSTEM
                elif good_list:
                    fs_type = 'positive'
                    label = good_list[0]
                    build_fn = build_positive_fs_prompt
                    sys_prompt = POSITIVE_FS_SYSTEM
                else:
                    fs_type = 'positive'
                    label = f'Uncovered pattern in {gs["student"]}'
                    build_fn = build_positive_fs_prompt
                    sys_prompt = POSITIVE_FS_SYSTEM

                prompt = build_fn(criterion, rubric_criteria, label,
                                 [(gs['student'], code)])  # single-student list for FCC
                data = _call_ai_for_fs(sys_prompt, prompt, model_override)
                if not data: continue

                # Validate: support both check_function and regex
                check_fn_body = data.get('check_function', '').strip()
                regex = data.get('regex', '').strip()

                if check_fn_body:
                    fn_source = check_fn_body.strip()
                    try:
                        import ast as _ast
                        _ast.parse(fn_source)
                        local_ns = {}
                        exec(fn_source, {'__builtins__': __builtins__}, local_ns)
                        fn = local_ns.get('check')
                        if not fn or not _call_check_fn(fn, code)[0]:
                            continue
                    except Exception:
                        continue

                    sign = {'check_function': check_fn_body, 'signature_type': 'check_function'}
                elif regex:
                    try:
                        compiled = re.compile(regex, re.IGNORECASE | re.DOTALL)
                        if not compiled.search(code): continue
                    except re.error:
                        continue
                    sign = {'regex': regex, 'signature_type': 'regex'}
                else:
                    continue

                fs = {
                    'name': data.get('name', f'FCC: {label[:60]}'),
                    'fs_type': fs_type, 'criterion': criterion,
                    **sign,
                    'regex_flags': 'IGNORECASE',
                    'feedback': data.get('feedback', ''),
                    'task': task_id, 'files': [f'task{task_num}.py'],
                    'auto_generated': True,
                    'source': 'taffies_fcc', 'source_detail': f'fcc_r{fcc_round}',
                }
                all_fs.append(fs)
                added += 1
                existing_neg_now = sum(1 for f in all_fs
                                      if f.get('criterion') == criterion
                                      and f.get('fs_type') == 'negative'
                                      and f.get('_scoring_weight', 1.0) > 0)
                print(f'    [{criterion}] +1 {fs_type} FS for {gs["student"]}: {fs["name"][:60]}')
                # Stop if we've generated enough (2 negative or 1 positive per round)
                if fs_type == 'negative' and existing_neg_now >= 2:
                    break
                if fs_type == 'positive':
                    break

        if added == 0:
            print('  No new FS added — converged.')
            break
        print(f'  FCC Round {fcc_round}: +{added} FS')

    return all_fs


# ============================================================
# Validation
# ============================================================

def validate_taffies_fs(
    all_fs: list[dict], ref_code: str, template_code: str,
    task_subs: dict[int, list[dict]],
) -> list[dict]:
    """Post-generation validation.
    - Negative FS matching reference: DOWNGRADE weight=0.5 (not delete)
    - Positive FS matching template: DOWNGRADE weight=0.5
    - Broken/unrunnable FS: remove
    - FS matching zero students: remove
    """
    print(f'\n{"=" * 60}')
    print('  TAFFIES FS VALIDATION')
    print('=' * 60)

    # Build code lookup per task — use student IDs as-is from task_subs
    task_codes = {}
    task_sids = {}
    for tn in [1, 2, 3]:
        task_codes[tn] = {}
        task_sids[tn] = set()
        for s in task_subs.get(tn, []):
            sid = s['student']
            task_codes[tn][sid] = s.get('code', '')
            task_sids[tn].add(sid)

    removed = set()
    downgraded = 0

    for fs in all_fs:
        fid = fs.get('id', fs.get('name', '?')[:30])
        ftype = fs.get('fs_type', '?')
        task_str = fs.get('task', '')
        tn = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0
        sig_type = fs.get('signature_type', 'regex')

        # Get the check function or regex
        check_fn = None
        regex = None

        if sig_type == 'check_function':
            fn_body = fs.get('check_function', '')
            if not fn_body:
                print(f'  REMOVED {fid}: empty check_function')
                removed.add(id(fs))
                continue
            fn_source = fn_body.strip()
            try:
                import ast as _ast
                _ast.parse(fn_source)
                local_ns = {}
                exec(fn_source, {'__builtins__': __builtins__}, local_ns)
                check_fn = local_ns.get('check')
                if not callable(check_fn):
                    raise ValueError('no callable check()')
            except Exception as e:
                print(f'  REMOVED {fid}: check function error: {e}')
                removed.add(id(fs))
                continue
        else:
            regex = fs.get('regex', '')
            if not regex:
                continue
            try:
                flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
                re.compile(regex, flags)
            except re.error:
                print(f'  REMOVED {fid}: broken regex')
                removed.add(id(fs))
                continue

        # Check against reference code (negative FS)
        if ftype == 'negative' and ref_code:
            matched_ref = False
            if check_fn:
                try: matched_ref, _ = _call_check_fn(check_fn, ref_code)
                except Exception: pass
            elif regex:
                try: matched_ref = bool(re.search(regex, ref_code, flags))
                except Exception: pass

            if matched_ref:
                fs['_warn_ref_match'] = True
                fs['_scoring_weight'] = 0.5
                downgraded += 1
                print(f'  DOWNGRADED {fid}: negative FS matches reference -> weight=0.5')
                # Keep it! Don't remove.

        # Check against template code (positive FS)
        if ftype == 'positive' and template_code:
            matched_tpl = False
            if check_fn:
                try: matched_tpl, _ = _call_check_fn(check_fn, template_code)
                except Exception: pass
            elif regex:
                try: matched_tpl = bool(re.search(regex, template_code, flags))
                except Exception: pass

            if matched_tpl:
                fs['_warn_matches_template'] = True
                fs['_scoring_weight'] = 0.5
                downgraded += 1
                print(f'  DOWNGRADED {fid}: positive FS matches template -> weight=0.5')
                # Keep it!

        # Must match at least one student
        codes = task_codes.get(tn, {})
        matched_any = False
        for sid, code in codes.items():
            try:
                if check_fn:
                    matched_ok, _ = _call_check_fn(check_fn, code)
                    if matched_ok:
                        matched_any = True
                        break
                elif regex:
                    if re.search(regex, code, flags):
                        matched_any = True
                        break
            except Exception:
                pass

        if not matched_any:
            # For FCC-generated FS, try a broader match across all tasks
            for t2 in [1, 2, 3]:
                if t2 == tn:
                    continue
                for sid, code in task_codes.get(t2, {}).items():
                    try:
                        if check_fn:
                            matched_ok, _ = _call_check_fn(check_fn, code)
                            if matched_ok:
                                matched_any = True
                                fs['task'] = f'Task{t2}'
                                fs['files'] = [f'task{t2}.py']
                                break
                        elif regex:
                            if re.search(regex, code, flags):
                                matched_any = True
                                fs['task'] = f'Task{t2}'
                                fs['files'] = [f'task{t2}.py']
                                break
                    except Exception:
                        pass
                if matched_any:
                    break

        if not matched_any:
            print(f'  REMOVED {fid}: matches zero students')
            removed.add(id(fs))
            continue

    before = len(all_fs)
    all_fs = [f for f in all_fs if id(f) not in removed]
    print(f'\n  {before} -> {len(all_fs)} FS (removed {before - len(all_fs)}, downgraded {downgraded})')
    return all_fs


# ============================================================
# Main orchestrator
# ============================================================

def generate_taffies_fs(
    tasks: list[dict], task_subs: dict[int, list[dict]],
    all_readmes: dict, ref_code: str = '', template_code: str = '',
    model_override: str | None = None,
    min_cluster_size: int = 2,
    behavioral_fingerprints: dict | None = None,
    behavioral_clusters: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Generate TAFFIES-aligned FS (both positive and negative).

    All FS follow the TAFFIES definition:
      - Signature = pattern detected in ACTUAL student code
      - Feedback = specific to that exact pattern
      - Narrow scope — one FS per specific pattern variant, not per category
      - No Type A (broad "missing X") patterns

    Args:
        min_cluster_size: Minimum students per cluster for direct generation.
        behavioral_fingerprints: {student_id: {criterion: {vulnerable, ...}}}
            Runtime behavioral test results. When available, used to:
            1. Cluster students by observed behavior (priority over README labels)
            2. Enrich FS generation prompts with ground-truth behavior
        behavioral_clusters: {criterion: [{label, students, count, ...}]}
            Pre-computed behavioral clusters from runtime/batch_runner.py.
            Takes priority over README-based clustering when available.
    """
    print(f'\n{"=" * 60}')
    print('  TAFFIES-ALIGNED FS GENERATION')
    print(f'  (narrow patterns from actual student code)')
    print(f'  Model: {model_override or DEEPSEEK_MODEL}')
    print('=' * 60)

    all_rubric_criteria = []
    for task in tasks:
        all_rubric_criteria.extend(task.get('rubric_criteria', []))

    # Find which criteria have good/bad patterns
    crit_good = _find_criteria_with_patterns(all_readmes, 'good')
    crit_bad = _find_criteria_with_patterns(all_readmes, 'bad')

    all_criteria = sorted(set(list(crit_good.keys()) + list(crit_bad.keys())))
    print(f'\n  Criteria with patterns: {len(all_criteria)}')
    for crit in all_criteria:
        g = crit_good.get(crit, 0)
        b = crit_bad.get(crit, 0)
        print(f'    {crit}: {g} good, {b} bad')

    # Generate template context dynamically from template code
    tmpl_ctx = _get_template_context(template_code)

    # ── Enrich behavioral clusters with code samples ──
    # Build per-task student→code lookup (avoid overwrite across tasks)
    task_code_lookup: dict[int, dict[str, str]] = {}
    for tn in [1, 2, 3]:
        task_code_lookup[tn] = {}
        for s in task_subs.get(tn, []):
            task_code_lookup[tn][s['student']] = s.get('code', '')

    # Convert behavioral clusters to FS-generator format
    enriched_behavioral: dict[str, list[dict]] = {}
    behavior_criteria: set[str] = set()
    if behavioral_clusters:
        for criterion, clist in behavioral_clusters.items():
            # Determine which task this criterion belongs to
            task_num = int(re.search(r'(\d)', criterion).group(1))

            enriched = []
            for c in clist:
                rep_sid = c.get('representative_student', '')
                # Get actual code from the CORRECT task file
                code = task_code_lookup.get(task_num, {}).get(rep_sid, '')
                # Extract criterion-relevant functions
                extracted = _extract_student_functions(code, criterion) if code else ''
                if not extracted or len(extracted.strip()) < 20:
                    extracted = code  # fallback to full code

                # Determine pattern type from behavior
                label = c.get('label', '')
                if 'VULNERABLE' in label:
                    ptype = 'bad'
                elif 'SAFE' in label:
                    ptype = 'good'
                else:
                    ptype = 'bad'  # default for UNKNOWN/ERROR

                enriched.append({
                    **c,
                    'representative_code': extracted,
                    'pattern_type': ptype,
                })
            enriched_behavioral[criterion] = enriched
            behavior_criteria.add(criterion)

        print(f'\n  Behavioral data available for {len(behavior_criteria)} criteria: '
              f'{sorted(behavior_criteria)}')
        for crit in sorted(behavior_criteria):
            clist = enriched_behavioral[crit]
            for c in clist:
                print(f'    {crit}: [{c["pattern_type"]}] {c["label"][:70]} '
                      f'({c["count"]} students)')

    all_fs: list[dict] = []

    # ── Generate Positive FS ──
    print(f'\n  {"=" * 40}')
    print(f'  POSITIVE FS (correct patterns)')
    print(f'  {"=" * 40}')

    for criterion in sorted(all_criteria):
        # PRIORITY: Use behavioral clusters for criteria with runtime data
        if criterion in behavior_criteria:
            behavior_clist = enriched_behavioral.get(criterion, [])
            safe_clusters = [c for c in behavior_clist if c.get('pattern_type') == 'good']
            if safe_clusters:
                print(f'\n  [{criterion}] {len(safe_clusters)} SAFE behavioral clusters:')
                for c in safe_clusters:
                    print(f'    [+] "{c["label"][:70]}" -- {c["count"]} students')

                pos_fs = generate_fs_for_clusters(
                    safe_clusters, criterion, all_rubric_criteria,
                    fs_type='positive', model_override=model_override,
                    template_context=tmpl_ctx,
                    behavioral_fingerprints=behavioral_fingerprints,
                    sub_lookup=task_code_lookup,
                )
                all_fs.extend(pos_fs)
                continue  # Don't also use README clusters for this criterion

        # Fallback: README-based clustering
        good_clusters = cluster_by_pattern(all_readmes, task_subs, criterion, 'good', min_cluster_size)
        if not good_clusters:
            print(f'\n  [{criterion}] No good pattern clusters -- skipping')
            continue

        print(f'\n  [{criterion}] {len(good_clusters)} good clusters:')
        for c in good_clusters:
            print(f'    [+] "{c["label"][:70]}" -- {c["count"]} students')

        pos_fs = generate_fs_for_clusters(
            good_clusters, criterion, all_rubric_criteria,
            fs_type='positive', model_override=model_override,
            template_context=tmpl_ctx,
            sub_lookup=task_code_lookup,
        )
        all_fs.extend(pos_fs)

    # ── Generate Negative FS (Error + Mistake combined, with label distinction) ──
    # PRIORITY: For criteria with behavioral data, use VULNERABLE behavioral clusters.
    # For criteria without behavioral data, fall back to README-based clustering.
    print(f'\n  {"=" * 40}')
    print(f'  NEGATIVE FS (Error + Mistake patterns)')
    print(f'  {"=" * 40}')

    for criterion in sorted(all_criteria):
        # PRIORITY: Use behavioral clusters for criteria with runtime data
        if criterion in behavior_criteria:
            behavior_clist = enriched_behavioral.get(criterion, [])
            vuln_clusters = [c for c in behavior_clist if c.get('pattern_type') == 'bad']
            if vuln_clusters:
                print(f'\n  [{criterion}] {len(vuln_clusters)} VULNERABLE behavioral clusters:')
                for c in vuln_clusters:
                    print(f'    [-] "{c["label"][:70]}" -- {c["count"]} students')

                neg_fs = generate_fs_for_clusters(
                    vuln_clusters, criterion, all_rubric_criteria,
                    fs_type='negative', model_override=model_override,
                    template_context=tmpl_ctx,
                    behavioral_fingerprints=behavioral_fingerprints,
                    sub_lookup=task_code_lookup,
                )
                all_fs.extend(neg_fs)
                continue  # Don't also use README clusters for this criterion

        # Fallback: README-based clustering
        bad_clusters = cluster_by_pattern(all_readmes, task_subs, criterion, 'bad', min_cluster_size)
        mistake_clusters = cluster_by_pattern(all_readmes, task_subs, criterion, 'mistake', min_cluster_size)
        all_neg_clusters = bad_clusters + mistake_clusters
        all_neg_clusters.sort(key=lambda c: -c['count'])

        if not all_neg_clusters:
            print(f'\n  [{criterion}] No negative clusters -- skipping')
            continue

        print(f'\n  [{criterion}] {len(bad_clusters)} Error + {len(mistake_clusters)} Mistake clusters:')
        for c in all_neg_clusters:
            tag = '[M]' if c.get('pattern_type') == 'mistake' else '[E]'
            print(f'    {tag} "{c["label"][:70]}" -- {c["count"]} students')

        neg_fs = generate_fs_for_clusters(
            all_neg_clusters, criterion, all_rubric_criteria,
            fs_type='negative', model_override=model_override,
            template_context=tmpl_ctx,
            sub_lookup=task_code_lookup,
        )
        all_fs.extend(neg_fs)

    pos_count = sum(1 for f in all_fs if f.get('fs_type') == 'positive')
    neg_count = sum(1 for f in all_fs if f.get('fs_type') == 'negative')
    print(f'\n  Clustered: {pos_count} positive + {neg_count} negative = {len(all_fs)} FS')

    # ── FCC Iteration ──
    print(f'\n  {"=" * 40}')
    print(f'  FCC ITERATION')
    print(f'  {"=" * 40}')
    all_fs = fcc_fill_gaps(
        all_fs, task_subs, all_readmes, all_rubric_criteria,
        max_rounds=3, model_override=model_override,
    )

    # ── Validate ──
    all_fs = validate_taffies_fs(all_fs, ref_code, template_code, task_subs)

    # ── Summary ──
    pos = sum(1 for f in all_fs if f.get('fs_type') == 'positive')
    neg = sum(1 for f in all_fs if f.get('fs_type') == 'negative')
    by_crit = defaultdict(lambda: {'positive': 0, 'negative': 0})
    for f in all_fs:
        by_crit[f.get('criterion', '?')][f.get('fs_type', '?')] += 1

    print(f'\n{"=" * 60}')
    print(f'  TAFFIES FS GENERATION COMPLETE')
    print(f'  Total: {len(all_fs)} FS ({pos} positive, {neg} negative)')
    for crit in sorted(by_crit.keys()):
        c = by_crit[crit]
        print(f'    {crit}: {c["positive"]}+ {c["negative"]}-')
    print('=' * 60)

    return all_fs


def _find_criteria_with_patterns(all_readmes: dict, ptype: str) -> dict[str, int]:
    crit_counts: dict[str, set] = defaultdict(set)
    for sid, rdata in all_readmes.items():
        criteria = rdata.get('criteria', {})
        if isinstance(criteria, str):
            try: criteria = eval(criteria)
            except Exception: continue
        if not isinstance(criteria, dict): continue
        for crit, pats in criteria.items():
            if pats.get(ptype, []):
                crit_counts[crit].add(sid)
    return {c: len(s) for c, s in crit_counts.items()}


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys

    question_dir = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(BASE_DIR, 'question')
    submission_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(BASE_DIR, 'submission')
    question_id = sys.argv[3] if len(sys.argv) >= 4 else 'q1_iMusic'

    rubric_cache = os.path.join(BASE_DIR, 'output', question_id, 'rubric_cache.json')
    if not os.path.exists(rubric_cache):
        print('ERROR: rubric_cache.json not found. Run plan_d.py Phase 0 first.')
        sys.exit(1)
    with open(rubric_cache, 'r', encoding='utf-8') as f:
        question_config = json.load(f)

    tasks = question_config.get('tasks', [])

    task_subs = {}
    for tn in [1, 2, 3]:
        task_subs[tn] = collect_submissions_by_task(submission_dir, tn, max_students=None)
        print(f'  Task{tn}: {len(task_subs[tn])} submissions')

    all_readmes = load_all_readmes(submission_dir)
    print(f'  READMEs: {len(all_readmes)} students')

    ref_dir = os.path.join(BASE_DIR, 'references', question_id)
    ref_code = ''
    if os.path.isdir(ref_dir):
        for root, _, files in os.walk(ref_dir):
            for fn in files:
                if fn.endswith('.py'):
                    ref_code += read_file(os.path.join(root, fn)) + '\n'

    template_code = ''
    for dp in [os.path.join(question_dir, 'code'), question_dir]:
        if os.path.isdir(dp):
            for fn in os.listdir(dp):
                if fn.endswith('.py'):
                    template_code += read_file(os.path.join(dp, fn)) + '\n'

    use_reasoner = '--reasoner' in sys.argv
    model = 'deepseek-reasoner' if use_reasoner else None

    all_fs = generate_taffies_fs(tasks, task_subs, all_readmes, ref_code, template_code,
                                  model_override=model)

    out_dir = os.path.join(BASE_DIR, 'output', question_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fs_registry_taffies.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'question': question_config.get('question_name', question_id),
            'model': model or DEEPSEEK_MODEL,
            'pipeline': 'TAFFIES-aligned (narrow patterns from actual code)',
            'total_fs': len(all_fs),
            'fs_registry': all_fs,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n  Output: {out_path}')
