"""
Coverage Matrix & Gap Detection -- Phase 2.5 (local, no AI)
=============================================================
Runs all FS regexes against all submission files to build a
coverage matrix, then identifies gaps where students are
not covered by any FS for a given criterion.

Fully generic -- no per-question hardcoding.
"""

import os
import re
from collections import defaultdict


# ── Function-level code extraction ──
# Regex-based (not AST, because student code may have syntax errors).
# Extracts individual function bodies so FS regexes are scoped to the
# right function, preventing cross-function false matches.

def _extract_functions(code: str) -> dict[str, str]:
    """Extract each top-level function body from Python source.
    Returns {function_name: function_full_text}.
    Handles decorators, type annotations, and nested functions.
    """
    functions = {}
    lines = code.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # Collect decorators
        decorators = []
        while line.strip().startswith('@'):
            decorators.append(line.strip())
            i += 1
            if i >= len(lines):
                break
            line = lines[i].rstrip()

        # Match function definition
        m = re.match(r'(\s*)def\s+(\w+)\s*\(', line)
        if m:
            base_indent = len(m.group(1))
            func_name = m.group(2)
            start = i
            i += 1
            # Collect function body: lines with indentation > base_indent
            while i < len(lines):
                body_line = lines[i]
                if body_line.strip() == '':
                    i += 1
                    continue
                current_indent = len(body_line) - len(body_line.lstrip())
                if current_indent <= base_indent and body_line.strip():
                    break
                i += 1
            end = i
            func_text = '\n'.join(decorators + lines[start:end])
            functions[func_name] = func_text
        else:
            i += 1
    return functions


def _get_criterion_functions(fs_list: list[dict]) -> dict[str, set[str]]:
    """Build mapping: criterion -> set of function names to search.
    Derived from FS names and patterns — automatically finds function
    references in each criterion's FS regex patterns.
    """
    crit_funcs = defaultdict(set)
    for fs in fs_list:
        crit = fs.get('criterion', '')
        rx = fs.get('regex', '')
        if not crit or not rx:
            continue
        # Extract function names from regex: def\s+(\w+) or \b(function_name)\b
        for m in re.finditer(r'def\s+(\w+)', rx):
            crit_funcs[crit].add(m.group(1))
    return dict(crit_funcs)


def run_coverage_check(
    fs_list: list[dict],
    submissions: list[dict],
    task_filter: str | None = None,
) -> dict:
    """
    Run FS regexes against submissions. Build coverage matrix.

    Args:
        fs_list: List of FS dicts, each with 'id', 'criterion', 'regex', 'regex_flags', 'task'.
        submissions: List of {student, code} dicts.
        task_filter: If set, only run FS whose 'task' field matches this value.
                     Use this when submissions only contain code for a specific task.

    Returns:
        {
            'matrix': {student_id: {criterion: hit_count}},
            'fs_hits': {fs_id: [student_id, ...]},
            'unmatched': [fs_id, ...],
            'per_criterion': {criterion: {covered: N, total: N, coverage_pct: N}}
        }
    """
    matrix = defaultdict(lambda: defaultdict(int))
    fs_hits = defaultdict(list)
    fs_stats = defaultdict(lambda: {'hits': 0, 'total': len(submissions)})

    # Filter FS by task if requested
    active_fs = [fs for fs in fs_list
                 if not task_filter or fs.get('task') == task_filter]

    # Build criterion → function mapping once
    crit_funcs = _get_criterion_functions(active_fs)

    for sub in submissions:
        sid = sub['student']
        code = sub.get('code', '')
        if not code.strip():
            continue

        # Extract functions from this student's code (cached per student)
        functions = _extract_functions(code)

        for fs in active_fs:
            fs_id = fs.get('id', '?')
            regex = fs.get('regex')
            criterion = fs.get('criterion', '?')

            if regex is None:
                continue

            # Determine search scope: target functions for this criterion
            target_func_names = crit_funcs.get(criterion, set())

            flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
            try:
                if target_func_names:
                    # Search only in the relevant function bodies
                    matched = False
                    search_texts = []
                    for fname in target_func_names:
                        if fname in functions:
                            search_texts.append(functions[fname])
                    if not search_texts:
                        # No target function found in code — fall back to full code
                        search_texts = [code]
                    for search_text in search_texts:
                        if re.search(regex, search_text, flags):
                            matched = True
                            break
                    if matched:
                        matrix[sid][criterion] += 1
                        fs_hits[fs_id].append(sid)
                        fs_stats[fs_id]['hits'] += 1
                else:
                    # No function mapping — search full code (backward compatible)
                    if re.search(regex, code, flags):
                        matrix[sid][criterion] += 1
                        fs_hits[fs_id].append(sid)
                        fs_stats[fs_id]['hits'] += 1
            except re.error:
                pass

    # ── Rule Engine Supplement ──
    # Apply deterministic rules (stub detection, keyword checks) to fill gaps
    try:
        from rule_engine import is_pass_stub, check_good_pattern
        rule_criteria = set(fs.get('criterion', '?') for fs in active_fs)
        for sub in submissions:
            sid = sub['student']
            code = sub.get('code', '')
            if not code.strip():
                continue
            funcs = functions.get(sid) or _extract_functions(code)

            for criterion in rule_criteria:
                if matrix[sid].get(criterion, 0) > 0:
                    continue
                target_funcs = crit_funcs.get(criterion, set())
                if not target_funcs:
                    continue
                relevant_bodies = [funcs[fn] for fn in target_funcs if fn in funcs]
                if not relevant_bodies:
                    continue
                combined = '\n'.join(relevant_bodies)

                if is_pass_stub(combined):
                    matrix[sid][criterion] += 1
                    fs_hits.setdefault('RULE_STUB_' + criterion, []).append(sid)
                    continue

                has_code = any(
                    check_good_pattern(gp, combined)
                    for gp in ['parameterized', 'select_query', 'insert_query',
                              'update_query', 'delete_query', 'flash_message',
                              'redirect', 'csv_reader']
                )
                if has_code:
                    matrix[sid][criterion] += 1
                    fs_hits.setdefault('RULE_KEYWORD_' + criterion, []).append(sid)
    except ImportError:
        pass

    # Per-criterion summary
    all_criteria = set(fs.get('criterion', '?') for fs in active_fs)
    criteria_summary = {}
    total_students = len(submissions)

    for criterion in sorted(all_criteria):
        covered = sum(
            1 for sid in matrix
            if matrix[sid].get(criterion, 0) > 0
        )
        criteria_summary[criterion] = {
            'covered': covered,
            'total': total_students,
            'coverage_pct': round(100 * covered / total_students, 1) if total_students else 0,
        }

    unmatched = [fid for fid, stats in fs_stats.items() if stats['hits'] == 0]

    return {
        'matrix': {sid: dict(c) for sid, c in matrix.items()},
        'fs_hits': dict(fs_hits),
        'unmatched': unmatched,
        'per_criterion': criteria_summary,
    }


def find_gaps(
    coverage: dict,
    submissions: list[dict],
    fs_list: list[dict],
    min_gap_size: int = 1,
) -> dict[str, list[dict]]:
    """
    Find gaps: (criterion × student) where no FS matched.

    Args:
        coverage: Output of run_coverage_check().
        submissions: List of {student, code} dicts.
        fs_list: List of FS dicts (used to know which criteria exist).
        min_gap_size: Minimum number of students with the same gap
                      to qualify as "worth filling". Default 2.

    Returns:
        {criterion: [{student, code}, ...]}  -- only criteria with >= min_gap_size students.
    """
    matrix = coverage['matrix']
    all_criteria = set(fs.get('criterion', '?') for fs in fs_list)

    gaps = defaultdict(list)

    for sub in submissions:
        sid = sub['student']
        code = sub.get('code', '')
        if not code.strip():
            continue

        for criterion in all_criteria:
            if matrix.get(sid, {}).get(criterion, 0) == 0:
                gaps[criterion].append({'student': sid, 'code': code})

    # Filter by minimum gap size
    significant = {}
    for criterion, gap_students in gaps.items():
        if len(gap_students) >= min_gap_size:
            significant[criterion] = gap_students

    return significant


def build_supplement_prompt(
    criterion: str,
    gap_students: list[dict],
    existing_fs: list[dict],
    task_id: str,
    target_file: str,
) -> str:
    """
    Build prompt for AI to supplement FS for uncovered students.

    Args:
        criterion: Which rubric criterion has gaps (e.g., '1B').
        gap_students: List of {student, code} for uncovered students.
        existing_fs: All existing FS (so AI avoids duplicates).
        task_id: Task identifier.
        target_file: Target filename.
    """
    existing_text = '\n'.join(
        f"- {fs.get('id', '?')}: name={fs.get('name', '?')}, regex={fs.get('regex', 'null')}"
        for fs in existing_fs
        if fs.get('criterion') == criterion or fs.get('criterion') == '?'
    )

    gap_text = '\n\n'.join(
        f"### Student: {g['student']}\n```python\n{g['code'][:2000]}\n```"
        for g in gap_students
    )

    return f"""## Context
Task: {task_id}
Criterion: {criterion}
Target file: {target_file}

## Existing FS for this criterion (DO NOT duplicate)
{existing_text if existing_text else '(none)'}

## Uncovered Students ({len(gap_students)})
Each of these students has ZERO FS matching criterion {criterion}:
{gap_text}

## Your Task
For EACH uncovered student, generate ONE FS.

## CRITICAL: Pattern generalisation rules
Your regex MUST generalise beyond the exact student code so it matches
other students who wrote conceptually equivalent code with minor variations:

  - Table/column names   -> \\w+  (covers Playlist vs playlists, Genre vs genres)
  - Variable names       -> \\w+
  - Function names       -> \\w+
  - String literals      -> ['\"][^'\"]*['\"]
  - Numbers              -> \\d+
  - Whitespace           -> \\s+

Example of OVERLY NARROW (will fail to match variations):
  Bad:  UPDATE\\s+Playlist\\s+SET   (only matches exact table name "Playlist")
  Good: UPDATE\\s+\\w+\\s+SET\\s+\\w+\\s*=\\s*\\?\\s+WHERE\\s+\\w+\\s*=\\s*\\?

## If the student DID NOT implement the required code:
Look carefully: if the student's function body is just `pass`, a TODO comment,
returns hardcoded/dummy data, or is entirely missing the required operation
(e.g. no SQL ORDER BY clause anywhere), then generate a NEGATIVE FS that
matches the ABSENCE of the pattern:

  For missing ORDER BY:  a regex that matches a SELECT query WITHOUT ORDER BY
  For missing UPDATE:     a regex that matches the entire function body with no UPDATE
  For dummy data return:  a regex that matches hardcoded return values

## Method for EACH student:
1. Find WHERE in this student's code the criterion should be implemented
2. If the code is absent/incomplete -> generate NEGATIVE FS for the missing pattern
3. If the code is present -> extract it and GENERALISE per the rules above
4. Test mentally: would this regex match at least 2 other students who wrote
   the same concept with different table names/variable names/formatting?
5. Classify as positive or negative based on whether the actual code meets the criterion
6. Write 2-3 sentences of specific, constructive feedback

Output ONLY JSON:
{{"supplement_fs": [
  {{"name": "...", "fs_type": "positive|negative", "criterion": "{criterion}", "regex": "...", "regex_flags": "IGNORECASE", "feedback": "..."}}
]}}"""


def build_multi_criterion_supplement_prompt(
    gaps_by_criterion: dict[str, list[dict]],
    existing_fs: list[dict],
    task_id: str,
    target_file: str,
    max_students_per_criterion: int = 5,
) -> str:
    """Build ONE prompt for AI to generate FS for MULTIPLE criteria at once.

    This is the main FCC optimisation: instead of one API call per criterion
    (easily 15-20 calls per round), all gaps for a task are batched into
    a single call.  Reduces API calls by ~80%.

    Args:
        gaps_by_criterion: {criterion: [{student, code}, ...]} for all gap criteria.
        existing_fs: All existing FS (to avoid duplicates).
        task_id: Task identifier.
        target_file: Target filename.
        max_students_per_criterion: Cap on student samples per criterion (keep prompts small).
    """
    task_fs = [fs for fs in existing_fs if fs.get('task') == task_id]
    existing_text = '\n'.join(
        f"- {fs.get('id','?')}: c={fs.get('criterion','?')}, regex={fs.get('regex','null')[:60]}"
        for fs in task_fs[:20]
    )

    sections = []
    for criterion, gap_students in sorted(gaps_by_criterion.items()):
        sample = gap_students[:max_students_per_criterion]
        student_text = '\n\n'.join(
            f"### {g['student']}\n```python\n{g['code'][:1500]}\n```"
            for g in sample
        )
        sections.append(
            f"## Criterion: {criterion}  ({len(gap_students)} students uncovered)\n"
            f"{student_text}"
        )

    all_sections = '\n\n'.join(sections)
    criteria_list = ', '.join(sorted(gaps_by_criterion.keys()))

    return f"""## Task: {task_id}  |  Target file: {target_file}
## Gap criteria: {criteria_list}  ({sum(len(v) for v in gaps_by_criterion.values())} total student-criterion gaps)

## Existing FS for this task (DO NOT duplicate)
{existing_text if existing_text else '(none)'}

## Uncovered Students by Criterion
{all_sections}

## Your Task
For EACH criterion above, generate 1-3 FS that cover the uncovered students.

CRITICAL — Pattern generalisation (avoid regex that matches zero students):
  - Table/column/database names -> \\w+
  - Variable/function names      -> \\w+
  - String literals              -> ['\"][^'\"]*['\"]
  - Numbers                      -> \\d+
  - Whitespace                   -> \\s+
  - Do NOT hardcode specific variable names (stats, conn, genres, etc.)
  - Do NOT depend on exact formatting or argument order.

CRITICAL — Python 3 Type Annotation Compatibility:
  Student code uses type hints like "def f(path: Path) -> bool:".
  Use \\)\\s*(?:->\\s*\\w+\\s*)?\\s*: to optionally match the return type.
  WRONG: \\)\\s*:  (fails on "def f(...) -> bool:")
  RIGHT: \\)\\s*(?:->\\s*\\w+\\s*)?\\s*:

CRITICAL — Negative FS MUST detect ABSENCE, not presence:
  If a student DID NOT implement the required code (pass, TODO, hardcoded return):
    -> generate a NEGATIVE FS, BUT the regex MUST use negative lookahead (?!...)
       or negative lookbehind (?<!...) to verify the required pattern is MISSING.

    WRONG:  regex = "def\\s+statistics\\s*\\(\\s*\\)\\s*:"
            → This matches ALL students who have this function, including correct ones.

    RIGHT:  regex = "def\\s+statistics\\s*\\([^)]*\\)\\s*:\\s*\\n\\s*pass"
            → This only matches when the function body is just 'pass' (a stub).

    RIGHT:  regex = "def\\s+update_playlist_tracks[{{^}}]*\\{{(?!.*INSERT\\s+INTO)"
            → This only matches when INSERT is truly absent from the function body.

  GOLDEN RULE: Before finalising a negative FS, ask:
    "Would the reference (correct) solution match this regex?"
    If YES → rewrite with a negative assertion.

Output ONLY JSON:
{{"supplement_fs": [
  {{"name": "...", "fs_type": "positive|negative", "criterion": "RX_Y",
    "regex": "...", "regex_flags": "IGNORECASE", "feedback": "..."}}
]}}"""


def format_coverage_report(coverage: dict) -> str:
    """Format FCC (Feedback Coverage Check) per TAFFIES §2.3.

    Reports per-criterion coverage: how many students received at least
    one FS match for each rubric criterion.  This is the TAFFIES FCC
    metric — per-student per-criterion, not per-line.
    """
    lines = []
    lines.append('=' * 60)
    lines.append('TAFFIES FCC REPORT (per-criterion student coverage)')
    lines.append('=' * 60)

    pc = coverage.get('per_criterion', {})
    rubric_criteria = {k: v for k, v in pc.items() if k.startswith('RQ')}
    other_criteria = {k: v for k, v in pc.items() if not k.startswith('RQ')}

    if rubric_criteria:
        for criterion, info in sorted(rubric_criteria.items()):
            bar = _coverage_bar(info['coverage_pct'])
            gaps = info['total'] - info['covered']
            lines.append(
                f"  {criterion}: {info['covered']}/{info['total']} students "
                f"({info['coverage_pct']}%) {bar}"
                + (f'  [{gaps} gap(s)]' if gaps > 0 else '')
            )
        avg = round(sum(v['coverage_pct'] for v in rubric_criteria.values()) / len(rubric_criteria), 1)
        lines.append(f'  --- rubric avg: {avg}%')

    if other_criteria:
        lines.append(f'\n  Other criteria:')
        for criterion, info in sorted(other_criteria.items()):
            lines.append(f'    {criterion}: {info["covered"]}/{info["total"]}')

    unmatched = coverage.get('unmatched', [])
    if unmatched:
        lines.append(f'\n  !! Unmatched FS (hit zero students): {len(unmatched)}')

    # Per-criterion gap warnings
    for criterion, info in sorted(rubric_criteria.items()):
        if info['coverage_pct'] < 80:
            missing = info['total'] - info['covered']
            lines.append(f'\n  !! GAP: {criterion} — {missing} student(s) have no FS match')

    return '\n'.join(lines)


def find_low_coverage_students(
    coverage: dict,
    threshold: int = 1,
) -> dict[str, list[dict]]:
    """
    Find students with hit_count <= threshold for any criterion.

    These are "barely covered" students -- their code has only a single
    (or zero) FS match, making them more likely to have undetected issues.

    Args:
        coverage: Output of run_coverage_check().
        threshold: Maximum hit_count to flag (default 1 = single-match students).

    Returns:
        {criterion: [{student, hit_count}, ...]}
        Only includes criteria that have at least one low-coverage student.
    """
    matrix = coverage.get('matrix', {})
    all_criteria: set[str] = set()
    for sid, crits in matrix.items():
        all_criteria.update(crits.keys())

    flagged: dict[str, list[dict]] = defaultdict(list)

    for criterion in sorted(all_criteria):
        for sid, crits in matrix.items():
            hit = crits.get(criterion, 0)
            if 0 < hit <= threshold:
                flagged[criterion].append({
                    'student': sid,
                    'hit_count': hit,
                })

    # Remove empty entries
    return {c: v for c, v in flagged.items() if v}


def _coverage_bar(pct: float, width: int = 20) -> str:
    filled = int(width * pct / 100)
    return f'[{"#" * filled}{"-" * (width - filled)}]'


def _parse_flags(flags_str: str) -> int:
    flags = 0
    if 'IGNORECASE' in flags_str:
        flags |= re.IGNORECASE
    if 'DOTALL' in flags_str:
        flags |= re.DOTALL
    if 'MULTILINE' in flags_str:
        flags |= re.MULTILINE
    return flags


# ============================================================
# Line-level helpers (used by line_coverage.py)
# ============================================================

def find_matching_lines(
    regex: str,
    code: str,
    flags: int = 0,
) -> list[tuple[int, str]]:
    """Find which lines in source code are matched by a regex.

    Uses re.finditer() to locate all match spans, then maps each
    match to the corresponding physical line number(s).

    Args:
        regex: The FS regex pattern.
        code: Full source code string.
        flags: Regex flags (re.IGNORECASE, etc.).

    Returns:
        List of (line_num, matched_text) tuples. line_num is 1-based.
        If a match spans multiple lines, each covered line gets an entry.
    """
    try:
        line_starts = _compute_line_starts(code)
        matches: list[tuple[int, str]] = []
        seen_lines: set[int] = set()
        for m in re.finditer(regex, code, flags):
            start = m.start()
            end = m.end()
            matched_text = code[start:end]
            start_line = _find_line_number(start, line_starts)
            end_line = _find_line_number(max(end - 1, start), line_starts)
            for ln in range(start_line, end_line + 1):
                if ln not in seen_lines:
                    seen_lines.add(ln)
                    matches.append((ln, matched_text))
        return matches
    except re.error:
        return []


def check_regex_on_line(regex: str, line_text: str, flags: int = 0) -> bool:
    """Test if a regex matches a single line of code.

    Unlike running re.search against the whole file, this tests
    against exactly one line.

    Args:
        regex: FS regex pattern.
        line_text: A single line of code.
        flags: Regex flags.

    Returns:
        True if the regex matches this line.
    """
    try:
        return bool(re.search(regex, line_text, flags))
    except re.error:
        return False


def _compute_line_starts(code: str) -> list[int]:
    """Compute character offsets where each physical line begins.

    Returns:
        List of 0-indexed character positions for the start of each line.
    """
    starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            starts.append(i + 1)
    return starts


def _find_line_number(char_offset: int, line_starts: list[int]) -> int:
    """Map a character offset to a 1-based physical line number.

    Args:
        char_offset: 0-indexed character position.
        line_starts: Output of _compute_line_starts().

    Returns:
        1-based line number.
    """
    import bisect
    idx = bisect.bisect_right(line_starts, char_offset) - 1
    return max(idx + 1, 1)
