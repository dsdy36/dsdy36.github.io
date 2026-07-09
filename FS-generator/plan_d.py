"""
Plan D: Iterative Refinement FS Generation Pipeline
====================================================
Replaces the 27-batch "generate-then-filter" approach with:
  Round 1 (Draft):  3 API calls, 20 stratified students → ~200 FS
  Validation:        Run on all 50 students, generate feedback report
  Round 2 (Refine): 3 API calls, feedback + 10 new students → fix + add
  Round 3 (Polish): Per-criterion targeted supplement for stubborn gaps
  FCC Safety Net:   Final coverage check (fallback)

Key improvements over V3:
  - 27 API calls → 7-9 (3× faster wall-clock)
  - 981 initial FS → ~200 (less waste)
  - Phase 0 identifier whitelist → no post-hoc variable generalisation
  - Feedback-driven iteration → AI learns from its mistakes
  - Type B negative FS (pure positive regex for bad patterns) explicitly allowed
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

# Import pipeline utilities
from ai_pipeline import (
    read_file, call_deepseek as _call_deepseek_original, extract_json, _repair_json,
    collect_submissions_by_task, build_fs_prompt, build_bad_pattern_summary,
    apply_quality_gates, fcc_supplement_loop,
    detect_conflicts, check_quality_coverage,
    SYSTEM_PROMPT,
)
from ground_truth import (
    load_all_readmes, build_pattern_inventory,
    get_task_patterns, format_patterns_for_prompt,
    verify_all_readmes, print_verification_report,
)
from coverage import run_coverage_check, format_coverage_report, find_gaps, _parse_flags

# ── Whitelist-aware API call wrapper ──

_whitelist_rule_cache: str = ''

def set_whitelist(whitelist: dict):
    """Cache the whitelist rule for injection into system prompt."""
    global _whitelist_rule_cache
    _whitelist_rule_cache = build_whitelist_rule(whitelist)

def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.3,
                  model_override: str | None = None) -> str | None:
    """Wrapper that injects identifier whitelist into system prompt.
    Passes model_override through to the underlying API call.
    """
    if _whitelist_rule_cache and '{WHITELIST_RULE}' in system_prompt:
        system_prompt = system_prompt.replace('{WHITELIST_RULE}', _whitelist_rule_cache)
    elif _whitelist_rule_cache:
        # System prompt already has whitelist injected (no placeholder) — append
        system_prompt = system_prompt + '\n\n' + _whitelist_rule_cache
    return _call_deepseek_original(system_prompt, user_prompt, temperature,
                                   model_override=model_override)


# ============================================================
# Phase 0 Enhancement: Identifier Whitelist
# ============================================================

PHASE0_WHITELIST_SYSTEM = """You are an educational technology assistant. Your job is to analyze
a programming assignment and extract its structure AND identifier whitelist.
Be precise. Output ONLY valid JSON."""


def build_phase0_whitelist_prompt(question_dir: str) -> str:
    """Build Phase 0 prompt that also requests identifier whitelist."""
    from ai_pipeline import _read_pdf

    pdf_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        for fname in sorted(filenames):
            if fname.endswith('.pdf'):
                pdf_text = _read_pdf(os.path.join(dirpath, fname))
                break

    code_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        for fname in sorted(filenames):
            if fname.endswith('.py'):
                code_text += f'### {fname}\n```python\n{read_file(os.path.join(dirpath, fname))}\n```\n\n'

    data_text = ''
    for dirpath, dirnames, filenames in os.walk(question_dir):
        for fname in sorted(filenames):
            if fname.endswith(('.tsv', '.csv', '.db')):
                rel = os.path.relpath(os.path.join(dirpath, fname), question_dir)
                data_text += f'### {rel} (data file present)\n'

    return f"""Extract the EXACT grading rubric AND an identifier whitelist.

## Complete Assignment Document
{data_text}
{code_text}

## Full Assignment PDF
{pdf_text}

## EXTRACTION RULES

### Part 1: Rubric (same as before)
Extract all rubric criteria with: id, name, description, marks, good_patterns, bad_patterns.

### Part 2: Identifier Whitelist
Extract ALL domain-specific identifiers that appear in the assignment:
- **function_names**: All required function names students must implement
- **api_names**: Flask/SQLite methods and attributes students will use
  (cursor, execute, flash, render_template, redirect, url_for, request, etc.)
- **table_names**: Database table names (Playlist, Track, Genre, PlaylistTrack, etc.)
- **column_names**: Database column names (PlaylistId, TrackId, Name, GenreId, etc.)
- **constants**: Fixed string/numeric constants in the assignment
  (ASC, DESC, All, danger, success, /statistics/, /playlists/, etc.)

The whitelist is used to tell the FS generator: "These identifiers can appear literally
in regex. Everything else MUST use \\\\w+."

## Output format
{{"question_name": "...", "tasks": [...],
  "identifier_whitelist": {{
    "function_names": ["..."],
    "api_names": ["..."],
    "table_names": ["..."],
    "column_names": ["..."],
    "constants": ["..."]
  }}
}}"""


def phase0_with_whitelist(question_dir: str, cache_path: str = '') -> dict | None:
    """Enhanced Phase 0 that extracts rubric AND identifier whitelist.

    Returns dict with 'tasks' and 'identifier_whitelist', or None on failure.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('identifier_whitelist'):
                print(f'  Loaded rubric + whitelist from cache: {cache_path}')
                return cached
        except Exception:
            pass

    prompt = build_phase0_whitelist_prompt(question_dir)
    print(f'  Phase 0 (whitelist) prompt: {len(prompt)} chars')

    for attempt in range(3):
        resp = call_deepseek(PHASE0_WHITELIST_SYSTEM, prompt)
        if not resp:
            continue
        try:
            result = extract_json(resp)
            # Validate whitelist
            wl = result.get('identifier_whitelist', {})
            if wl:
                for key in ['function_names', 'api_names', 'table_names', 'column_names', 'constants']:
                    if key not in wl:
                        wl[key] = []
                print(f'  Whitelist: {sum(len(v) for v in wl.values())} identifiers '
                      f'(funcs={len(wl.get("function_names",[]))}, '
                      f'apis={len(wl.get("api_names",[]))}, '
                      f'tables={len(wl.get("table_names",[]))}, '
                      f'cols={len(wl.get("column_names",[]))}, '
                      f'consts={len(wl.get("constants",[]))})')
            else:
                print('  WARNING: No whitelist in Phase 0 output — using empty whitelist')
                result['identifier_whitelist'] = {
                    'function_names': [], 'api_names': [],
                    'table_names': [], 'column_names': [], 'constants': [],
                }

            # Cache
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            return result
        except Exception as e:
            print(f'  Phase 0 parse error (attempt {attempt + 1}): {e}')

    return None


def build_whitelist_rule(whitelist: dict) -> str:
    """Build the whitelist rule text for the system prompt."""
    if not whitelist:
        return ''

    parts = []
    for category, label in [
        ('function_names', 'Function names'),
        ('api_names', 'API/method names'),
        ('table_names', 'Database table names'),
        ('column_names', 'Database column names'),
        ('constants', 'Fixed constants/strings'),
    ]:
        items = whitelist.get(category, [])
        if items:
            parts.append(f'  {label}: {json.dumps(items[:30])}')

    if not parts:
        return ''

    return f"""
CRITICAL — IDENTIFIER WHITELIST (use these, generalise everything else):
The following identifiers are the ONLY ones that may appear literally in regex.
ALL other identifiers (variable names, parameter names, local function names)
MUST be matched with \\\\w+.

WHITELISTED IDENTIFIERS:
{chr(10).join(parts)}

RULE: When writing regex, check each identifier against this whitelist.
  - IN whitelist → can use literally: (?:Playlist|playlists?)
  - NOT in whitelist → MUST use \\\\w+: \\\\w+\\\\.execute\\\\(
This ensures your regex generalises across different student variable names.
"""


# ============================================================
# Stratified Sampling
# ============================================================

def select_stratified_sample(all_readmes: dict, n: int = 20,
                              task_num: int | None = None) -> list[str]:
    """Select n students stratified by quality tier.

    Returns list of student IDs.
    If task_num is specified, only considers students who have that task's README.
    """
    tiers = defaultdict(list)
    for sid, rdata in all_readmes.items():
        if task_num is not None:
            # Only include students who have this task's patterns
            task_key = f'Task{task_num}'
            criteria = rdata.get('criteria', {})
            if isinstance(criteria, str):
                criteria = eval(criteria) if criteria else {}
            if task_key not in str(criteria):
                continue
        tier = rdata.get('quality_tier', 'medium')
        tiers[tier].append(sid)

    # Sort within each tier for determinism
    for tier in tiers:
        tiers[tier].sort()

    # Allocate proportionally, minimum 1 per tier
    total = sum(len(v) for v in tiers.values())
    if total == 0:
        return []

    selected = []
    remaining = n

    tier_order = ['excellent', 'strong', 'medium', 'adequate', 'borderline', 'weak', 'poor']
    existing_tiers = [t for t in tier_order if t in tiers and tiers[t]]

    # First pass: allocate proportionally, min 1 per tier
    allocations = {}
    for tier in existing_tiers:
        proportion = len(tiers[tier]) / total
        alloc = max(1, min(len(tiers[tier]), round(n * proportion)))
        allocations[tier] = alloc

    # Adjust to match n exactly
    total_alloc = sum(allocations.values())
    while total_alloc < n:
        # Add to largest tier first
        for tier in existing_tiers:
            if allocations[tier] < len(tiers[tier]):
                allocations[tier] += 1
                total_alloc += 1
                if total_alloc >= n:
                    break
    while total_alloc > n:
        for tier in reversed(existing_tiers):
            if allocations[tier] > 1:
                allocations[tier] -= 1
                total_alloc -= 1
                if total_alloc <= n:
                    break

    # Pick students
    for tier in existing_tiers:
        pick_n = allocations.get(tier, 0)
        pool = tiers[tier]
        # Take evenly from the pool
        if pick_n >= len(pool):
            selected.extend(pool)
        else:
            step = len(pool) / pick_n
            for i in range(pick_n):
                idx = min(int(i * step), len(pool) - 1)
                selected.append(pool[idx])

    selected.sort()
    print(f'  Stratified sample {len(selected)}/{total}: '
          f'{dict((t, allocations.get(t, 0)) for t in existing_tiers)}')
    return selected


# ============================================================
# Feedback Report Generator
# ============================================================

def _extract_api_keywords(criterion: str, task_num: int,
                           all_fs_for_criterion: list[dict]) -> list[str]:
    """Extract API-level keywords for finding relevant code lines.

    Combines:
    1. Domain patterns from existing FS regexes (sql/file/flask)
    2. General bad-pattern indicators (assignment-agnostic)
    """
    keywords = set()

    # ── General bad-pattern indicators (assignment-agnostic) ──
    # These catch common student mistakes regardless of assignment domain.
    keywords.update([
        # SQL injection patterns (plain strings, used with re.search)
        '%[sd]', r'%\s*\(',            # % formatting
        r'\.format\s*\(',               # .format()
        r'\+\s*str\s*\(',               # String concatenation with str()
        r'\bf["\x27]',                  # f-string
        # File/IO bad patterns
        r'\.readlines\s*\(',            # manual file reading
        r'\.split\s*\(',                # string split
        # General
        r'print\s*\(',                  # debug prints left in
        r'except\s*:',                  # bare except
    ])

    # ── Domain-specific patterns from existing FS ──
    domain_patterns = {
        'sql': [r'\.execute', r'\.fetch', r'INSERT\s+INTO', r'SELECT\s+FROM',
                r'UPDATE\s+\w+', r'DELETE\s+FROM', r'sqlite3\.', r'\.commit',
                r'\.cursor', r'\.close', r'ORDER\s+BY', r'WHERE\s+\w'],
        'file': [r'open\s*\(', r'csv\.', r'DictReader', r'\.reader\s*\(',
                 r'delimiter', r'\.readlines', r'\.split\s*\(', r'\.write',
                 r'with\s+open'],
        'flask': [r'@app\.route', r'render_template', r'redirect', r'url_for',
                  r'flash\s*\(', r'request\.', r'\.form', r'\.args'],
    }
    all_regex_text = ' '.join(fs.get('regex', '') for fs in all_fs_for_criterion)
    for domain, patterns in domain_patterns.items():
        if any(re.search(p, all_regex_text, re.I) for p in patterns[:3]):
            keywords.update(patterns)

    # Also extract literal API names from existing FS regexes
    for fs in all_fs_for_criterion:
        regex = fs.get('regex', '')
        for m in re.finditer(r'(?:INSERT|SELECT|UPDATE|DELETE|CREATE|DROP|'
                             r'csv\.\w+|sqlite3\.\w+|open|flash|render_template|'
                             r'redirect|url_for|\.execute|\.commit|\.cursor|'
                             r'\.fetch|DictReader|DictWriter)',
                             regex, re.I):
            keywords.add(m.group())

    return sorted(keywords)[:30]


def _extract_code_lines(code: str, keywords: list[str]) -> list[str]:
    """Extract lines from code that contain API-level keywords.

    Strips comments and empty lines. Returns list of (line_number, line_text).
    """
    lines = []
    for i, line in enumerate(code.split('\n'), 1):
        stripped = re.sub(r'#.*$', '', line).strip()
        if not stripped:
            continue
        if any(re.search(kw, stripped, re.I) for kw in keywords):
            lines.append(stripped)
    return lines


def _normalise_code_line(line: str) -> str:
    """Normalize a code line for comparison: replace variable names with placeholders."""
    # Replace string literals
    line = re.sub(r"'[^']*'", "''", line)
    line = re.sub(r'"[^"]*"', '""', line)
    # Replace numbers
    line = re.sub(r'\b\d+\b', '0', line)
    # Replace variable names (identifiers not in common APIs)
    # Keep common Python keywords and known APIs intact
    preserve = {'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'try',
                'except', 'with', 'as', 'import', 'from', 'class', 'pass', 'not',
                'in', 'and', 'or', 'True', 'False', 'None', 'continue', 'break',
                'cursor', 'conn', 'db', 'execute', 'fetchall', 'fetchone',
                'flash', 'redirect', 'render_template', 'url_for', 'request',
                'INSERT', 'INTO', 'VALUES', 'SELECT', 'FROM', 'WHERE', 'UPDATE',
                'DELETE', 'SET', 'ON', 'JOIN', 'AND', 'OR', 'NOT', 'csv',
                'DictReader', 'reader', 'open', 'Path', 'str', 'int', 'float',
                'bool', 'list', 'dict', 'print', 'len', 'range', 'append',
                'format', 'split', 'replace', 'strip', 'join'}
    words = re.findall(r'\b\w+\b', line)
    for w in words:
        if w not in preserve and not w.startswith('__'):
            line = re.sub(r'\b' + re.escape(w) + r'\b', 'VAR', line)
    return line


def extract_uncovered_patterns(
    criterion: str,
    task_num: int,
    uncovered_students: list[str],
    all_subs_by_task: dict[int, list[dict]],
    all_fs_for_criterion: list[dict],
) -> list[dict]:
    """Extract representative code patterns from students not covered by any FS.

    Algorithm:
    1. Get API keywords from existing FS regexes for this criterion
    2. For each gap student, extract lines containing those keywords
    3. Remove lines already matched by ANY existing FS
    4. Normalize and cluster remaining lines
    5. Return top clusters with example code and student counts

    Returns list of {pattern, student_count, example_students, suggestion}.
    """
    # Get student code for uncovered students
    subs = all_subs_by_task.get(task_num, [])
    sub_lookup = {s['student']: s.get('code', '') for s in subs}

    # Get API keywords
    keywords = _extract_api_keywords(criterion, task_num, all_fs_for_criterion)
    if not keywords:
        return []

    # Compile existing FS regexes
    compiled_fs = []
    for fs in all_fs_for_criterion:
        regex = fs.get('regex', '')
        if not regex:
            continue
        try:
            compiled_fs.append(re.compile(regex, _parse_flags(fs.get('regex_flags', 'IGNORECASE'))))
        except re.error:
            pass

    # Extract unmatched lines from each gap student
    all_unmatched: list[tuple[str, str]] = []  # [(student_id, normalized_line)]
    raw_examples: dict[str, list[str]] = defaultdict(list)  # normalized_line -> [raw examples]

    for sid in uncovered_students:
        code = sub_lookup.get(sid, '')
        if not code:
            continue
        api_lines = _extract_code_lines(code, keywords)
        for line in api_lines:
            # Check if ANY existing FS matches this line
            matched = False
            for compiled in compiled_fs:
                if compiled.search(line):
                    matched = True
                    break
            if not matched:
                norm = _normalise_code_line(line)
                all_unmatched.append((sid, norm))
                if len(raw_examples[norm]) < 3:
                    raw_examples[norm].append(line)

    if not all_unmatched:
        return []

    # Cluster by normalized text
    from collections import Counter
    norm_counts = Counter(norm for _, norm in all_unmatched)
    norm_students: dict[str, set[str]] = defaultdict(set)
    for sid, norm in all_unmatched:
        norm_students[norm].add(sid)

    # Sort by frequency, take top clusters
    clusters = []
    for norm, count in norm_counts.most_common(10):
        if count < 2:  # Ignore single-occurrence patterns
            continue
        students = sorted(norm_students[norm])
        examples = raw_examples.get(norm, [])[:3]
        clusters.append({
            'pattern': norm,
            'student_count': count,
            'example_students': students[:8],
            'examples': examples,
        })

    return clusters[:5]  # Top 5 clusters


def generate_feedback_report(all_fs: list[dict],
                              all_subs_by_task: dict[int, list[dict]],
                              all_ref_code: str,
                              template_code: str,
                              readmes: dict | None = None) -> dict:
    """Run all FS on all students and generate structured feedback report.

    Returns dict with sections: needs_fix, missing, keep, stats.
    """
    report = {
        'needs_fix': [],       # FS that need regex fixes
        'missing': [],         # Criteria with uncovered students
        'keep': [],            # FS working correctly
        'stats': {},           # Overall statistics
    }

    # Compile all FS
    compiled_fs = []
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue
        try:
            flags_raw = fs.get('regex_flags', 'IGNORECASE')
            flags = _parse_flags(flags_raw)
            compiled = re.compile(regex, flags)
            compiled_fs.append((fs, compiled))
        except re.error:
            fs['_report_broken_regex'] = True
            report['needs_fix'].append({
                'fs_id': fs.get('id', '?'),
                'fs_type': fs.get('fs_type', '?'),
                'criterion': fs.get('criterion', '?'),
                'problem': 'BROKEN_REGEX',
                'detail': f'Regex fails to compile: {regex[:80]}',
                'suggestion': 'Fix regex syntax error',
            })

    # Count matches per FS across all students
    all_students = []
    for task_num in [1, 2, 3]:
        for s in all_subs_by_task.get(task_num, []):
            if s['student'] not in [x['student'] for x in all_students]:
                all_students.append(s)

    # Per-FS match counting
    for fs, compiled in compiled_fs:
        fid = fs.get('id', '?')
        ftype = fs.get('fs_type', '?')
        criterion = fs.get('criterion', '?')
        regex = fs.get('regex', '')
        task = fs.get('task', '')

        # Get relevant students for this task
        task_num = int(task.replace('Task', '')) if task.startswith('Task') else 0
        relevant_subs = all_subs_by_task.get(task_num, all_students)

        matched = []
        for sub in relevant_subs:
            try:
                if compiled.search(sub['code']):
                    matched.append(sub['student'])
            except re.error:
                pass

        match_count = len(matched)
        total = len(relevant_subs) if relevant_subs else 1
        match_pct = match_count / total * 100 if total > 0 else 0

        fs['_report_matches'] = match_count
        fs['_report_total'] = total
        fs['_report_pct'] = match_pct

        # --- Diagnose problems ---

        # Problem 1: Negative FS matches reference code
        if ftype == 'negative' and all_ref_code:
            try:
                if compiled.search(all_ref_code):
                    report['needs_fix'].append({
                        'fs_id': fid,
                        'fs_type': ftype,
                        'criterion': criterion,
                        'problem': 'MATCHES_REFERENCE',
                        'detail': f'Negative FS matches reference (correct) code',
                        'suggestion': 'Tighten regex or switch to Type A (negative lookahead)',
                    })
                    continue
            except re.error:
                pass

        # Problem 2: Positive FS matches template code
        if ftype == 'positive' and template_code:
            try:
                if compiled.search(template_code):
                    report['needs_fix'].append({
                        'fs_id': fid,
                        'fs_type': ftype,
                        'criterion': criterion,
                        'problem': 'MATCHES_TEMPLATE',
                        'detail': f'Positive FS matches template/starter code',
                        'suggestion': 'Add specific implementation detail not in template',
                    })
                    continue
            except re.error:
                pass

        # Problem 3: Too narrow (positive FS matching <10%)
        if ftype == 'positive' and match_pct < 10 and match_count > 0:
            report['needs_fix'].append({
                'fs_id': fid,
                'fs_type': ftype,
                'criterion': criterion,
                'problem': 'TOO_NARROW',
                'detail': f'Only matches {match_count}/{total} students ({match_pct:.1f}%)',
                'suggestion': f'Broaden regex — consider variant spellings, add (?:alt1|alt2) alternation',
            })
            continue

        # Problem 4: Too broad (negative FS matching >40%)
        if ftype == 'negative' and match_pct > 40:
            report['needs_fix'].append({
                'fs_id': fid,
                'fs_type': ftype,
                'criterion': criterion,
                'problem': 'TOO_BROAD',
                'detail': f'Matches {match_count}/{total} students ({match_pct:.1f}%)',
                'suggestion': 'Narrow regex — scope to specific function, add more specific pattern',
            })
            continue

        # Problem 5: Matches nothing (both types)
        if match_count == 0:
            report['needs_fix'].append({
                'fs_id': fid,
                'fs_type': ftype,
                'criterion': criterion,
                'problem': 'MATCHES_NOTHING',
                'detail': 'Zero matches across all students',
                'suggestion': 'Rewrite regex — it may have incorrect syntax or impossible pattern',
            })
            continue

        # If we get here, FS is working
        report['keep'].append(fid)

    # --- Find missing coverage ---
    # Group FS by criterion
    fs_by_criterion = defaultdict(list)
    for fs in all_fs:
        crit = fs.get('criterion', '?')
        if fs.get('_scoring_weight', 1.0) > 0:
            fs_by_criterion[crit].append(fs)

    # Check each criterion × student pair
    all_criteria = sorted(set(fs.get('criterion', '?') for fs in all_fs if fs.get('criterion', '?') != '?'))
    for criterion in all_criteria:
        uncovered_students = []
        task_num = 0
        for tn in [1, 2, 3]:
            tn_str = f'Task{tn}'
            if tn_str in str(fs_by_criterion.get(criterion, [])[0].get('task', '')) if fs_by_criterion.get(criterion) else False:
                task_num = tn
                break
        # Heuristic: extract task number from criterion ID
        m = re.search(r'(\d)', criterion)
        if m and not task_num:
            task_num = int(m.group(1))

        relevant_subs = all_subs_by_task.get(task_num, all_students)

        for sub in relevant_subs:
            sid = sub['student']
            has_match = False
            for fs in fs_by_criterion.get(criterion, []):
                fs_matches = getattr(fs, '_report_matches_list', None)
                # Check if student matched this FS
                regex = fs.get('regex', '')
                if not regex:
                    continue
                try:
                    flags_raw = fs.get('regex_flags', 'IGNORECASE')
                    flags = _parse_flags(flags_raw)
                    if re.search(regex, sub['code'], flags):
                        has_match = True
                        break
                except re.error:
                    pass
            if not has_match:
                uncovered_students.append(sid)

        if uncovered_students:
            # Extract uncovered patterns from gap students
            criterion_fs = fs_by_criterion.get(criterion, [])
            uncovered_patterns = extract_uncovered_patterns(
                criterion, task_num, uncovered_students,
                all_subs_by_task, criterion_fs,
            )
            report['missing'].append({
                'criterion': criterion,
                'task': f'Task{task_num}' if task_num else '?',
                'count': len(uncovered_students),
                'students': uncovered_students[:10],
                'total_uncovered': len(uncovered_students),
                'uncovered_patterns': uncovered_patterns,
            })

    # ── Pattern-level coverage: find code patterns not matched by ANY FS ──
    # This catches patterns "shadowed" by other FS matching the same student.
    # Runs on ALL students, not just gap students.
    report['shadowed_patterns'] = []
    for criterion in all_criteria:
        task_num = 0
        m = re.search(r'(\d)', criterion)
        if m: task_num = int(m.group(1))
        all_sids = [s['student'] for s in all_subs_by_task.get(task_num, [])]
        if not all_sids:
            continue
        criterion_fs = fs_by_criterion.get(criterion, [])
        shadowed = extract_uncovered_patterns(
            criterion, task_num, all_sids,
            all_subs_by_task, criterion_fs,
        )
        if shadowed:
            report['shadowed_patterns'].append({
                'criterion': criterion,
                'task': f'Task{task_num}',
                'patterns': shadowed,
            })

    # --- Stats ---
    total_fs = len(all_fs)
    total_needs_fix = len(report['needs_fix'])
    total_keep = len(report['keep'])
    total_missing = len(report['missing'])
    report['stats'] = {
        'total_fs': total_fs,
        'needs_fix': total_needs_fix,
        'keep': total_keep,
        'missing_criteria': total_missing,
        'fix_rate': round(total_needs_fix / total_fs * 100, 1) if total_fs > 0 else 0,
    }

    return report


def format_feedback_report(report: dict, all_fs: list[dict]) -> str:
    """Format feedback report as AI-readable text."""
    fs_lookup = {f['id']: f for f in all_fs}
    lines = []

    lines.append(f"## Round 1 FS Performance on All Students")
    lines.append(f"")
    lines.append(f"### Summary")
    s = report.get('stats', {})
    n_fix = len(report.get('needs_fix', []))
    n_keep = len(report.get('keep', []))
    n_missing = len(report.get('missing', []))
    n_total = s.get('total_fs', n_fix + n_keep)
    lines.append(f"- {n_total} FS total: {n_fix} need fixing, {n_keep} working, {n_missing} criteria with gaps")
    lines.append(f"")

    # Needs fix section
    if report['needs_fix']:
        lines.append(f"### NEEDS FIX ({len(report['needs_fix'])} FS)")
        lines.append(f"")
        # Group by problem type
        by_problem = defaultdict(list)
        for item in report['needs_fix']:
            by_problem[item['problem']].append(item)

        for problem, items in sorted(by_problem.items()):
            lines.append(f"#### {problem} ({len(items)} FS)")
            for item in items[:15]:  # Cap at 15 per problem
                fs = fs_lookup.get(item['fs_id'], {})
                name = fs.get('name', '?')
                regex = fs.get('regex', '')[:80]
                lines.append(f"- **{item['fs_id']}** [{item['fs_type']}] {name}")
                lines.append(f"  Problem: {item['detail']}")
                lines.append(f"  Suggestion: {item['suggestion']}")
                lines.append(f"  Current regex: `{regex}`")
            lines.append(f"")

    # Missing coverage
    if report['missing']:
        lines.append(f"### MISSING COVERAGE ({len(report['missing'])} criteria)")
        lines.append(f"")
        for item in sorted(report['missing'], key=lambda x: -x['count']):
            lines.append(f"- **{item['criterion']}** ({item['task']}): "
                        f"{item['count']} students uncovered")
            lines.append(f"  Sample uncovered: {', '.join(item['students'][:5])}")

            # Show extracted uncovered patterns
            patterns = item.get('uncovered_patterns', [])
            if patterns:
                lines.append(f"  **Uncovered code patterns found (NOT matched by any FS):**")
                for pi, p in enumerate(patterns, 1):
                    students_str = ', '.join(p['example_students'][:5])
                    lines.append(f"")
                    lines.append(f"  **Pattern {pi}** ({p['student_count']} students): "
                                f"{students_str}")
                    for ex in p['examples'][:2]:
                        lines.append(f"    ```python")
                        lines.append(f"    {ex[:120]}")
                        lines.append(f"    ```")
                    lines.append(f"    → Write FS matching this specific pattern variant")
            lines.append(f"")
        lines.append(f"")

    # Shadowed patterns (not in "missing" — students have other FS, but these code patterns lack coverage)
    if report.get('shadowed_patterns'):
        lines.append(f"### UNCOVERED CODE PATTERNS (exist in student code but no FS matches them)")
        lines.append(f"These patterns are 'shadowed' — students have other FS matches, so FCC shows 100%,")
        lines.append(f"but these specific code variants are NOT detected by any FS.")
        lines.append(f"")
        for item in report['shadowed_patterns']:
            lines.append(f"**{item['criterion']}** ({item['task']}):")
            for pi, p in enumerate(item['patterns'], 1):
                students_str = ', '.join(p['example_students'][:5])
                lines.append(f"  Pattern {pi} ({p['student_count']} students): {students_str}")
                for ex in p['examples'][:2]:
                    lines.append(f"    ```python")
                    lines.append(f"    {ex[:120]}")
                    lines.append(f"    ```")
                lines.append(f"    → Generate FS for this pattern variant")
            lines.append(f"")
        lines.append(f"")

    # Keep section (ultra-brief to save context)
    if report['keep']:
        lines.append(f"### KEEP ({len(report['keep'])} FS working correctly — no changes needed)")
        lines.append(f"  These FS are already correct. Include them unchanged in your output.")
        lines.append(f"")

    lines.append(f"## Instructions for Round 2")
    lines.append(f"")
    lines.append(f"1. For each FS in NEEDS FIX: MODIFY the regex according to the suggestion.")
    lines.append(f"   Keep the same id, name, criterion, and feedback. Only fix the regex.")
    lines.append(f"2. For each criterion in MISSING: Generate NEW FS targeting the uncovered students.")
    lines.append(f"   Study the uncovered students' code patterns and write regex that matches them.")
    lines.append(f"3. For FS in KEEP: Do NOT modify. Include them unchanged in your output.")
    lines.append(f"4. Output the COMPLETE updated FS list (fixed + new + kept) as JSON.")

    return '\n'.join(lines)


# ============================================================
# Round 1: Draft Generation
# ============================================================

def build_round1_prompt(task_id: str, task_cfg: dict, ref_code: str,
                         submissions: list[dict], template_code: str = '',
                         readme_patterns: str = '',
                         whitelist_rule: str = '',
                         bad_pattern_summary: str = '') -> str:
    """Build Round 1 FS generation prompt — single batch, all students at once."""
    prompt = build_fs_prompt(
        task_id, task_cfg, ref_code, submissions,
        template_code=template_code,
        readme_patterns=readme_patterns,
        previous_fs=None,
        batch_label=f'Round 1 — {len(submissions)} students',
        bad_pattern_summary=bad_pattern_summary,
    )

    # Inject whitelist rule after the template/rubric section
    if whitelist_rule:
        # Find insertion point: after "## Rubric Criteria"
        insert_pos = prompt.find('## Student Submissions')
        if insert_pos > 0:
            prompt = (prompt[:insert_pos] +
                      whitelist_rule + '\n' +
                      prompt[insert_pos:])

    return prompt


def round1_generate(tasks: list[dict], submissions_dir: str, ref_dir: str,
                     template_code: str, all_readmes: dict,
                     identifier_whitelist: dict) -> list[dict]:
    """Round 1: Generate draft FS from 20 stratified students per task.

    Returns combined FS list (~200 FS total across 3 tasks).
    """
    TASK_NUM_MAP = {1: 'Task1', 2: 'Task2', 3: 'Task3'}
    all_fs: list[dict] = []
    fs_id_counter: dict[str, int] = {}
    whitelist_rule = build_whitelist_rule(identifier_whitelist)

    # Load README patterns
    readme_patterns_by_task = {}
    if all_readmes:
        for tn in [1, 2, 3]:
            readme_patterns_by_task[tn] = get_task_patterns(all_readmes, TASK_NUM_MAP[tn])

    print(f'\n{"=" * 60}')
    print('  ROUND 1: Draft Generation (3 API calls)')
    print('=' * 60)

    for task_num in [1, 2, 3]:
        task_id = TASK_NUM_MAP[task_num]
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            continue

        target_file = f'task{task_num}.py'
        all_submissions = collect_submissions_by_task(submissions_dir, task_num, max_students=None)
        if not all_submissions:
            print(f'  [{task_id}]: No submissions')
            continue

        # Stratified sample: 20 students
        sample_sids = select_stratified_sample(all_readmes, n=20, task_num=task_num)
        sample_subs = [s for s in all_submissions if s['student'] in sample_sids]
        if len(sample_subs) < 5:
            # Fallback: use all students
            sample_subs = all_submissions[:20]
            print(f'  [{task_id}]: Stratified sample too small, using first {len(sample_subs)}')

        # Build reference code
        ref_code = _build_ref_code_simple(task, ref_dir)

        # README patterns
        task_patterns = readme_patterns_by_task.get(task_num, {})
        readme_text = ''
        if task_patterns:
            readme_text = format_patterns_for_prompt(task_patterns, task_id)

        # Build bad pattern summary for this task
        bad_summary = build_bad_pattern_summary(all_readmes, task_id) if all_readmes else ''

        # Build prompt
        prompt = build_round1_prompt(
            task_id, task, ref_code, sample_subs,
            template_code=template_code,
            readme_patterns=readme_text,
            whitelist_rule=whitelist_rule,
            bad_pattern_summary=bad_summary,
        )
        print(f'  [{task_id}]: {len(sample_subs)} students, {len(prompt)} chars')

        # Call AI
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
            print(f'  [{task_id}]: FAILED after 3 attempts')
            continue

        batch_fs = result.get('fs_registry', [])
        for fs in batch_fs:
            fs.setdefault('task', task_id)
            fs.setdefault('files', [target_file])
            fs['source'] = 'round1'
            fs['auto_generated'] = True
            fs.pop('marks', None)
            crit = fs.get('criterion', '?')
            m = re.search(r'\d+', str(crit))
            num = m.group() if m else '0'
            fs_id_counter.setdefault(num, 0)
            fs_id_counter[num] += 1
            fs['id'] = f'FS{num}.{fs_id_counter[num]}'

        all_fs.extend(batch_fs)
        pos = sum(1 for f in batch_fs if f.get('fs_type') == 'positive')
        neg = sum(1 for f in batch_fs if f.get('fs_type') == 'negative')
        print(f'  [{task_id}]: {len(batch_fs)} FS ({pos}+, {neg}-)')

    # Dedup across tasks (reuse existing function)
    from ai_pipeline import _deduplicate_fs
    before = len(all_fs)
    all_fs = _deduplicate_fs(all_fs)
    print(f'\n  Round 1 total: {before} → {len(all_fs)} FS (dedup)')
    return all_fs


def _build_ref_code_simple(task: dict, ref_dir: str) -> str:
    """Build reference code string for a task."""
    from ai_pipeline import _build_ref_code
    return _build_ref_code(task, ref_dir)


# ============================================================
# Round 2: Feedback-Driven Refinement
# ============================================================

def build_round2_prompt(task_id: str, task_cfg: dict, ref_code: str,
                         all_fs_for_task: list[dict], feedback_report_text: str,
                         new_submissions: list[dict], template_code: str = '',
                         whitelist_rule: str = '') -> str:
    """Build Round 2 prompt: feedback report + new students → fix + add FS."""
    criteria = json.dumps(task_cfg.get('rubric_criteria', []), indent=2)

    student_text = '\n\n'.join(
        f"### Student: {s['student']}\n```python\n{s['code']}\n```"
        for s in new_submissions
    )

    # Current FS for this task (ultra-brief: id + type + name only, no regex)
    existing_text = '\n'.join(
        f"- {fs.get('id','?')} [{fs.get('fs_type','?')}] c={fs.get('criterion','?')}: "
        f"{fs.get('name','?')[:50]}"
        for fs in all_fs_for_task[:20]  # Cap at 20
    )
    if len(all_fs_for_task) > 20:
        existing_text += f'\n... and {len(all_fs_for_task) - 20} more FS (regex omitted to save context)'

    prompt = f"""## Round 2: Fix and Supplement FS for {task_id}

You are refining FS based on validation feedback from Round 1.

## Rubric Criteria
{criteria}

## Reference Implementations (correct code)
{ref_code if ref_code else '(None)'}
{whitelist_rule}
## Current FS for {task_id}
{existing_text if existing_text else '(none)'}

{feedback_report_text}

## New Students (not seen in Round 1 — {len(new_submissions)} students)
{student_text}

## Instructions

### PART 1: Fix broken FS
For each FS listed in NEEDS FIX:
- Modify the REGEX ONLY according to the suggestion
- Keep the same id, name, criterion, fs_type, and feedback
- If the suggestion says "switch to Type A", add (?!...) to detect absence
- If the suggestion says "broaden regex", add (?:variant1|variant2) alternation
- If the suggestion says "narrow regex", scope to specific function(s)

### PART 2: Add missing FS for uncovered students AND uncovered code patterns
For each criterion in MISSING COVERAGE:
- Study the uncovered students' code
- Generate NEW FS that specifically matches their code patterns

For EACH pattern in "UNCOVERED CODE PATTERNS":
- You MUST generate at least 1 FS matching that specific code pattern
- Study the example code snippets provided
- Write regex to match the pattern variant shown
- These patterns exist in student code but have NO FS matching them — fix this

### PART 3: Keep working FS
Include all FS from KEEP unchanged in your output.

### Output Format
Output the COMPLETE updated FS list as JSON:
{{"fs_registry": [
  // ALL FS: fixed ones, new ones, and kept ones
  {{"id": "FS1.1", "name": "...", "fs_type": "positive", "criterion": "RQ1_1",
    "regex": "...", "regex_flags": "IGNORECASE", "feedback": "..."}},
  ...
]}}"""

    return prompt


def round2_refine(tasks: list[dict], submissions_dir: str, ref_dir: str,
                   template_code: str, all_readmes: dict,
                   all_fs: list[dict], feedback_report: dict,
                   identifier_whitelist: dict) -> list[dict]:
    """Round 2: Fix + supplement FS based on feedback report.

    Returns updated FS list.
    """
    TASK_NUM_MAP = {1: 'Task1', 2: 'Task2', 3: 'Task3'}
    whitelist_rule = build_whitelist_rule(identifier_whitelist)
    fs_lookup = {f['id']: f for f in all_fs}

    # Format feedback report
    feedback_text = format_feedback_report(feedback_report, all_fs)

    # Split feedback by task
    needs_fix_by_task = defaultdict(list)
    for item in feedback_report.get('needs_fix', []):
        fs = fs_lookup.get(item['fs_id'], {})
        task = fs.get('task', 'Task1')
        needs_fix_by_task[task].append(item)

    missing_by_task = defaultdict(list)
    for item in feedback_report.get('missing', []):
        task = item.get('task', 'Task1')
        missing_by_task[task].append(item)

    print(f'\n{"=" * 60}')
    print('  ROUND 2: Feedback-Driven Refinement (3 API calls)')
    print('=' * 60)

    # Get fresh students (not seen in Round 1)
    round1_sids = set()
    for fs in all_fs:
        if '_round1_students' in fs:
            round1_sids.update(fs['_round1_students'])

    updated_fs = []

    for task_num in [1, 2, 3]:
        task_id = TASK_NUM_MAP[task_num]
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            continue

        all_submissions = collect_submissions_by_task(submissions_dir, task_num, max_students=None)
        if not all_submissions:
            continue

        # Get FS for this task
        task_fs = [f for f in all_fs if f.get('task') == task_id]

        # Pick 5 new students (reduced from 10 to save context)
        new_subs = all_submissions[:5]
        # Truncate code to 1000 chars per student
        for s in new_subs:
            if len(s.get('code', '')) > 1000:
                s['code'] = s['code'][:1000] + '\n# ... (truncated)'
        print(f'  [{task_id}]: {len(task_fs)} existing FS, {len(new_subs)} new students (truncated)')

        # Build task-specific feedback (including shadowed patterns)
        task_fix_items = needs_fix_by_task.get(task_id, [])
        task_missing_items = missing_by_task.get(task_id, [])
        task_shadowed = [sp for sp in feedback_report.get('shadowed_patterns', [])
                        if sp.get('task') == task_id]
        task_report = {
            'needs_fix': task_fix_items,
            'missing': task_missing_items,
            'keep': [fid for fid in feedback_report.get('keep', [])
                    if fs_lookup.get(fid, {}).get('task') == task_id],
            'stats': {'total_fs': len(task_fs)},
            'shadowed_patterns': task_shadowed,
        }
        task_feedback = format_feedback_report(task_report, all_fs)

        ref_code = _build_ref_code_simple(task, ref_dir)

        prompt = build_round2_prompt(
            task_id, task, ref_code, task_fs, task_feedback,
            new_subs, template_code=template_code,
            whitelist_rule=whitelist_rule,
        )
        print(f'  [{task_id}]: {len(prompt)} chars')

        result = None
        for attempt in range(3):
            resp = call_deepseek(SYSTEM_PROMPT, prompt, temperature=0.3)
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
            print(f'  [{task_id}]: FAILED — keeping original FS')
            updated_fs.extend(task_fs)
            continue

        round2_fs = result.get('fs_registry', [])
        # Build Round 1 lookup for this task: id → regex
        round1_lookup = {}
        for f in task_fs:
            fid = f.get('id', '')
            if fid:
                round1_lookup[fid] = f.get('regex', '')

        for fs in round2_fs:
            fs.setdefault('task', task_id)
            fs.setdefault('files', [f'task{task_num}.py'])
            fs['source'] = 'round2'
            fs['auto_generated'] = True
            fs.pop('marks', None)

            # Compare with Round 1 to mark detail
            fid = fs.get('id', '')
            if fid not in round1_lookup:
                fs['source_detail'] = 'new'
            elif fs.get('regex', '') != round1_lookup[fid]:
                fs['source_detail'] = 'modified'
                fs['original_regex'] = round1_lookup[fid]
            else:
                fs['source_detail'] = 'kept'

        # Count
        n_new = sum(1 for f in round2_fs if f.get('source_detail') == 'new')
        n_mod = sum(1 for f in round2_fs if f.get('source_detail') == 'modified')
        n_kept = sum(1 for f in round2_fs if f.get('source_detail') == 'kept')
        print(f'  [{task_id}]: {len(round2_fs)} FS ({n_new} new, {n_mod} modified, {n_kept} kept)')
        updated_fs.extend(round2_fs)

    print(f'\n  Round 2 total: {len(updated_fs)} FS')
    return updated_fs


# ============================================================
# Round 3: Targeted Supplement (Stubborn Gaps)
# ============================================================

def round3_targeted(tasks: list[dict], all_subs_by_task: dict[int, list[dict]],
                     all_fs: list[dict], all_ref_code: str,
                     template_code: str,
                     shadowed_patterns: list[dict] | None = None) -> list[dict]:
    """Round 3: Per-criterion targeted FS for stubborn gaps + shadowed patterns.

    Returns updated FS list.
    """
    print(f'\n{"=" * 60}')
    print('  ROUND 3: Targeted Supplement (per-criterion)')
    print('=' * 60)

    # Build shadowed patterns lookup by criterion
    shadowed_by_crit: dict[str, list[dict]] = defaultdict(list)
    if shadowed_patterns:
        for sp in shadowed_patterns:
            shadowed_by_crit[sp.get('criterion', '')].extend(sp.get('patterns', []))

    # Find remaining gaps
    TASK_NUM_MAP = {1: 'Task1', 2: 'Task2', 3: 'Task3'}
    all_gaps = {}

    for task_num in [1, 2, 3]:
        task_id = TASK_NUM_MAP[task_num]
        subs = all_subs_by_task.get(task_num, [])
        if not subs:
            continue

        task_fs = [f for f in all_fs if f.get('task') == task_id
                   and f.get('_scoring_weight', 1.0) > 0]

        cov = run_coverage_check(all_fs, subs, task_filter=task_id)
        gaps = find_gaps(cov, subs, task_fs, min_gap_size=1)

        for crit, gap_students in gaps.items():
            if gap_students:
                all_gaps[crit] = gap_students

    # Also add shadowed-only criteria (no FCC gaps but patterns uncovered)
    for crit, patterns in shadowed_by_crit.items():
        if crit not in all_gaps and patterns:
            # Get relevant task students
            m = re.search(r'(\d)', crit)
            task_num = int(m.group(1)) if m else 1
            subs = all_subs_by_task.get(task_num, [])
            # Find which students have these patterns
            pattern_sids = set()
            for p in patterns:
                pattern_sids.update(p.get('example_students', []))
            gap_list = [s for s in subs if s['student'] in pattern_sids]
            if gap_list:
                all_gaps[crit] = gap_list

    if not all_gaps:
        print('  No gaps or shadowed patterns — Round 3 not needed')
        return all_fs

    total_gaps = sum(len(v) for v in all_gaps.values())
    print(f'  {len(all_gaps)} criteria with gaps/shadowed, {total_gaps} student-criterion pairs')

    # Per-criterion targeted prompts
    fs_id_counter: dict[str, int] = {}
    for fs in all_fs:
        crit = fs.get('criterion', '?')
        try:
            seq = int(fs.get('id', 'FS0.0').split('.')[-1] or 0)
        except (ValueError, IndexError):
            seq = 0
        fs_id_counter[crit] = max(fs_id_counter.get(crit, 0), seq)

    total_added = 0
    for criterion, gap_students in sorted(all_gaps.items(), key=lambda x: -len(x[1])):
        m = re.search(r'\d+', str(criterion))
        task_num = int(m.group()) if m else 1
        task_id = f'Task{task_num}'
        task_fs = [f for f in all_fs if f.get('task') == task_id]

        existing_text = '\n'.join(
            f"- {fs.get('id','?')}: {fs.get('name','')[:60]} | {fs.get('regex','')[:80]}"
            for fs in task_fs if fs.get('criterion') == criterion
        )

        student_codes_str = '\n'.join(
            "### " + g['student'] + "\n```python\n" + g['code'][:1500] + "\n```"
            for g in gap_students[:10]
        )
        prompt = "Generate FS for " + criterion + " (" + task_id + ") — " + str(len(gap_students)) + " students uncovered.\n\n"
        prompt += "## Existing FS for " + criterion + "\n"
        prompt += (existing_text if existing_text else '(none)') + "\n\n"
        prompt += "## ALL Uncovered Students — study EACH one carefully\n"
        prompt += student_codes_str + "\n\n"
        # Include shadowed patterns if available
        crit_shadowed = shadowed_by_crit.get(criterion, [])
        if crit_shadowed:
            prompt += "## Uncovered Code Patterns (exist in student code but NO FS matches them)\n"
            for spi, sp in enumerate(crit_shadowed[:3], 1):
                prompt += f"Pattern {spi} ({sp.get('student_count', '?')} students):\n"
                for ex in sp.get('examples', [])[:2]:
                    prompt += f"  ```python\n  {ex[:120]}\n  ```\n"
                prompt += "  -> You MUST generate at least 1 negative FS for this pattern.\n\n"

        prompt += "## Task\n"
        prompt += "Generate 3-8 FS that SPECIFICALLY match the patterns in these students' code.\n\n"
        prompt += "REQUIRED: Generate FS for these specific bad patterns if present:\n"
        prompt += "- f-string SQL injection: f[\"'].*\\b(?:INSERT|UPDATE|DELETE)\\b\n"
        prompt += "- % formatting SQL: \\.execute\\s*\\(\\s*[\"'].*%[sd].*[\"']\\s*%\\s*\\(\n"
        prompt += "- String concatenation SQL: \\.execute\\s*\\(\\s*[\"'].*[\"']\\s*\\+\\s*str\\(\n"
        prompt += "- .format() SQL: \\.execute\\s*\\(\\s*[\"'].*\\{\\}.*[\"']\\s*\\.\\s*format\\s*\\(\n\n"
        prompt += "For each FS:\n"
        prompt += "- Type A (missing good): use (?!...) negative lookahead\n"
        prompt += "- Type B (present bad): use pure positive regex — IS the correct approach\n\n"
        prompt += "Each regex MUST match at least one uncovered student above.\n"
        prompt += 'Output ONLY JSON: {"supplement_fs": [...]}'

        print(f'    {criterion}: {len(gap_students)} gaps, {len(prompt)} chars')
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

        added = 0
        for fs in result.get('supplement_fs', []):
            regex = fs.get('regex')
            if not regex:
                continue
            try:
                flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
                re.compile(regex, flags)
            except re.error:
                continue

            # Validate: must match at least one gap student
            if not any(re.search(regex, g['code'], flags)
                      for g in gap_students[:15]):
                continue

            # Cross-check: negative FS must not match reference
            if fs.get('fs_type') == 'negative' and all_ref_code:
                try:
                    if re.search(regex, all_ref_code, flags):
                        fs['_warn_ref_match'] = True
                except re.error:
                    pass

            # Cross-check: positive FS must not match template
            if fs.get('fs_type') == 'positive' and template_code:
                try:
                    if re.search(regex, template_code, flags):
                        fs['_warn_matches_template'] = True
                except re.error:
                    pass

            fs.setdefault('task', task_id)
            tf = next((t.get('target_file', f'task{task_num}.py')
                       for t in tasks if t['id'] == task_id), f'task{task_num}.py')
            fs.setdefault('files', [tf])
            fs.setdefault('auto_generated', True)
            fs.setdefault('criterion', criterion)
            fs['source'] = 'round3'
            fs_id_counter.setdefault(criterion, 0)
            fs_id_counter[criterion] += 1
            fs['id'] = f'FS{criterion.replace("RQ", "").replace("_", "")}.{fs_id_counter[criterion]}'
            all_fs.append(fs)
            added += 1
            total_added += 1

        print(f'    +{added} FS for {criterion}')

    print(f'\n  Round 3 total added: {total_added} FS')
    return all_fs


# ============================================================
# Plan D Orchestrator
# ============================================================

# ============================================================
# Post-processing: Broad Negative FS Filter
# ============================================================

def _apply_broad_negative_filter(all_fs: list[dict],
                                   task_subs: dict[int, list[dict]],
                                   threshold: float = 0.40):
    """Remove negative FS that match >threshold of students.

    Assignment-agnostic — only needs FS regexes + student code.
    Ported from V3 Gate 7 (Broad Negative FS Detection).
    """
    print(f'\n{"=" * 60}')
    print('  BROAD NEGATIVE FS FILTER')
    print(f'  (threshold: >{threshold:.0%} match rate = excluded)')
    print('=' * 60)

    # Build per-task student code lookup
    task_codes: dict[int, list[str]] = {}
    for tn in [1, 2, 3]:
        task_codes[tn] = [s.get('code', '') for s in task_subs.get(tn, [])
                          if s.get('code', '').strip()]

    removed = 0
    flagged = 0
    for fs in all_fs:
        if fs.get('fs_type') != 'negative':
            continue
        if fs.get('_scoring_weight', 1.0) == 0.0:
            continue
        regex = fs.get('regex', '')
        if not regex:
            continue

        # Determine task
        task_str = fs.get('task', '')
        tn = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0
        codes = task_codes.get(tn, [])
        total = len(codes)
        if total == 0:
            continue

        # Count matches
        try:
            compiled = re.compile(regex, re.IGNORECASE | re.DOTALL)
            matched = sum(1 for c in codes if compiled.search(c))
        except re.error:
            continue

        match_pct = matched / total

        if match_pct > threshold:
            fs['_warn_broad_negative'] = True
            fs['_broad_match_pct'] = round(match_pct, 2)
            fs['_scoring_weight'] = 0.0
            removed += 1
            print(f'  EXCLUDED {fs.get("id","?")}: {matched}/{total} '
                  f'({match_pct:.0%}) — {fs.get("name","?")[:50]}')
        elif match_pct > 0.25:
            flagged += 1

    print(f'  Excluded: {removed}, Flagged (>25%): {flagged}')
    print(f'  Remaining: {len(all_fs)} FS')


# ============================================================
# Post-processing: Variable Name Generalisation
# ============================================================

# Common Python keywords that should never be replaced
_PY_RESERVED = frozenset({
    'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
    'finally', 'with', 'as', 'import', 'from', 'class', 'pass', 'break',
    'continue', 'yield', 'raise', 'assert', 'del', 'global', 'nonlocal',
    'lambda', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None',
    'self', 'cls', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set',
    'tuple', 'bytes', 'print', 'len', 'range', 'open', 'Path', 'path',
    'append', 'extend', 'insert', 'remove', 'pop', 'get', 'items', 'keys',
    'values', 'update', 'close', 'commit', 'execute', 'fetchall', 'fetchone',
    'csv', 'sqlite3', 'json', 'os', 'sys', 're', 'datetime', 'io',
})

# SQL keywords — keep literal
_SQL_KEYWORDS = frozenset({
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN', 'ON',
    'INTO', 'VALUES', 'SET', 'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX',
    'ORDER', 'BY', 'ASC', 'DESC', 'GROUP', 'HAVING', 'UNION', 'ALL',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'IFNULL', 'NULL',
    'NOT', 'AND', 'OR', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'LIMIT',
    'IGNORE', 'INNER', 'OUTER', 'LEFT', 'RIGHT', 'CROSS', 'NATURAL',
    'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CASCADE', 'DEFAULT',
    'CHECK', 'UNIQUE',
})


def _apply_variable_generalisation(all_fs: list[dict],
                                     task_subs: dict[int, list[dict]],
                                     whitelist: dict):
    """Replace hardcoded variable names in FS regex with \\w+.

    Uses Python tokenizer to identify identifiers in student code,
    excludes whitelisted names, replaces remaining in FS regex.
    Assignment-agnostic — ported from V3 Gate 1.
    """
    print(f'\n{"=" * 60}')
    print('  VARIABLE NAME GENERALISATION')
    print('=' * 60)

    # Build whitelist from Phase 0 + built-in sets
    allowed = set()
    for cat in ['function_names', 'api_names', 'table_names', 'column_names', 'constants']:
        for item in whitelist.get(cat, []):
            allowed.add(str(item))
    allowed.update(_PY_RESERVED)
    allowed.update(_SQL_KEYWORDS)
    # Also allow regex metachar sequences
    allowed.update({'s', 'S', 'w', 'W', 'd', 'D', 'n', 'r', 't', 'b', 'B', 'Z', 'A'})

    # Collect all identifiers from student code
    all_identifiers: set[str] = set()
    for tn in [1, 2, 3]:
        for s in task_subs.get(tn, []):
            code = s.get('code', '')
            if not code:
                continue
            try:
                import tokenize as _tokenize, io as _io
                tokens = _tokenize.generate_tokens(_io.StringIO(code).readline)
                for tok in tokens:
                    if tok.type == _tokenize.NAME:
                        name = tok.string
                        if name not in allowed and not name.startswith('__'):
                            all_identifiers.add(name)
            except Exception:
                # Fallback: regex-based extraction
                for m in re.finditer(r'\b([a-zA-Z_]\w*)\b', code):
                    name = m.group(1)
                    if name not in allowed and not name.startswith('__'):
                        all_identifiers.add(name)

    if not all_identifiers:
        print('  No variable names to generalise')
        return

    # Sort by length (longest first) to avoid partial replacements
    sorted_vars = sorted(all_identifiers, key=len, reverse=True)
    print(f'  Found {len(sorted_vars)} non-whitelist identifiers in student code')
    print(f'  Sample: {", ".join(sorted_vars[:10])}')

    modified = 0
    for fs in all_fs:
        regex = fs.get('regex', '')
        if not regex:
            continue

        original = regex
        for var in sorted_vars:
            # Only replace whole-word matches (not part of longer identifiers)
            # Use word boundary matching
            pattern = r'(?<!\\)\b' + re.escape(var) + r'\b'
            replacement = r'\\w+'
            new_regex = re.sub(pattern, replacement, regex)
            if new_regex != regex:
                regex = new_regex

        if regex != original:
            fs['regex'] = regex
            fs['_variable_generalised'] = True
            fs.setdefault('_original_regex', original)
            modified += 1

    print(f'  Generalised {modified}/{len(all_fs)} FS regexes')


def _inject_blind_test_findings(report: dict, all_fs: list[dict],
                                  task_subs: dict[int, list[dict]],
                                  all_readmes: dict):
    """Compute per-FS false positive rate using README ground truth.

    Adds high-FPR negative FS to the report's needs_fix section so
    Round 2 AI can fix them.
    """
    from ground_truth import parse_readme

    fs_lookup = {f['id']: f for f in all_fs}

    # Per-negative-FS: count matches on students WITHOUT bad patterns
    per_fs_fp: dict[str, dict] = {}

    for fn in [1, 2, 3]:
        task_crit_prefix = f'RQ{fn}'
        task_subs_list = task_subs.get(fn, [])

        for s in task_subs_list:
            sid = s['student']
            code = s.get('code', '')
            if not code:
                continue

            # Get ground truth for this student
            rdata = all_readmes.get(sid, {})
            criteria_raw = rdata.get('criteria', '{}')
            if isinstance(criteria_raw, str):
                try:
                    criteria_raw = eval(criteria_raw)
                except Exception:
                    criteria_raw = {}
            gt = criteria_raw if isinstance(criteria_raw, dict) else {}

            for fs in all_fs:
                if fs.get('fs_type') != 'negative':
                    continue
                crit = fs.get('criterion', '')
                if not crit.startswith(task_crit_prefix):
                    continue
                regex = fs.get('regex', '')
                if not regex:
                    continue
                try:
                    compiled = re.compile(regex, re.IGNORECASE | re.DOTALL)
                    matched = compiled.search(code) is not None
                except re.error:
                    continue

                fid = fs.get('id', '?')
                if fid not in per_fs_fp:
                    per_fs_fp[fid] = {'total_matches': 0, 'fp_matches': 0}

                if matched:
                    per_fs_fp[fid]['total_matches'] += 1
                    # Check if student has bad patterns for this criterion
                    crit_gt = gt.get(crit, {})
                    has_bad = bool(crit_gt.get('bad', []))
                    if not has_bad:
                        per_fs_fp[fid]['fp_matches'] += 1

    # Find high-FPR FS (FPR > 0.5, at least 5 total matches)
    injected = 0
    for fid, data in per_fs_fp.items():
        total = data['total_matches']
        fp = data['fp_matches']
        if total < 5:
            continue
        fpr = fp / total
        if fpr > 0.5:
            fs = fs_lookup.get(fid, {})
            # Check not already in needs_fix
            already = any(item.get('fs_id') == fid for item in report.get('needs_fix', []))
            if not already:
                report['needs_fix'].append({
                    'fs_id': fid,
                    'fs_type': 'negative',
                    'criterion': fs.get('criterion', '?'),
                    'problem': 'HIGH_FALSE_POSITIVE',
                    'detail': f'Matches {total} students but {fp} ({fpr:.0%}) do NOT have the bad pattern (ground truth)',
                    'suggestion': 'Tighten regex to only match students who actually made this mistake. '
                                  'Add more specific code pattern that distinguishes real errors from correct code.',
                })
                injected += 1

    if injected > 0:
        print(f'  Blind test: {injected} high-FPR negative FS added to needs_fix')
    else:
        print(f'  Blind test: no high-FPR FS found ({len(per_fs_fp)} negative FS checked)')


def _remove_negative_fs_for_no_bad_criteria(all_fs: list[dict],
                                              all_readmes: dict):
    """Remove negative FS for criteria that have ZERO bad patterns in ground truth.

    If no student has bad patterns for criterion X, any negative FS matching X
    is a false positive by definition. Deterministic, 100% reliable.
    """
    print(f'\n{"=" * 60}')
    print('  NEGATIVE FS — CRITERIA FILTER')
    print('  (removing neg FS for criteria with 0 bad patterns)')
    print('=' * 60)

    # Collect criteria that have bad patterns from READMEs
    criteria_with_bad: set[str] = set()
    for sid, rdata in all_readmes.items():
        criteria = rdata.get('criteria', {})
        if isinstance(criteria, str):
            try:
                criteria = eval(criteria)
            except Exception:
                continue
        if not isinstance(criteria, dict):
            continue
        for crit, patterns in criteria.items():
            if patterns.get('bad', []):
                criteria_with_bad.add(crit)

    # Also trust the bad pattern summary for criteria that exist in README
    removed = 0
    for fs in list(all_fs):  # iterate copy to allow removal
        if fs.get('fs_type') != 'negative':
            continue
        crit = fs.get('criterion', '')
        if crit and crit not in criteria_with_bad:
            fs['_excluded_no_bad_patterns'] = True
            fs['_scoring_weight'] = 0.0
            removed += 1
            print(f'  EXCLUDED {fs.get("id","?")}: [{crit}] no students have bad patterns')

    print(f'  Removed: {removed} negative FS (criteria without bad patterns)')

    # Also print which criteria DO have bad patterns
    all_fs_criteria = set(fs.get('criterion', '') for fs in all_fs if fs.get('fs_type') == 'negative')
    for crit in sorted(all_fs_criteria):
        if crit in criteria_with_bad:
            pass  # OK
        elif crit:
            print(f'  NOTE: {crit} has negative FS but no bad patterns in README — all excluded')


def run_plan_d(question_dir: str, submissions_dir: str,
                question_id: str = '', ref_dir: str = '') -> list[dict]:
    """Full Plan D iterative refinement pipeline.

    Phase 0 (whitelist) → Round 1 (draft) → Validate → Round 2 (refine) →
    Validate → Round 3 (targeted) → FCC safety net → Quality Gates → Output.
    """
    print('=' * 60)
    print('  PLAN D: ITERATIVE REFINEMENT PIPELINE')
    print(f'  Model: {DEEPSEEK_MODEL}')
    print('=' * 60)

    if not ref_dir:
        ref_dir = os.path.join(BASE_DIR, 'references', question_id)
    os.makedirs(ref_dir, exist_ok=True)

    # ── Phase 0: Question Analysis with Whitelist ──
    print(f'\n{"=" * 60}')
    print('  Phase 0: Question Analysis + Identifier Whitelist')
    print('=' * 60)
    rubric_cache = os.path.join(BASE_DIR, 'output', question_id, 'rubric_cache.json')
    whitelist_cache = os.path.join(BASE_DIR, 'output', question_id, 'whitelist_cache.json')

    # Use ORIGINAL Phase 0 for rubric structure (produces Task1/Task2/Task3)
    from ai_pipeline import phase0_analyze_question
    question_config = phase0_analyze_question(question_dir, rubric_cache)

    # Run whitelist-extraction Phase 0 separately
    wl_result = phase0_with_whitelist(question_dir, whitelist_cache)
    if wl_result:
        identifier_whitelist = wl_result.get('identifier_whitelist', {})
        question_config['identifier_whitelist'] = identifier_whitelist
    else:
        identifier_whitelist = {}
        question_config['identifier_whitelist'] = identifier_whitelist

    # Inject whitelist into ALL subsequent API calls via system prompt
    set_whitelist(identifier_whitelist)

    tasks = question_config.get('tasks', [])
    question_name = question_config.get('question_name', question_id)

    # ── Phase 0.5: Load README ground truth ──
    print(f'\n{"=" * 60}')
    print('  Phase 0.5: Loading README Ground Truth')
    print('=' * 60)
    all_readmes = load_all_readmes(submissions_dir)
    print(f'  Loaded {len(all_readmes)} student READMEs')

    cw_source_dir = os.path.join(BASE_DIR, '..', 'CW-generater', 'submissions_imusic_v5')
    if os.path.isdir(cw_source_dir):
        verification = verify_all_readmes(all_readmes, cw_source_dir)
        print_verification_report(verification)

    inventory = build_pattern_inventory(all_readmes)
    for task_id_str in ['Task1', 'Task2', 'Task3']:
        inv = inventory.get(task_id_str, {})
        print(f'  {task_id_str}: {len(inv.get("good", []))} good, '
              f'{len(inv.get("bad", []))} bad patterns')

    # ── Phase 1: Reference solutions ──
    from ai_pipeline import phase1_ensure_references
    print(f'\n{"=" * 60}')
    print('  Phase 1: Reference Solutions')
    print('=' * 60)
    phase1_ensure_references(tasks, question_dir, ref_dir)

    # ── Phase 1.5: Sandbox Verification ──
    print(f'\n{"=" * 60}')
    print('  Phase 1.5: Sandbox Verification')
    print('=' * 60)
    from sandbox import verify_references as sandbox_verify, print_summary as sandbox_summary
    sandbox_report = sandbox_verify(question_dir, ref_dir)
    sandbox_summary(sandbox_report)

    # ── Collect submissions ──
    print(f'\n{"=" * 60}')
    print('  Collecting Submissions')
    print('=' * 60)
    task_subs = {}
    for tn in [1, 2, 3]:
        subs = collect_submissions_by_task(submissions_dir, tn, max_students=None)
        task_subs[tn] = subs
        task_id_str = f'Task{tn}'
        if subs:
            tiers = defaultdict(int)
            for s in subs:
                sid = s['student']
                if sid in all_readmes:
                    tiers[all_readmes[sid].get('quality_tier', '?')] += 1
            print(f'  {task_id_str}: {len(subs)} students {dict(tiers)}')

    # ── Build reference & template code ──
    all_ref_code = ''
    for task in tasks:
        for rf in task.get('reference_files', []):
            for root, _, files in os.walk(ref_dir):
                if rf in files:
                    all_ref_code += read_file(os.path.join(root, rf)) + '\n'

    template_code = ''
    for dp, _, fn in os.walk(os.path.join(question_dir, 'code')):
        for f in fn:
            if f.endswith('.py'):
                template_code += read_file(os.path.join(dp, f)) + '\n'
    if not template_code:
        for dp, _, fn in os.walk(question_dir):
            for f in fn:
                if f.endswith('.py'):
                    template_code += read_file(os.path.join(dp, f)) + '\n'

    # ── Phase 1.5: Behavioral Sandbox (runtime testing) ──
    # Runs student code in subprocess isolation, collects behavioral fingerprints,
    # and clusters students by observed behavior for each criterion.
    # This provides GROUND TRUTH about what code actually DOES, not just what
    # README labels say.
    behavioral_fingerprints = None
    behavioral_clusters = None

    sandbox_enabled = os.getenv('TAFFIES_SANDBOX', '1') == '1'
    if sandbox_enabled:
        print(f'\n{"=" * 60}')
        print('  PHASE 1.5: BEHAVIORAL SANDBOX')
        print('=' * 60)

        from runtime.batch_runner import (
            collect_students, load_or_run_fingerprints, cluster_by_behavior,
        )
        from runtime.test_generator import get_testable_criteria

        students = collect_students()
        testable = get_testable_criteria()
        print(f'  Students: {len(students)}, Testable criteria: {testable}')

        fp_cache = os.path.join(BASE_DIR, 'output', question_id, 'behavioral_fingerprints.json')
        behavioral_fingerprints = load_or_run_fingerprints(
            fp_cache, students, testable,
            force=('--force-sandbox' in sys.argv),
        )

        behavioral_clusters = cluster_by_behavior(behavioral_fingerprints, testable)
        for crit, clist in sorted(behavioral_clusters.items()):
            print(f'  {crit}: {len(clist)} behavioral clusters')
            for c in clist:
                print(f'    [{c["count"]:2d}] {c["label"][:80]}')
    else:
        print('\n  Phase 1.5 SKIPPED (TAFFIES_SANDBOX=0)')

    # ── TAFFIES-ALIGNED FS GENERATION (ALL FS: positive + negative) ──
    # Replaces old Round 1/2/3. Every FS follows the TAFFIES definition:
    # signature = pattern from ACTUAL student code, feedback = specific to that pattern.
    # No Type A (broad "missing X"), no rubric-inferred categories.
    from taffies_fs_generator import generate_taffies_fs as _generate_taffies_fs

    use_reasoner = os.getenv('DEEPSEEK_USE_REASONER', '') == '1'
    taffies_model = 'deepseek-reasoner' if use_reasoner else None

    all_fs = _generate_taffies_fs(
        tasks, task_subs, all_readmes, all_ref_code, template_code,
        model_override=taffies_model,
        min_cluster_size=2,
        behavioral_fingerprints=behavioral_fingerprints,
        behavioral_clusters=behavioral_clusters,
    )

    # Assign IDs
    fs_id_counter: dict[str, int] = {}
    for f in all_fs:
        crit = f.get('criterion', '?')
        m = re.search(r'\d+', str(crit))
        num = m.group() if m else '0'
        fs_id_counter.setdefault(num, 0)
        fs_id_counter[num] += 1
        f['id'] = f'FS{num}.{fs_id_counter[num]}'

    _save_plan_d_snapshot(all_fs, question_id, '01_taffies_generated')

    # ── FCC Safety Net ──
    print(f'\n{"=" * 60}')
    print('  FCC SAFETY NET')
    print('=' * 60)

    # Build all_subs_by_batch for FCC
    all_subs_by_batch = {
        'q1-': task_subs.get(1, []),
        'q2-': task_subs.get(2, []),
        'q3-': task_subs.get(3, []),
    }

    fcc_supplement_loop(all_fs, all_subs_by_batch, tasks, max_rounds=2,
                         ref_code=all_ref_code, template_code=template_code)
    _save_plan_d_snapshot(all_fs, question_id, '04_fcc_safety_net')

    # ── Post-processing: Remove negative FS for criteria with no bad patterns ──
    # Deterministic: if README says no student has bad patterns for criterion X,
    # all negative FS for X are false positives by definition. Remove them.
    _remove_negative_fs_for_no_bad_criteria(all_fs, all_readmes)
    _save_plan_d_snapshot(all_fs, question_id, '04b_post_criteria_filter')

    # ── Post-processing: Broad Negative FS filter ──
    # Removes negative FS that match >40% of students (too broad to be useful).
    # This is assignment-agnostic — works for any Python task.
    _apply_broad_negative_filter(all_fs, task_subs)
    _save_plan_d_snapshot(all_fs, question_id, '05_post_broad_filter')

    # ── Post-processing: Variable name generalisation ──
    # Replaces hardcoded variable names in regex with \w+.
    # Assignment-agnostic — tokenizes student code, identifies non-whitelist identifiers.
    _apply_variable_generalisation(all_fs, task_subs, identifier_whitelist)
    _save_plan_d_snapshot(all_fs, question_id, '06_post_variable_gen')

    # ── Round 4: Positive/Negative Example Refinement ──
    # For each high-FPR negative FS, shows AI concrete TP/FP code examples
    # and asks it to tighten the regex. Validates: must keep TP, must drop FP.
    print(f'\n{"=" * 60}')
    print('  ROUND 4: Positive/Negative Example Refinement')
    print('=' * 60)
    from round4_refine import round4_refine as _round4_refine
    all_fs = _round4_refine(
        all_fs, task_subs, all_readmes, all_ref_code,
        max_fs_to_refine=30, min_matches=3, min_fpr=0.3,
        question_id=question_id,
    )
    _save_plan_d_snapshot(all_fs, question_id, '06b_round4_refined')

    # ── Quality Gates (original apply_quality_gates skipped) ──
    all_subs_flat = []
    seen = set()
    for tn in [1, 2, 3]:
        for s in task_subs.get(tn, []):
            if s['student'] not in seen:
                seen.add(s['student'])
                all_subs_flat.append(s)
    # apply_quality_gates kept in codebase but not called in Plan D pipeline

    # ── Late-Stage MATCHES_NOTHING Gate (SKIPPED) ──
    print(f'\n{"=" * 60}')
    print('  LATE-STAGE GATE: SKIPPED')
    print('  (all FS retained without MATCHES_NOTHING check)')
    print('=' * 60)
    # BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}
    # ... (late-stage gate code preserved in ai_pipeline.py)
    _save_plan_d_snapshot(all_fs, question_id, '07_post_late_gate_skipped')

    # ── Final Audit & Output ──
    from audit import final_audit
    audit_result = {}
    if all_fs:
        try:
            audit_result = final_audit(all_fs, all_subs_by_batch, all_ref_code, template_code)
        except Exception as e:
            print(f'  WARNING: Audit failed: {e}')
            audit_result = {'coverage_pct': 0, 'full_weight': 0, 'reduced_weight': 0, 'excluded': 0}
    else:
        audit_result = {'coverage_pct': 0, 'full_weight': 0, 'reduced_weight': 0, 'excluded': 0}

    out_dir = os.path.join(BASE_DIR, 'output', question_id or 'output')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'fs_registry.json')

    output = {
        'generated_at': datetime.now().isoformat(),
        'question': question_name,
        'model': DEEPSEEK_MODEL,
        'pipeline': 'Plan D + Round 4 (example refinement)',
        'total_fs': len(all_fs),
        'coverage_pct': audit_result.get('coverage_pct', 0),
        'audit': audit_result,
        'identifier_whitelist': identifier_whitelist,
        'fs_registry': all_fs,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f'\n{"=" * 60}')
    print(f'  PLAN D COMPLETE')
    print(f'  FS: {len(all_fs)} | Coverage: {audit_result.get("coverage_pct", 0)}%')
    print(f'  Full-weight: {audit_result.get("full_weight", 0)} | '
          f'Reduced: {audit_result.get("reduced_weight", 0)} | Excluded: {audit_result.get("excluded", 0)}')
    print(f'  Output: {json_path}')
    print('=' * 60)

    return all_fs


def _save_plan_d_snapshot(all_fs: list[dict], question_id: str, label: str):
    """Save snapshot of FS at current Plan D stage."""
    from ai_pipeline import _parse_flags
    snap_dir = os.path.join(BASE_DIR, 'output', question_id, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    total = len(all_fs)
    pos = sum(1 for f in all_fs if f.get('fs_type') == 'positive')
    neg = sum(1 for f in all_fs if f.get('fs_type') == 'negative')
    fw = sum(1 for f in all_fs if f.get('_scoring_weight', 1.0) == 1.0)
    rw = sum(1 for f in all_fs if f.get('_scoring_weight', 1.0) == 0.5)
    ex = sum(1 for f in all_fs if f.get('_scoring_weight', 1.0) == 0.0)

    task_counts = defaultdict(lambda: {'positive': 0, 'negative': 0, 'other': 0})
    for f in all_fs:
        ft = f.get('fs_type', '?')
        if ft not in ('positive', 'negative'):
            ft = 'other'
        task_counts[f.get('task', '?')][ft] += 1

    # Source distribution
    source_counts = defaultdict(int)
    for f in all_fs:
        source_counts[f.get('source', 'unknown')] += 1

    snapshot = {
        'label': label,
        'timestamp': datetime.now().isoformat(),
        'total_fs': total,
        'positive': pos,
        'negative': neg,
        'full_weight': fw,
        'reduced_weight': rw,
        'excluded': ex,
        'by_task': {t: dict(c) for t, c in task_counts.items()},
        'by_source': dict(source_counts),
    }

    path = os.path.join(snap_dir, f'{label}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    # Also save full FS
    fs_path = os.path.join(snap_dir, f'{label}_fs_registry.json')
    with open(fs_path, 'w', encoding='utf-8') as f:
        json.dump(all_fs, f, indent=2, ensure_ascii=False, default=str)
    print(f'  [snapshot] {label}: {total} FS ({pos}+/{neg}-) fw={fw} rw={rw} ex={ex}')


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        q_dir = sys.argv[1]
        s_dir = sys.argv[2]
        qid = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.abspath(q_dir))
        run_plan_d(q_dir, s_dir, qid)
    else:
        print('Usage: python plan_d.py <question_dir> <submission_dir> [question_id]')
        print('Example: python plan_d.py question submission q1_iMusic')
        sys.exit(1)
