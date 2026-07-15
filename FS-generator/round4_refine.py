"""
Round 4: Positive/Negative Example Refinement for High-FPR Negative FS
======================================================================
Uses blind test results to provide AI with concrete TP (must-keep matching)
and FP (must-stop matching) examples for each high-FPR negative FS.

The AI sees:
  - The current regex
  - Students it MUST continue to match (TP: has the bad pattern)
  - Students it MUST stop matching (FP: correct code incorrectly flagged)
  - The specific code lines that caused the match

Then AI tightens the regex. We validate:
  1. Tightened regex still matches ALL TP students (no recall loss)
  2. Tightened regex no longer matches FP students (precision gain)
  3. If validation fails, keep original regex (safety fallback)

Usage:
    python round4_refine.py output/q1_iMusic/fs_registry.json submission

Integrated into Plan D via:
    from round4_refine import round4_refine
    all_fs = round4_refine(all_fs, task_subs, all_readmes, all_ref_code)
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

from ai_pipeline import call_deepseek as _call_deepseek_original, extract_json, _repair_json
from coverage import _parse_flags
from ground_truth import load_all_readmes, parse_readme


# ============================================================
# API call wrapper
# ============================================================

ROUND4_SYSTEM = """You are an expert in writing precise Python regex patterns for code analysis.
Your job: tighten a regex so it stops matching FALSE POSITIVES while continuing to match
TRUE POSITIVES.

CRITICAL RULES:

1. STUDY the TP (True Positive) code first. These students GENUINELY have the bad pattern.
   Your tightened regex MUST still match ALL TP students.

2. STUDY the FP (False Positive) code. These students wrote CORRECT code but are being
   incorrectly flagged. Your tightened regex MUST NOT match ANY FP student.

3. TIGHTENING STRATEGIES (in priority order):
   a. SCOPE NARROWING: Add function-name anchoring. Instead of matching anywhere in the file,
      match only inside the specific function. Use:
        def\\s+target_func_name\\s*\\([^)]*\\)\\s*:(?![\\s\\S]*?required_good_pattern)

   b. ADD EXCLUSION PATTERNS: If the FP students use a different-but-correct approach,
      add a negative lookahead to exclude that pattern:
        (?!.*correct_alternative_pattern)

   c. REQUIRE MORE CONTEXT: Add surrounding code context that only appears in the bad pattern.
      Instead of matching just the mistake, match the mistake IN CONTEXT:
        cursor\\.execute\\(f["']  ← Type B: detects f-string SQL
        NOT just f["'] anywhere (matches flash() messages)

   d. ADD TYPE-SPECIFIC GUARDS: If the bad pattern is SQL-related, ensure the regex
      only matches inside .execute() calls, not inside flash() or print().

4. For Type A negative FS (missing good pattern → uses (?!...)):
   - The (?!...) negative lookahead may be too greedy and exclude correct variants.
   - FIX: Add specific exclusion for the FP pattern inside the lookahead.
   - FIX: Or restructure to use (?=.*required_structure)(?!.*excluded_structure).

5. For Type B negative FS (detecting presence of bad pattern):
   - Add more specific context around the bad pattern.
   - Require database-related keywords nearby (cursor, execute, INSERT, etc.).
   - Exclude patterns that look similar but are not SQL (flash messages, print statements).

6. REGEX CONSTRUCTION RULES:
   - Use [\\s\\S]*? for cross-line non-greedy matching (NOT .* or .+)
   - Use \\\\w+ for variable names, \\\\s+ for whitespace
   - Use (?:alt1|alt2) for alternation of known function/table names
   - NEVER use { or } (Python doesn't use curly braces for blocks)
   - Escape literal dots, parens, brackets: \\., \\(, \\), \\[, \\]

7. OUTPUT: Only the TIGHTENED REGEX STRING. No JSON, no markdown, no explanation.
   Just the regex pattern itself, one line."""


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str | None:
    """Call DeepSeek API."""
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f'  [Round4] API call failed: {e}')
        return None


# ============================================================
# Blind test analysis
# ============================================================

def compute_per_fs_metrics(
    all_fs: list[dict],
    task_subs: dict[int, list[dict]],
    all_readmes: dict,
) -> dict[str, dict]:
    """Compute per-negative-FS TP/FP/FN counts using ground truth.

    Returns:
        {fs_id: {
            'tp_students': [...],   # has bad pattern + FS matched
            'fp_students': [...],   # no bad pattern + FS matched
            'fn_students': [...],   # has bad pattern + FS NOT matched
            'tp_count': int, 'fp_count': int, 'fn_count': int,
            'fpr': float,           # FP / (TP + FP)
            'total_matches': int,
        }}
    """
    per_fs: dict[str, dict] = {}

    # Build FS lookup
    fs_lookup = {}
    for fs in all_fs:
        fid = fs.get('id', '')
        if fid:
            fs_lookup[fid] = fs

    # Compile negative FS only
    neg_fs_compiled: dict[str, re.Pattern] = {}
    for fs in all_fs:
        if fs.get('fs_type') != 'negative':
            continue
        if fs.get('_scoring_weight', 1.0) == 0.0:
            continue
        regex = fs.get('regex', '')
        if not regex:
            continue
        try:
            flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
            neg_fs_compiled[fs['id']] = re.compile(regex, flags)
        except re.error:
            continue

    # For each task, check each student
    for task_num in [1, 2, 3]:
        task_crit_prefix = f'RQ{task_num}'
        subs = task_subs.get(task_num, [])

        for s in subs:
            sid = s['student']
            code = s.get('code', '')
            if not code:
                continue

            # Ground truth
            rdata = all_readmes.get(sid, {})
            criteria_raw = rdata.get('criteria', '{}')
            if isinstance(criteria_raw, str):
                try:
                    criteria_raw = eval(criteria_raw)
                except Exception:
                    criteria_raw = {}
            gt = criteria_raw if isinstance(criteria_raw, dict) else {}

            for fid, compiled in neg_fs_compiled.items():
                fs = fs_lookup.get(fid, {})
                crit = fs.get('criterion', '')
                if not crit.startswith(task_crit_prefix):
                    continue

                if fid not in per_fs:
                    per_fs[fid] = {
                        'tp_students': [], 'fp_students': [], 'fn_students': [],
                        'tp_count': 0, 'fp_count': 0, 'fn_count': 0,
                        'fpr': 0.0, 'total_matches': 0,
                        'criterion': crit,
                        'regex': fs.get('regex', ''),
                        'name': fs.get('name', ''),
                        'fs_type': fs.get('fs_type', ''),
                    }

                try:
                    matched = compiled.search(code) is not None
                except re.error:
                    continue

                crit_gt = gt.get(crit, {})
                has_bad = bool(crit_gt.get('bad', []))

                if matched:
                    per_fs[fid]['total_matches'] += 1
                    if has_bad:
                        per_fs[fid]['tp_students'].append(sid)
                        per_fs[fid]['tp_count'] += 1
                    else:
                        per_fs[fid]['fp_students'].append(sid)
                        per_fs[fid]['fp_count'] += 1
                elif has_bad:
                    per_fs[fid]['fn_students'].append(sid)
                    per_fs[fid]['fn_count'] += 1

    # Compute FPR
    for fid, data in per_fs.items():
        total = data['tp_count'] + data['fp_count']
        if total > 0:
            data['fpr'] = data['fp_count'] / total

    return per_fs


def select_high_fpr_fs(per_fs: dict[str, dict],
                       min_matches: int = 3,
                       min_fpr: float = 0.3) -> list[dict]:
    """Select negative FS with high false positive rates for refinement.

    Args:
        per_fs: Output of compute_per_fs_metrics().
        min_matches: Minimum total matches to consider (avoid noisy low-N FS).
        min_fpr: Minimum false positive rate to trigger refinement.

    Returns:
        List of FS entries sorted by FP count descending.
    """
    candidates = []
    for fid, data in per_fs.items():
        total = data['tp_count'] + data['fp_count']
        if total < min_matches:
            continue
        if data['fpr'] >= min_fpr and data['fp_count'] >= 1:
            candidates.append({
                'fs_id': fid,
                **data,
            })

    candidates.sort(key=lambda x: -x['fp_count'])
    return candidates


# ============================================================
# Collect code examples for TP/FP students
# ============================================================

def _extract_matched_lines(code: str, regex: str, flags: int,
                           context_lines: int = 2) -> str:
    """Extract the lines around a regex match in student code."""
    try:
        compiled = re.compile(regex, flags)
    except re.error:
        return ''

    match = compiled.search(code)
    if not match:
        return ''

    lines = code.split('\n')
    match_start_line = code[:match.start()].count('\n')
    start = max(0, match_start_line - context_lines)
    end = min(len(lines), match_start_line + context_lines + 1)
    snippet = '\n'.join(lines[start:end])
    return snippet.strip()


def collect_examples(
    candidates: list[dict],
    task_subs: dict[int, list[dict]],
    all_readmes: dict,
    max_tp: int = 5,
    max_fp: int = 5,
) -> list[dict]:
    """For each candidate FS, collect TP and FP code examples.

    Returns candidates enriched with 'tp_examples' and 'fp_examples'.
    """
    # Build student code lookup
    code_lookup: dict[str, tuple[str, int]] = {}  # sid -> (code, task_num)
    for tn in [1, 2, 3]:
        for s in task_subs.get(tn, []):
            sid = s['student']
            if sid not in code_lookup:
                code_lookup[sid] = (s.get('code', ''), tn)

    for c in candidates:
        regex = c['regex']
        flags = _parse_flags('IGNORECASE')

        tp_examples = []
        for sid in c['tp_students'][:max_tp]:
            code_info = code_lookup.get(sid)
            if not code_info:
                continue
            code, _ = code_info
            matched_lines = _extract_matched_lines(code, regex, flags)
            if matched_lines:
                tp_examples.append({
                    'student': sid,
                    'code': matched_lines,
                })

        fp_examples = []
        for sid in c['fp_students'][:max_fp]:
            code_info = code_lookup.get(sid)
            if not code_info:
                continue
            code, _ = code_info
            matched_lines = _extract_matched_lines(code, regex, flags)
            if matched_lines:
                fp_examples.append({
                    'student': sid,
                    'code': matched_lines,
                })

        c['tp_examples'] = tp_examples
        c['fp_examples'] = fp_examples

    return candidates


# ============================================================
# Build refinement prompts
# ============================================================

def build_round4_prompt(candidate: dict) -> str:
    """Build the AI refinement prompt for a single high-FPR FS.

    Shows the AI:
    - Current regex and its problem
    - TP students: code that MUST still match
    - FP students: code that MUST stop matching
    """
    fid = candidate['fs_id']
    criterion = candidate.get('criterion', '?')
    name = candidate.get('name', '?')
    regex = candidate.get('regex', '')
    fpr = candidate.get('fpr', 0)
    tp_count = candidate.get('tp_count', 0)
    fp_count = candidate.get('fp_count', 0)

    tp_examples = candidate.get('tp_examples', [])
    fp_examples = candidate.get('fp_examples', [])

    lines = [f"## Tighten regex for FS: {fid}"]
    lines.append(f"")
    lines.append(f"**Criterion**: {criterion}")
    lines.append(f"**Name**: {name}")
    lines.append(f"**Current FPR**: {fpr:.0%} ({fp_count} false / {tp_count + fp_count} total matches)")
    lines.append(f"")
    lines.append(f"### Current Regex")
    lines.append(f"```")
    lines.append(f"{regex}")
    lines.append(f"```")
    lines.append(f"")

    if tp_examples:
        lines.append(f"### [KEEP] MUST CONTINUE MATCHING ({len(tp_examples)} TP examples)")
        lines.append(f"These students GENUINELY have the bad pattern. Your tightened regex MUST still match ALL of them.")
        lines.append(f"")
        for ex in tp_examples:
            lines.append(f"**{ex['student']}**:")
            lines.append(f"```python")
            lines.append(f"{ex['code'][:500]}")
            lines.append(f"```")
            lines.append(f"")

    if fp_examples:
        lines.append(f"### [REMOVE] MUST STOP MATCHING ({len(fp_examples)} FP examples)")
        lines.append(f"These students wrote CORRECT code but are incorrectly flagged.")
        lines.append(f"Your tightened regex MUST NOT match ANY of them.")
        lines.append(f"")
        for ex in fp_examples:
            lines.append(f"**{ex['student']}**:")
            lines.append(f"```python")
            lines.append(f"{ex['code'][:500]}")
            lines.append(f"```")
            lines.append(f"")

    lines.append(f"### Instructions")
    lines.append(f"1. Study the FP code to understand WHY the current regex is over-matching")
    lines.append(f"2. Study the TP code to understand what the real bad pattern looks like")
    lines.append(f"3. Find the KEY DIFFERENCE between TP and FP code")
    lines.append(f"4. Write a tightened regex that captures this difference")
    lines.append(f"5. Verify mentally: does it match all TP? Does it exclude all FP?")
    lines.append(f"")
    lines.append(f"Output ONLY the tightened regex string (one line, no markdown, no explanation).")

    return '\n'.join(lines)


# ============================================================
# Validate tightened regex
# ============================================================

def validate_tightened_regex(
    candidate: dict,
    new_regex: str,
    code_lookup: dict[str, tuple[str, int]],
) -> dict:
    """Validate that the tightened regex:
    1. Still matches all TP students (no recall loss)
    2. No longer matches FP students (precision gain)

    Returns:
        {'valid': bool, 'tp_lost': [...], 'fp_remaining': [...], 'reason': str}
    """
    result = {
        'valid': True,
        'tp_lost': [],
        'fp_remaining': [],
        'reason': '',
    }

    try:
        compiled = re.compile(new_regex, re.IGNORECASE | re.DOTALL)
    except re.error as e:
        result['valid'] = False
        result['reason'] = f'Regex compile error: {e}'
        return result

    # Check TP: must still match ALL
    for sid in candidate.get('tp_students', []):
        code_info = code_lookup.get(sid)
        if not code_info:
            continue
        code, _ = code_info
        try:
            if not compiled.search(code):
                result['tp_lost'].append(sid)
                result['valid'] = False
        except re.error:
            result['tp_lost'].append(sid)
            result['valid'] = False

    # Check FP: must NOT match ANY
    for sid in candidate.get('fp_students', []):
        code_info = code_lookup.get(sid)
        if not code_info:
            continue
        code, _ = code_info
        try:
            if compiled.search(code):
                result['fp_remaining'].append(sid)
                # FP remaining is a partial success (fewer FP is still better)
                # Only invalidate if ALL FP remain
        except re.error:
            pass

    if result['tp_lost']:
        result['reason'] = f'Lost {len(result["tp_lost"])} TP: {", ".join(result["tp_lost"][:3])}'
    elif result['fp_remaining']:
        # Check if we improved (fewer FP than original)
        original_fp = len(candidate.get('fp_students', []))
        if len(result['fp_remaining']) < original_fp:
            result['reason'] = f'Improved: {original_fp}→{len(result["fp_remaining"])} FP (kept)'
            result['valid'] = True  # Partial improvement is accepted
        else:
            result['reason'] = f'No improvement: {len(result["fp_remaining"])} FP remain'
            result['valid'] = False
    else:
        result['reason'] = 'Perfect: all TP kept, zero FP'

    return result


# ============================================================
# Main Round 4 function
# ============================================================

def round4_refine(
    all_fs: list[dict],
    task_subs: dict[int, list[dict]],
    all_readmes: dict,
    all_ref_code: str = '',
    max_fs_to_refine: int = 30,
    min_matches: int = 3,
    min_fpr: float = 0.3,
    dry_run: bool = False,
    question_id: str = 'q1_iMusic',
) -> list[dict]:
    """Round 4: Positive/Negative Example Refinement.

    For each high-FPR negative FS:
    1. Collect TP students (have bad pattern, FS matched)
    2. Collect FP students (no bad pattern, FS matched incorrectly)
    3. Show AI both sets of code examples
    4. AI tightens the regex
    5. Validate: still matches TP, no longer matches FP
    6. If validation fails, keep original regex (safety fallback)

    Args:
        all_fs: Current FS registry list.
        task_subs: {task_num: [{student, code}, ...]}.
        all_readmes: Output of load_all_readmes().
        all_ref_code: Combined reference code (for cross-check).
        max_fs_to_refine: Max number of FS to refine (API calls = candidates).
        min_matches: Minimum total matches to consider an FS.
        min_fpr: Minimum FPR to trigger refinement.
        dry_run: If True, only compute metrics without making API calls.

    Returns:
        Updated all_fs list with tightened regexes.
    """
    print(f'\n{"=" * 60}')
    print('  ROUND 4: Positive/Negative Example Refinement')
    print(f'  (tightening high-FPR negative FS with concrete examples)')
    print('=' * 60)

    # Build code lookup
    code_lookup: dict[str, tuple[str, int]] = {}
    for tn in [1, 2, 3]:
        for s in task_subs.get(tn, []):
            sid = s['student']
            if sid not in code_lookup:
                code_lookup[sid] = (s.get('code', ''), tn)

    # Step 1: Compute per-FS metrics
    print(f'\n  --- Step 1: Computing per-FS TP/FP metrics ---')
    per_fs = compute_per_fs_metrics(all_fs, task_subs, all_readmes)
    print(f'  Analyzed {len(per_fs)} active negative FS')

    # Step 2: Select high-FPR candidates
    candidates = select_high_fpr_fs(per_fs, min_matches=min_matches, min_fpr=min_fpr)
    print(f'\n  --- Step 2: Selecting high-FPR candidates ---')
    print(f'  Found {len(candidates)} FS with FPR >= {min_fpr:.0%} and >= {min_matches} matches')

    if not candidates:
        print('  No high-FPR FS to refine — Round 4 complete!')
        return all_fs

    # Print top candidates
    print(f'\n  Top candidates for refinement:')
    for c in candidates[:10]:
        print(f'    {c["fs_id"]}: FPR={c["fpr"]:.0%} '
              f'(TP={c["tp_count"]}, FP={c["fp_count"]}, FN={c["fn_count"]}) '
              f'— {c["name"][:60]}')

    # Step 3: Collect code examples
    print(f'\n  --- Step 3: Collecting code examples ---')
    candidates = collect_examples(candidates, task_subs, all_readmes, max_tp=5, max_fp=5)
    for c in candidates[:10]:
        print(f'    {c["fs_id"]}: {len(c.get("tp_examples",[]))} TP, '
              f'{len(c.get("fp_examples",[]))} FP examples')

    if dry_run:
        print(f'\n  DRY RUN — no API calls made.')
        print(f'  Would refine {min(len(candidates), max_fs_to_refine)} FS.')
        _save_round4_report(candidates, [], all_fs, per_fs, question_id)
        return all_fs

    # Step 4: Refine each candidate (with API calls)
    print(f'\n  --- Step 4: AI refinement ({min(len(candidates), max_fs_to_refine)} FS) ---')

    fs_lookup = {f['id']: f for f in all_fs}
    refined_count = 0
    improved_count = 0
    failed_count = 0
    results_log: list[dict] = []

    for i, candidate in enumerate(candidates[:max_fs_to_refine]):
        fid = candidate['fs_id']
        print(f'\n  [{i+1}/{min(len(candidates), max_fs_to_refine)}] {fid}: '
              f'FPR={candidate["fpr"]:.0%} ({candidate["fp_count"]} FP)')

        if not candidate.get('fp_examples'):
            print(f'    SKIP: no FP code examples available')
            continue

        # Build prompt
        prompt = build_round4_prompt(candidate)
        print(f'    Prompt: {len(prompt)} chars, '
              f'{len(candidate.get("tp_examples",[]))} TP, '
              f'{len(candidate.get("fp_examples",[]))} FP examples')

        # Call AI
        response = None
        for attempt in range(3):
            response = call_deepseek(ROUND4_SYSTEM, prompt, temperature=0.2)
            if response and response.strip():
                break
            response = None

        if not response:
            print(f'    FAILED: API returned empty response')
            failed_count += 1
            results_log.append({
                'fs_id': fid,
                'outcome': 'api_failed',
                'original_regex': candidate['regex'],
                'new_regex': candidate['regex'],
            })
            continue

        # Extract regex from response (strip markdown fences, whitespace)
        new_regex = response.strip()
        # Remove markdown code fences if present
        new_regex = re.sub(r'^```(?:regex|python)?\s*\n?', '', new_regex)
        new_regex = re.sub(r'\n?```\s*$', '', new_regex)
        # Remove any surrounding quotes
        new_regex = new_regex.strip().strip('"\'')
        # Remove leading/trailing whitespace from each line
        new_regex = '\n'.join(line.strip() for line in new_regex.split('\n'))
        new_regex = new_regex.strip()

        if not new_regex or len(new_regex) < 5:
            print(f'    FAILED: empty or too-short regex')
            failed_count += 1
            results_log.append({
                'fs_id': fid,
                'outcome': 'empty_regex',
                'original_regex': candidate['regex'],
                'new_regex': candidate['regex'],
            })
            continue

        # Check if regex actually changed
        if new_regex == candidate['regex']:
            print(f'    UNCHANGED: AI returned identical regex')
            results_log.append({
                'fs_id': fid,
                'outcome': 'unchanged',
                'original_regex': candidate['regex'],
                'new_regex': new_regex,
            })
            continue

        # Step 5: Validate
        print(f'    Old: {candidate["regex"][:80]}')
        print(f'    New: {new_regex[:80]}')

        validation = validate_tightened_regex(candidate, new_regex, code_lookup)

        if validation['valid']:
            # Apply the tightened regex
            fs = fs_lookup.get(fid)
            if fs:
                fs['regex'] = new_regex
                fs['_round4_tightened'] = True
                fs['_round4_original_regex'] = candidate['regex']
                fs['_round4_original_fpr'] = candidate['fpr']
                fs['_round4_original_fp'] = candidate['fp_count']
                fs['source_detail'] = fs.get('source_detail', '') + '+round4'

            refined_count += 1
            if validation['reason'].startswith('Improved') or validation['reason'].startswith('Perfect'):
                improved_count += 1

            print(f'    [OK] {validation["reason"]}')
            results_log.append({
                'fs_id': fid,
                'outcome': 'refined',
                'validation': validation['reason'],
                'original_regex': candidate['regex'],
                'new_regex': new_regex,
                'original_fpr': candidate['fpr'],
                'original_fp': candidate['fp_count'],
            })
        else:
            print(f'    [REJECTED] {validation["reason"]} — keeping original')
            failed_count += 1
            results_log.append({
                'fs_id': fid,
                'outcome': 'rejected',
                'validation': validation['reason'],
                'original_regex': candidate['regex'],
                'new_regex': new_regex,
            })

    # Summary
    print(f'\n  --- Round 4 Summary ---')
    print(f'  Refined: {refined_count} (improved: {improved_count})')
    print(f'  Failed/Rejected: {failed_count}')
    print(f'  Total FS: {len(all_fs)}')

    _save_round4_report(candidates, results_log, all_fs, per_fs, question_id)

    return all_fs


def _save_round4_report(
    candidates: list[dict],
    results_log: list[dict],
    all_fs: list[dict],
    per_fs: dict[str, dict],
    question_id: str = 'q1_iMusic',
):
    """Save Round 4 report to output directory."""
    out_dir = os.path.join(BASE_DIR, 'output', question_id)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'candidates_count': len(candidates),
        'candidates': [
            {
                'fs_id': c['fs_id'],
                'criterion': c.get('criterion', '?'),
                'name': c.get('name', '?'),
                'fpr': c['fpr'],
                'tp_count': c['tp_count'],
                'fp_count': c['fp_count'],
                'fn_count': c['fn_count'],
            }
            for c in candidates[:30]
        ],
        'results': results_log,
        'refined_count': sum(1 for r in results_log if r['outcome'] == 'refined'),
        'failed_count': sum(1 for r in results_log if r['outcome'] in ('rejected', 'api_failed', 'empty_regex')),
    }

    path = os.path.join(out_dir, 'round4_refinement_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'  Report saved: {path}')


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        registry_path = sys.argv[1]
        submission_dir = sys.argv[2]
    else:
        registry_path = os.path.join(BASE_DIR, 'output', 'q1_iMusic', 'fs_registry.json')
        submission_dir = os.path.join(BASE_DIR, 'submission')

    # Load FS registry
    with open(registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_fs = data.get('fs_registry', [])

    # Load submissions by task
    from ai_pipeline import collect_submissions_by_task
    task_subs = {}
    for tn in [1, 2, 3]:
        task_subs[tn] = collect_submissions_by_task(submission_dir, tn, max_students=None)
        print(f'  Task{tn}: {len(task_subs[tn])} submissions')

    # Load READMEs
    all_readmes = load_all_readmes(submission_dir)
    print(f'  READMEs: {len(all_readmes)} students')

    # Load reference code
    ref_dir = os.path.join(BASE_DIR, 'references', 'q1_iMusic')
    all_ref_code = ''
    if os.path.isdir(ref_dir):
        for root, _, files in os.walk(ref_dir):
            for fn in files:
                if fn.endswith('.py'):
                    with open(os.path.join(root, fn), 'r', encoding='utf-8') as f:
                        all_ref_code += f.read() + '\n'

    # Run Round 4
    all_fs = round4_refine(
        all_fs, task_subs, all_readmes, all_ref_code,
        max_fs_to_refine=30, min_matches=3, min_fpr=0.3,
    )

    # Save updated registry
    data['fs_registry'] = all_fs
    data['total_fs'] = len(all_fs)
    data['pipeline'] = data.get('pipeline', 'Plan D') + ' + Round 4 (example refinement)'
    data['generated_at'] = datetime.now().isoformat()

    out_path = registry_path.replace('.json', '_round4.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'\n  Updated registry saved: {out_path}')
