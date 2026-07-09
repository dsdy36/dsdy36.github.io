"""
TAFFIES Final Audit
====================
Comprehensive FS quality check: false negatives, false positives,
regex integrity, TAFFIES FCC coverage, scoring weight distribution.
"""

import re
from collections import Counter

from coverage import run_coverage_check


def _parse_flags(flags_str: str) -> int:
    f = 0
    if 'IGNORECASE' in (flags_str or ''): f |= re.IGNORECASE
    if 'DOTALL' in (flags_str or ''): f |= re.DOTALL
    if 'MULTILINE' in (flags_str or ''): f |= re.MULTILINE
    return f


def final_audit(all_fs: list[dict], all_subs_by_batch: dict,
                ref_code: str = '', template_code: str = '') -> dict:
    """Run comprehensive audit and return report dict.

    Checks:
    1. False negatives: negative FS matching reference code
    2. False positives: positive FS matching template stubs
    3. Regex quality: compiles without error
    4. TAFFIES FCC: per-criterion, per-student coverage
    5. Scoring weights: distribution summary

    Automatically mitigates any full-weight false negatives to weight 0.5.
    """
    print('\n' + '=' * 60)
    print('FINAL AUDIT')
    print('=' * 60)

    # 1. False negatives
    false_negatives = []
    for fs in all_fs:
        if fs.get('fs_type') != 'negative':
            continue
        rx = fs.get('regex', '')
        if not rx:
            continue
        flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
        try:
            if ref_code and re.search(rx, ref_code, flags):
                false_negatives.append({
                    'id': fs['id'],
                    'weight': fs.get('_scoring_weight', 1.0),
                })
        except re.error:
            pass

    new_fn = [f for f in false_negatives if f['weight'] == 1.0]
    mitigated = [f for f in false_negatives if f['weight'] < 1.0]

    # 2. False positives (positive FS matching template)
    false_positives = 0
    for fs in all_fs:
        if fs.get('fs_type') != 'positive':
            continue
        rx = fs.get('regex', '')
        if not rx:
            continue
        flags = _parse_flags(fs.get('regex_flags', 'IGNORECASE'))
        try:
            if template_code and re.search(rx, template_code, flags):
                false_positives += 1
        except re.error:
            pass

    # 3. Broken regex
    broken = sum(1 for fs in all_fs
                 if fs.get('regex') and not _try_compile(fs))

    # 4. TAFFIES FCC coverage
    BATCH_TASK_MAP = {'q1-': 'Task1', 'q2-': 'Task2', 'q3-': 'Task3'}
    total_pairs = 0
    covered_pairs = 0
    for bp, task in sorted(BATCH_TASK_MAP.items()):
        subs = all_subs_by_batch.get(bp, [])
        if not subs:
            continue
        cov = run_coverage_check(all_fs, subs, task_filter=task)
        for c, info in cov['per_criterion'].items():
            if c.startswith('RQ'):
                total_pairs += info['total']
                covered_pairs += info['covered']

    # 5. Weights
    weights = Counter(fs.get('_scoring_weight', 1.0) for fs in all_fs)

    print(f'  Total FS: {len(all_fs)}')
    print(f'  False negatives: {len(false_negatives)} '
          f'({len(new_fn)} full-weight, {len(mitigated)} mitigated)')
    print(f'  False positives (template): {false_positives}')
    print(f'  Broken regex: {broken}')
    print(f'  FCC coverage: {covered_pairs}/{total_pairs} '
          f'({round(100 * covered_pairs / total_pairs, 1)}%)')
    print(f'  Scoring: 1.0={weights.get(1.0, 0)}, '
          f'0.5={weights.get(0.5, 0)}, 0.0={weights.get(0.0, 0)}')

    # Auto-mitigate new full-weight false negatives
    for fn in new_fn:
        for fs in all_fs:
            if fs.get('id') == fn['id']:
                fs['_warn_ref_match'] = True
                fs['_scoring_weight'] = 0.5
                print(f'  MITIGATED: {fn["id"]} weight 1.0 -> 0.5')

    return {
        'total_fs': len(all_fs),
        'false_negatives': len(false_negatives),
        'false_positives': false_positives,
        'broken_regex': broken,
        'coverage_pct': round(100 * covered_pairs / total_pairs, 1),
        'full_weight': weights.get(1.0, 0),
        'reduced_weight': weights.get(0.5, 0),
        'excluded': weights.get(0.0, 0),
    }


def _try_compile(fs: dict) -> bool:
    """Check if an FS regex compiles."""
    rx = fs.get('regex', '')
    if not rx:
        return False
    try:
        re.compile(rx, _parse_flags(fs.get('regex_flags', 'IGNORECASE')))
        return True
    except re.error:
        return False
