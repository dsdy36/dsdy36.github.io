"""
Blind Test Framework
=====================
Compares FS matching results against README.md ground truth to compute
precision, recall, and F1 for each criterion.

Ground truth (README.md) records exactly which good/bad pattern variants
were injected into each student's code by the CW-generator.

The blind test answers:
  - Are negative FS matching the RIGHT students? (precision)
  - Are negative FS catching ALL bad patterns? (recall)
  - Which specific FS have false positives?

Usage:
    python blind_test.py <fs_registry_path> <submissions_dir>
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from coverage import run_coverage_check, _parse_flags
from ground_truth import load_all_readmes, parse_readme


def run_blind_test(fs_path: str, submissions_dir: str) -> dict:
    """Run full blind test.

    Returns report dict with per-criterion and overall metrics.
    """
    # ── Load FS registry ──
    with open(fs_path, 'r', encoding='utf-8') as f:
        fs_data = json.load(f)
    fs_list = fs_data.get('fs_registry', [])

    # ── Load READMEs ──
    all_readmes = load_all_readmes(submissions_dir)

    # ── Collect submissions ──
    sub_lookup: dict[str, dict[int, str]] = {}  # sid -> {task_num: code}
    for sid_dir in Path(submissions_dir).iterdir():
        if not sid_dir.is_dir() or sid_dir.name.startswith('_'):
            continue
        sid = sid_dir.name
        sub_lookup[sid] = {}
        for tn in [1, 2, 3]:
            tf = sid_dir / f'task{tn}.py'
            if tf.exists():
                sub_lookup[sid][tn] = tf.read_text(encoding='utf-8', errors='ignore')

    # ── Compile FS by criterion ──
    pos_fs: dict[str, list[dict]] = defaultdict(list)
    neg_fs: dict[str, list[dict]] = defaultdict(list)
    for f in fs_list:
        crit = f.get('criterion', '?')
        if f.get('_scoring_weight', 1.0) == 0.0:
            continue

        # Support both check_function and regex signatures
        sig_type = f.get('signature_type', 'regex')
        if sig_type == 'check_function':
            fn_body = f.get('check_function', '')
            if not fn_body:
                continue
            fn_source = fn_body.strip()
            try:
                import ast as _ast
                _ast.parse(fn_source)
                local_ns = {}
                exec(fn_source, {'__builtins__': __builtins__}, local_ns)
                f['_check_fn'] = local_ns.get('check')
                if not callable(f['_check_fn']):
                    continue
            except Exception:
                continue
        else:
            regex = f.get('regex', '')
            if not regex:
                continue
            try:
                flags = _parse_flags(f.get('regex_flags', 'IGNORECASE'))
                f['_compiled'] = re.compile(regex, flags)
            except re.error:
                continue

        if f.get('fs_type') == 'positive':
            pos_fs[crit].append(f)
        else:
            neg_fs[crit].append(f)

    def _fs_matches(fs: dict, code: str) -> bool:
        """Check if an FS matches student code. Supports both regex and check_function.
        Handles both (bool, str) tuple returns and legacy bool returns."""
        try:
            if fs.get('_check_fn'):
                result = fs['_check_fn'](code)
                if isinstance(result, tuple) and len(result) == 2:
                    return bool(result[0])
                return bool(result)
            compiled = fs.get('_compiled')
            if compiled:
                return bool(compiled.search(code))
        except Exception:
            pass
        return False

    all_criteria = sorted(set(list(pos_fs.keys()) + list(neg_fs.keys())))

    # ── Per-criterion metrics ──
    criterion_metrics = {}
    fs_fpr: dict[str, dict] = {}  # per-FS false positive rate

    for crit in all_criteria:
        m = re.search(r'(\d)', crit)
        task_num = int(m.group(1)) if m else 1

        tp, fp, fn, tn = 0, 0, 0, 0

        for sid, tasks in sub_lookup.items():
            code = tasks.get(task_num, '')
            if not code:
                continue

            # Ground truth for this criterion
            rdata = all_readmes.get(sid, {})
            criteria_raw = rdata.get('criteria', '{}')
            if isinstance(criteria_raw, str):
                try:
                    criteria_raw = eval(criteria_raw)
                except Exception:
                    criteria_raw = {}
            student_criteria = criteria_raw if isinstance(criteria_raw, dict) else {}

            crit_data = student_criteria.get(crit, {})
            has_bad = bool(crit_data.get('bad', []))
            has_good = bool(crit_data.get('good', []))

            # Check negative FS matches
            neg_matched = False
            matched_neg_ids = []
            for nf in neg_fs[crit]:
                if _fs_matches(nf, code):
                    neg_matched = True
                    matched_neg_ids.append(nf.get('id', '?'))

            # Check positive FS matches
            pos_matched = any(_fs_matches(pf, code) for pf in pos_fs[crit])

            # ── Count TP/FP/FN/TN for negative FS ──
            if has_bad and neg_matched:
                tp += 1
            elif has_bad and not neg_matched:
                fn += 1
            elif not has_bad and neg_matched:
                fp += 1
                # Record which FS caused the FP
                for nid in matched_neg_ids:
                    if nid not in fs_fpr:
                        fs_fpr[nid] = {'fp_students': [], 'total_matches': 0}
                    fs_fpr[nid]['fp_students'].append(sid)
            elif not has_bad and not neg_matched:
                tn += 1

            # Also track per-FS match counts
            for nf in neg_fs[crit]:
                nid = nf.get('id', '?')
                if _fs_matches(nf, code):
                    if nid not in fs_fpr:
                        fs_fpr[nid] = {'fp_students': [], 'total_matches': 0}
                    fs_fpr[nid]['total_matches'] += 1

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        criterion_metrics[crit] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'total_students': tp + fp + fn + tn,
            'neg_fs_count': len(neg_fs[crit]),
            'pos_fs_count': len(pos_fs[crit]),
        }

    # ── Overall metrics ──
    total_tp = sum(m['tp'] for m in criterion_metrics.values())
    total_fp = sum(m['fp'] for m in criterion_metrics.values())
    total_fn = sum(m['fn'] for m in criterion_metrics.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    # ── Per-FS false positive ranking ──
    fs_fp_ranked = []
    for fs_id, data in fs_fpr.items():
        fp_count = len(data['fp_students'])
        total = data['total_matches']
        fpr_val = fp_count / total if total > 0 else 0
        if fp_count > 0:
            fs_fp_ranked.append({
                'fs_id': fs_id,
                'fp_count': fp_count,
                'total_matches': total,
                'fpr': round(fpr_val, 3),
                'fp_students': data['fp_students'][:5],
            })
    fs_fp_ranked.sort(key=lambda x: -x['fp_count'])

    return {
        'criterion_metrics': criterion_metrics,
        'overall': {
            'precision': round(overall_p, 3),
            'recall': round(overall_r, 3),
            'f1': round(overall_f1, 3),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
        },
        'top_false_positives': fs_fp_ranked[:10],
        'total_negative_fs': sum(len(v) for v in neg_fs.values()),
        'total_students': len(sub_lookup),
    }


def print_report(report: dict):
    """Human-readable blind test report."""
    print(f'\n{"=" * 70}')
    print('  BLIND TEST REPORT — Negative FS vs Ground Truth')
    print('=' * 70)
    print(f'  Students: {report["total_students"]}')
    print(f'  Negative FS: {report["total_negative_fs"]}')
    print()

    # Overall
    o = report['overall']
    print(f'  OVERALL:  P={o["precision"]:.3f}  R={o["recall"]:.3f}  F1={o["f1"]:.3f}')
    print(f'  TP={o["total_tp"]}  FP={o["total_fp"]}  FN={o["total_fn"]}')
    print()

    # Per-criterion
    print(f'  {"Criterion":<8} {"P":>6} {"R":>6} {"F1":>6} {"TP":>4} {"FP":>4} {"FN":>4} {"NegFS":>5}')
    print(f'  {"-"*8} {"-"*6} {"-"*6} {"-"*6} {"-"*4} {"-"*4} {"-"*4} {"-"*5}')
    for crit, m in sorted(report['criterion_metrics'].items()):
        bar = _bar(m['f1'])
        print(f'  {crit:<8} {m["precision"]:>6.3f} {m["recall"]:>6.3f} '
              f'{m["f1"]:>6.3f} {m["tp"]:>4} {m["fp"]:>4} {m["fn"]:>4} '
              f'{m["neg_fs_count"]:>5} {bar}')

    # Weak criteria (F1 < 0.8)
    weak = [(c, m) for c, m in report['criterion_metrics'].items() if m['f1'] < 0.8]
    if weak:
        print(f'\n  WEAK CRITERIA (F1 < 0.8):')
        for c, m in sorted(weak, key=lambda x: x[1]['f1']):
            issue = 'low recall (missing bad patterns)' if m['recall'] < m['precision'] else 'low precision (false positives)'
            print(f'    {c}: F1={m["f1"]:.3f} — {issue}')

    # Top false positive FS
    top = report.get('top_false_positives', [])
    if top:
        print(f'\n  Top false positive FS (matching students without bad pattern):')
        for item in top[:8]:
            print(f'    {item["fs_id"]}: {item["fp_count"]}/{item["total_matches"]} '
                  f'FPs (FPR={item["fpr"]:.2f}) '
                  f'on {", ".join(item["fp_students"][:3])}')

    print()


def _bar(f1: float) -> str:
    n = int(f1 * 20)
    return '[' + '#' * n + '-' * (20 - n) + ']'


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        fs_path = sys.argv[1]
        sub_dir = sys.argv[2]
    else:
        fs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'output', 'q1_iMusic', 'fs_registry.json')
        sub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission')

    report = run_blind_test(fs_path, sub_dir)
    print_report(report)
