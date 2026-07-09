"""Analyze false positive distribution across negative FS."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from ground_truth import load_all_readmes
from ai_pipeline import collect_submissions_by_task

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, 'output', 'q1_iMusic', 'fs_registry_taffies.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
fs_list = data['fs_registry']

all_readmes = load_all_readmes(os.path.join(BASE, 'submission'))
task_subs = {}
for tn in [1, 2, 3]:
    task_subs[tn] = collect_submissions_by_task(os.path.join(BASE, 'submission'), tn)

per_fs = {}
for fs in fs_list:
    if fs.get('fs_type') != 'negative':
        continue
    if fs.get('_scoring_weight', 1.0) == 0.0:
        continue
    fid = fs.get('id', fs.get('name', '?')[:30])
    crit = fs.get('criterion', '?')
    task_str = fs.get('task', '')
    tn = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0

    # Compile check function
    sig_type = fs.get('signature_type', 'regex')
    check_fn = None
    if sig_type == 'check_function':
        fn_body = fs.get('check_function', '')
        fn_source = fn_body.strip()
        try:
            import ast as _ast
            _ast.parse(fn_source)
            local_ns = {}
            exec(fn_source, {'__builtins__': __builtins__}, local_ns)
            check_fn = local_ns.get('check')
        except Exception:
            pass

    if not callable(check_fn):
        continue

    tp = 0
    fp = 0
    fn_count = 0
    tp_students = []
    fp_students = []

    for s in task_subs.get(tn, []):
        sid = s['student']
        code = s.get('code', '')
        if not code:
            continue

        rdata = all_readmes.get(sid, {})
        criteria = rdata.get('criteria', '{}')
        if isinstance(criteria, str):
            try:
                criteria = eval(criteria)
            except Exception:
                criteria = {}
        gt = criteria if isinstance(criteria, dict) else {}
        crit_gt = gt.get(crit, {})
        has_bad = bool(crit_gt.get('bad', []) or crit_gt.get('mistake', []))

        matched = False
        try:
            result = check_fn(code)
            matched = result[0] if isinstance(result, tuple) and len(result) == 2 else bool(result)
        except Exception:
            pass

        if matched and has_bad:
            tp += 1
            tp_students.append(sid)
        elif matched and not has_bad:
            fp += 1
            fp_students.append(sid)
        elif not matched and has_bad:
            fn_count += 1

    per_fs[fid] = {
        'criterion': crit, 'name': fs.get('name', '?'),
        'tp': tp, 'fp': fp, 'fn': fn_count,
        'fpr': fp / (tp + fp) if (tp + fp) > 0 else 0,
        'tp_students': tp_students[:5], 'fp_students': fp_students[:5],
    }

# Sort by FP descending
print('=== Negative FS sorted by FP count ===')
for fid, info in sorted(per_fs.items(), key=lambda x: -x[1]['fp']):
    total = info['tp'] + info['fp']
    fpr = info['fpr']
    bar = '#' * int(fpr * 30) + '-' * max(1, 30 - int(fpr * 30))
    print(f'{info["criterion"]:6s} {fid[:45]:45s} TP={info["tp"]:>3} FP={info["fp"]:>3} FN={info["fn"]:>2} FPR={fpr:.0%} [{bar}]')

print(f'\nTotal: {sum(i["tp"] for i in per_fs.values())} TP, {sum(i["fp"] for i in per_fs.values())} FP, {sum(i["fn"] for i in per_fs.values())} FN')

# FP concentration
fps_sorted = sorted([(fid, info['fp']) for fid, info in per_fs.items()], key=lambda x: -x[1])
total_fp = sum(info['fp'] for _, info in per_fs.items())
cum = 0
print(f'\nFP concentration (total FP={total_fp}):')
for i, (fid, fp) in enumerate(fps_sorted[:10]):
    cum += fp
    print(f'  Top {i+1}: {fid[:45]} — {fp:>3} FP (cumulative {cum/total_fp:.0%})')
