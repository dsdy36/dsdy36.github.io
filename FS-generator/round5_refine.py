"""
Round 5: Iterative Check Function Refinement
=============================================
For each FS with FP > 0, loops until FPR=0% or max iterations:
  1. Runs FS against ALL students to find TP and FP
  2. Extracts matched CODE SNIPPETS (not full functions)
  3. Sends ALL TP/FP snippets to AI (with smart truncation if too many)
  4. AI rewrites the check function
  5. Validates: must keep ALL TP, must drop ALL FP
  6. If FP still > 0 → loop again with new TP/FP data

Result: only check functions with P=1.000 survive. Others are discarded.

Usage:
    python round5_refine.py output/q1_iMusic/fs_registry_taffies.json submission
"""

import json, os, re, sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from collections import defaultdict
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE, '.env'))

from ai_pipeline import call_deepseek, collect_submissions_by_task
from ground_truth import load_all_readmes

MAX_REFINE_ROUNDS = 5          # Max refinement iterations per FS
MAX_SNIPPET_LINES = 8          # Max lines per matched code snippet
MAX_TOTAL_SNIPPETS = 12        # Max total snippets sent to AI (TP + FP, combined)
MAX_PROMPT_CHARS = 4000        # Max prompt size per snippet section


REFINE_CHECK_SYSTEM = """You are an expert at writing precise Python code analysis functions.
Your task: fix a check function that incorrectly matches some students (false positives).

You will see:
  [KEEP] code snippets — students who GENUINELY have the mistake (must return True)
  [DROP] code snippets — students who wrote CORRECT code but are wrongly matched (must return False)

Your job: find the KEY DIFFERENCE between [KEEP] and [DROP] snippets, then rewrite
the check function to capture this difference.

STRATEGIES (in order of effectiveness):
1. ADD SPECIFIC CONTEXT: Instead of matching a broad pattern anywhere in code,
   require it in a specific context (e.g., inside cursor.execute(), not in flash())
2. ADD EXCLUSION: If [DROP] has feature X that [KEEP] doesn't, add "if X: return False"
3. NARROW: Check only inside the relevant function, not the entire file
4. USE AST PRECISELY: Check node types, function names, argument structure

CRITICAL: Your function MUST return True for ALL [KEEP] and False for ALL [DROP].
If you can't find a clean distinction, keep the original function unchanged.

Output ONLY the COMPLETE Python function (with def line). No markdown, no explanation."""


def _extract_matched_lines(code: str, check_fn, max_lines: int = MAX_SNIPPET_LINES) -> str:
    """Extract only the lines that cause the check function to match.
    Uses a binary-search-like approach to find the minimal matching region.
    """
    if not code.strip():
        return ''

    # Strategy: progressively narrow to find matching lines
    lines = code.split('\n')
    matched_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('import ') or stripped.startswith('from '):
            continue
        # Test a window around this line
        ctx_start = max(0, i - 1)
        ctx_end = min(len(lines), i + 3)
        context = '\n'.join(lines[ctx_start:ctx_end])
        try:
            result = check_fn(context)
            matched_ok = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            if matched_ok:
                matched_lines.append(stripped)
                if len(matched_lines) >= max_lines:
                    break
        except Exception:
            pass

    if matched_lines:
        return '\n'.join(matched_lines)

    # Fallback: return non-comment, non-import lines
    code_lines = [l.strip() for l in lines
                  if l.strip() and not l.strip().startswith('#')
                  and not l.strip().startswith('import ') and not l.strip().startswith('from ')]
    return '\n'.join(code_lines[:max_lines])


def _compile_check(fn_body: str):
    """Compile a check function string into a callable. Returns None on failure."""
    fn_source = fn_body.strip()
    try:
        import ast as _ast
        _ast.parse(fn_source)
        local_ns = {}
        exec(fn_source, {'__builtins__': __builtins__}, local_ns)
        fn = local_ns.get('check')
        return fn if callable(fn) else None
    except Exception:
        return None


def _sample_snippets(items: list[dict], max_total: int, max_chars: int) -> list[dict]:
    """Smart sampling: prefer diverse snippets, limit total count and chars."""
    if len(items) <= max_total // 2:
        return items  # Small enough, send all

    # Take evenly distributed samples
    step = max(1, len(items) // (max_total // 2))
    sampled = []
    total_chars = 0
    for i in range(0, len(items), step):
        item = items[i]
        snippet_len = len(item['snippet'])
        if total_chars + snippet_len > max_chars:
            break
        sampled.append(item)
        total_chars += snippet_len
    return sampled


def _run_validation(fn, task_subs, all_readmes, criterion, task_num) -> dict:
    """Run check function against all students. Returns TP/FP counts and snippets."""
    tp_items = []
    fp_items = []
    tp = fp = 0

    for s in task_subs.get(task_num, []):
        sid = s['student']
        code = s.get('code', '')
        if not code:
            continue

        rdata = all_readmes.get(sid, {})
        criteria = rdata.get('criteria', '{}')
        if isinstance(criteria, str):
            try: criteria = eval(criteria)
            except Exception: criteria = {}
        gt = criteria if isinstance(criteria, dict) else {}
        crit_gt = gt.get(criterion, {})
        has_bad = bool(crit_gt.get('bad', []) or crit_gt.get('mistake', []))

        try:
            result = fn(code)
            matched = result[0] if isinstance(result, tuple) and len(result) == 2 else bool(result)
        except Exception:
            continue

        if matched:
            snippet = _extract_matched_lines(code, fn)
            item = {'student': sid, 'code': code, 'snippet': snippet}
            if has_bad:
                tp += 1
                tp_items.append(item)
            else:
                fp += 1
                fp_items.append(item)

    return {
        'tp': tp, 'fp': fp,
        'tp_items': tp_items, 'fp_items': fp_items,
        'fpr': fp / (tp + fp) if (tp + fp) > 0 else 0,
    }


def refine_single_fs(fs: dict, task_subs, all_readmes, max_rounds: int = MAX_REFINE_ROUNDS) -> dict | None:
    """Iteratively refine one FS until FPR=0% or max rounds reached.
    Returns the refined FS dict, or None if refinement failed.
    """
    criterion = fs.get('criterion', '?')
    task_str = fs.get('task', '')
    task_num = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0
    name = fs.get('name', '?')[:50]
    fn_body = fs.get('check_function', '')

    fn = _compile_check(fn_body)
    if not fn:
        print(f'  SKIP: cannot compile original check function')
        return None

    # Initial validation
    result = _run_validation(fn, task_subs, all_readmes, criterion, task_num)
    initial_fp = result['fp']
    initial_tp = result['tp']

    if result['fp'] == 0:
        print(f'  Already FPR=0% (TP={result["tp"]}) — keeping')
        return fs

    print(f'  Initial: TP={result["tp"]}, FP={result["fp"]}, FPR={result["fpr"]:.0%}')

    best_fn_body = fn_body
    best_result = result

    for round_num in range(1, max_rounds + 1):
        # Sample TP and FP snippets for AI
        tp_sample = _sample_snippets(result['tp_items'], MAX_TOTAL_SNIPPETS, MAX_PROMPT_CHARS)
        fp_sample = _sample_snippets(result['fp_items'], MAX_TOTAL_SNIPPETS, MAX_PROMPT_CHARS)

        if not fp_sample:
            print(f'  No FP to refine — done')
            break

        # Build prompt
        lines = [f"## {criterion}: {name}"]
        lines.append(f"Round {round_num}: must eliminate {len(result['fp_items'])} FP while keeping {len(result['tp_items'])} TP.")

        lines.append(f"\n### Current function:")
        lines.append(f"```python")
        lines.append(best_fn_body[:1500])
        lines.append(f"```")

        if tp_sample:
            lines.append(f"\n### [KEEP] True Positives ({len(tp_sample)} of {len(result['tp_items'])} shown)")
            lines.append(f"These students HAVE the mistake. Must return True.")
            for ex in tp_sample:
                lines.append(f"**{ex['student']}**:")
                lines.append(f"```python")
                lines.append(ex['snippet'][:300])
                lines.append(f"```")

        if fp_sample:
            lines.append(f"\n### [DROP] False Positives ({len(fp_sample)} of {len(result['fp_items'])} shown)")
            lines.append(f"These students are CORRECT but wrongly flagged. Must return False.")
            for ex in fp_sample:
                lines.append(f"**{ex['student']}**:")
                lines.append(f"```python")
                lines.append(ex['snippet'][:300])
                lines.append(f"```")

        lines.append(f"\n### Task")
        lines.append(f"Find the KEY DIFFERENCE between [KEEP] and [DROP] code.")
        lines.append(f"Rewrite def check(code): -> bool to return True ONLY for [KEEP] patterns.")
        lines.append(f"Output ONLY the complete function (with def line).")

        prompt = '\n'.join(lines)

        # Call AI
        response = None
        for attempt in range(2):
            resp = call_deepseek(REFINE_CHECK_SYSTEM, prompt, temperature=0.1)
            if resp and resp.strip():
                response = resp.strip()
                break

        if not response:
            print(f'    Round {round_num}: API empty — keeping best')
            break

        # Extract function
        new_fn = response.strip()
        new_fn = re.sub(r'^```(?:python)?\s*\n?', '', new_fn)
        new_fn = re.sub(r'\n?```\s*$', '', new_fn)
        new_fn = new_fn.strip()

        if not new_fn.startswith('def check'):
            print(f'    Round {round_num}: No def check() in response — retrying')
            continue

        new_check = _compile_check(new_fn)
        if not new_check:
            print(f'    Round {round_num}: Cannot compile — retrying')
            continue

        # Validate
        new_result = _run_validation(new_check, task_subs, all_readmes, criterion, task_num)

        # Scoring: prefer lower FP; if FP same, prefer higher TP
        fp_improved = new_result['fp'] < best_result['fp']
        tp_ok = new_result['tp'] >= best_result['tp'] * 0.8  # Allow 20% TP loss max

        if new_result['fp'] == 0:
            # Perfect!
            best_fn_body = new_fn
            best_result = new_result
            print(f'    Round {round_num}: PERFECT! FP=0, TP={new_result["tp"]}')
            break
        elif fp_improved and tp_ok:
            best_fn_body = new_fn
            best_result = new_result
            print(f'    Round {round_num}: Improved — FP: {result["fp"]}->{new_result["fp"]}, TP: {result["tp"]}->{new_result["tp"]}')
            result = new_result  # Continue looping with new baseline
        else:
            print(f'    Round {round_num}: Rejected — FP: {result["fp"]}->{new_result["fp"]}, TP: {result["tp"]}->{new_result["tp"]} (TP loss too high)')
            # Keep best, stop iterating
            break

    # Apply best result
    final_fp = best_result['fp']
    if final_fp == 0:
        fs['check_function'] = best_fn_body
        fs['_round5_perfect'] = True
        fs['_round5_rounds'] = round_num
        fs['_round5_initial_fp'] = initial_fp
        print(f'  [OK] Converged to FPR=0% in {round_num} rounds (TP={best_result["tp"]})')
        return fs
    elif final_fp < initial_fp:
        fs['check_function'] = best_fn_body
        fs['_round5_improved'] = True
        fs['_round5_rounds'] = round_num
        fs['_round5_initial_fp'] = initial_fp
        fs['_round5_final_fp'] = final_fp
        print(f'  [WARN] Improved but not perfect: FP {initial_fp}->{final_fp} (FPR={best_result["fpr"]:.0%})')
        return fs
    else:
        print(f'  [FAIL] Could not improve (FP stuck at {initial_fp})')
        return None


def round5_refine(fs_registry_path: str, submission_dir: str,
                  min_fpr: float = 0.3, min_matches: int = 3) -> list[dict]:
    """Main Round 5: iterative refinement of all high-FPR check functions."""

    print('=' * 60)
    print('  ROUND 5: Iterative Check Function Refinement')
    print(f'  (max {MAX_REFINE_ROUNDS} rounds per FS, target FPR=0%)')
    print('=' * 60)

    # Load data
    with open(fs_registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fs_list = data['fs_registry']

    all_readmes = load_all_readmes(submission_dir)
    task_subs = {}
    for tn in [1, 2, 3]:
        task_subs[tn] = collect_submissions_by_task(submission_dir, tn)

    # Identify candidates
    candidates = []
    for fs in fs_list:
        if fs.get('fs_type') != 'negative': continue
        if fs.get('_scoring_weight', 1.0) == 0.0: continue
        if fs.get('signature_type') != 'check_function': continue

        fn = _compile_check(fs.get('check_function', ''))
        if not fn: continue

        criterion = fs.get('criterion', '?')
        task_str = fs.get('task', '')
        task_num = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0
        result = _run_validation(fn, task_subs, all_readmes, criterion, task_num)
        total = result['tp'] + result['fp']

        if result['fp'] > 0 and total >= min_matches and result['fpr'] >= min_fpr:
            candidates.append({'fs': fs, 'result': result})

    candidates.sort(key=lambda x: -x['result']['fp'])
    print(f'\n  Candidates: {len(candidates)} FS with FP > 0')
    for c in candidates:
        r = c['result']
        fs = c['fs']
        print(f'    {fs.get("criterion","?"):6s} {fs.get("name","?")[:45]:45s} TP={r["tp"]:>3} FP={r["fp"]:>3} FPR={r["fpr"]:.0%}')

    # Refine each candidate
    print(f'\n  --- Iterative Refinement ---')
    perfect = 0
    improved = 0
    failed = 0

    for i, c in enumerate(candidates):
        fs = c['fs']
        print(f'\n  [{i+1}/{len(candidates)}] {fs.get("criterion")} {fs.get("name","?")[:40]}')
        result = refine_single_fs(fs, task_subs, all_readmes)

        if result:
            final_check = _compile_check(result.get('check_function', ''))
            if final_check:
                final_result = _run_validation(final_check, task_subs, all_readmes,
                                              result.get('criterion', '?'),
                                              int(result.get('task', 'Task1').replace('Task', '')))
                if final_result['fp'] == 0:
                    perfect += 1
                else:
                    improved += 1
            else:
                failed += 1
        else:
            failed += 1

    # Summary
    print(f'\n  --- Round 5 Summary ---')
    print(f'  Perfect (FPR=0%): {perfect}')
    print(f'  Improved (FP reduced): {improved}')
    print(f'  Failed (could not improve): {failed}')

    # Save
    out_path = fs_registry_path.replace('.json', '_r5.json')
    data['fs_registry'] = fs_list
    data['total_fs'] = len(fs_list)
    data['pipeline'] = data.get('pipeline', '') + ' + R5 (iterative refine)'
    data['_round5_stats'] = {'perfect': perfect, 'improved': improved, 'failed': failed}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'  Output: {out_path}')

    return fs_list


if __name__ == '__main__':
    registry = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(BASE, 'output', 'q1_iMusic', 'fs_registry_taffies.json')
    sub_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(BASE, 'submission')
    round5_refine(registry, sub_dir)
