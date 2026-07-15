"""
AST Diff-Based Check Function Synthesis
========================================
Replaces AI refinement for high-FPR check functions with deterministic,
AST-level pattern extraction.

Algorithm (基于 GumTree / Semantic Patch Inference 思想):
  1. 对每个高 FPR 的 FS，收集所有 TP 和 FP 学生的代码
  2. 解析 AST
  3. 提取 TP 共有但 FP 没有的 AST 子结构（错误特征）
  4. 提取 FP 共有但 TP 没有的 AST 子结构（正确特征 = 验证模式）
  5. 生成 check function: 有错误特征 AND 无正确特征 → True
  6. 验证: 必须 TP=all, FP=0
  7. 如果无法找到区分特征 → 标记为 unreliable，降权

Usage:
    python ast_diff_fs.py output/q1_iMusic/fs_registry_taffies.json submission
"""

import ast as _ast
import json, os, re, sys
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ground_truth import load_all_readmes
from ai_pipeline import collect_submissions_by_task


# ============================================================
# AST Node hashing for structure comparison
# ============================================================

def _node_signature(node: _ast.AST) -> str:
    """Generate a structural signature for an AST node.
    Ignores variable names, string values, numbers — only structure matters.
    """
    if isinstance(node, _ast.FunctionDef):
        return f'FunctionDef(name={node.name})'
    if isinstance(node, _ast.Call):
        if isinstance(node.func, _ast.Attribute):
            return f'Call(obj.{node.func.attr})'
        elif isinstance(node.func, _ast.Name):
            return f'Call({node.func.id})'
        return 'Call(?)'
    if isinstance(node, _ast.If):
        return 'If'
    if isinstance(node, _ast.For):
        return 'For'
    if isinstance(node, _ast.Try):
        return 'Try'
    if isinstance(node, _ast.With):
        return 'With'
    if isinstance(node, _ast.Assign):
        return 'Assign'
    if isinstance(node, _ast.Expr):
        return 'Expr'
    if isinstance(node, _ast.Constant):
        val = node.value
        if isinstance(val, str):
            # Normalize SQL/string patterns
            if 'SELECT' in val.upper() or 'INSERT' in val.upper() or 'DELETE' in val.upper():
                return 'Str_SQL'
            if 'flash' in val.lower():
                return 'Str_Flash'
            return 'Str'
        return 'Const'
    if isinstance(node, _ast.Name):
        return 'Name'
    if isinstance(node, _ast.Attribute):
        return 'Attr'
    if isinstance(node, _ast.Import) or isinstance(node, _ast.ImportFrom):
        return 'Import'
    if isinstance(node, _ast.Return):
        return 'Return'
    return type(node).__name__


def _extract_signatures(code: str) -> set[str]:
    """Extract all structural signatures from code."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return set()

    sigs = set()
    for node in _ast.walk(tree):
        sig = _node_signature(node)
        if sig:
            sigs.add(sig)
    return sigs


def _extract_control_paths(code: str, max_depth: int = 3) -> list[str]:
    """Extract control flow paths (sequences of node types).
    More specific than individual signatures.
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []

    paths = []

    class PathExtractor(_ast.NodeVisitor):
        def __init__(self):
            self.current = []

        def _visit_maybe(self, node, name):
            self.current.append(name)
            if len(self.current) <= max_depth:
                paths.append('|'.join(self.current))
            self.generic_visit(node)
            self.current.pop()

        def visit_FunctionDef(self, n): self._visit_maybe(n, 'def')
        def visit_If(self, n): self._visit_maybe(n, 'if')
        def visit_For(self, n): self._visit_maybe(n, 'for')
        def visit_Try(self, n): self._visit_maybe(n, 'try')
        def visit_With(self, n): self._visit_maybe(n, 'with')
        def visit_Call(self, n): self._visit_maybe(n, 'call')
        def visit_Assign(self, n): self._visit_maybe(n, 'assign')
        def visit_Return(self, n): self._visit_maybe(n, 'return')

    PathExtractor().visit(tree)
    return paths


# ============================================================
# String-level pattern extraction (for SQL, flash, etc.)
# ============================================================

def _extract_string_patterns(code: str) -> list[str]:
    """Extract meaningful string-level patterns for differentiation.
    Focuses on SQL patterns, API calls, and error handling patterns.
    """
    patterns = []

    # SQL patterns
    sql_patterns = [
        (r'INSERT\s+OR\s+IGNORE', 'insert_or_ignore'),
        (r'SELECT\s+COUNT\s*\(', 'select_count'),
        (r'SELECT\s+.*FROM\s+\w+', 'select_from'),
        (r'\.execute\s*\(\s*f["\x27]', 'fstring_execute'),
        (r'\.execute\s*\(\s*["\x27][^"\x27]*%[sd]', 'percent_format_execute'),
        (r'\.execute\s*\(\s*["\x27][^"\x27]*\{\}', 'format_execute'),
        (r'\.execute\s*\(\s*["\x27][^"\x27]*["\x27]\s*\+', 'concat_execute'),
        (r'\.execute\s*\(\s*["\x27][^"\x27]*\?[^"\x27]*["\x27]\s*,\s*\(', 'parameterized_execute'),
        (r'ORDER\s+BY\s+\w+\.\w+', 'order_by'),
        (r'\.commit\s*\(', 'commit'),
        (r'\.rollback\s*\(', 'rollback'),
        (r'\.close\s*\(', 'close'),
        (r'\.fetchone\s*\(', 'fetchone'),
        (r'\.fetchall\s*\(', 'fetchall'),
    ]

    for regex, label in sql_patterns:
        if re.search(regex, code, re.IGNORECASE):
            patterns.append(label)

    # API patterns
    api_patterns = [
        (r'flash\s*\(', 'flash'),
        (r'render_template\s*\(', 'render_template'),
        (r'redirect\s*\(', 'redirect'),
        (r'url_for\s*\(', 'url_for'),
        (r'@app\.route', 'app_route'),
        (r'csv\.DictReader', 'csv_dictreader'),
        (r'csv\.reader', 'csv_reader'),
        (r'import\s+pandas', 'pandas'),
        (r'sqlalchemy', 'sqlalchemy'),
    ]

    for regex, label in api_patterns:
        if re.search(regex, code, re.IGNORECASE):
            patterns.append(label)

    return patterns


# ============================================================
# Core diff logic
# ============================================================

def find_distinguishing_features(tp_codes: list[str], fp_codes: list[str]) -> dict:
    """Find features that distinguish TP from FP code.

    Returns:
        {
            'tp_only_signatures': set of AST sigs in ALL TP but NO FP,
            'fp_only_signatures': set of AST sigs in ALL FP but NO TP,
            'tp_only_strings': set of string patterns in ALL TP but NO FP,
            'fp_only_strings': set of string patterns in ALL FP but NO TP,
            'tp_all_paths': paths in ALL TP,
            'fp_all_paths': paths in ALL FP,
            'separable': bool — whether features can cleanly separate,
        }
    """
    if not tp_codes or not fp_codes:
        return {'separable': False}

    # Extract signatures for each code sample
    tp_sigs = [_extract_signatures(c) for c in tp_codes if c.strip()]
    fp_sigs = [_extract_signatures(c) for c in fp_codes if c.strip()]

    tp_strings = [set(_extract_string_patterns(c)) for c in tp_codes if c.strip()]
    fp_strings = [set(_extract_string_patterns(c)) for c in fp_codes if c.strip()]

    tp_paths = [_extract_control_paths(c) for c in tp_codes if c.strip()]
    fp_paths = [_extract_control_paths(c) for c in fp_codes if c.strip()]

    if not tp_sigs or not fp_sigs:
        return {'separable': False}

    # Intersection across ALL TP / ALL FP
    tp_common_sigs = set.intersection(*tp_sigs) if tp_sigs else set()
    fp_common_sigs = set.intersection(*fp_sigs) if fp_sigs else set()

    tp_common_strings = set.intersection(*tp_strings) if tp_strings else set()
    fp_common_strings = set.intersection(*fp_strings) if fp_strings else set()

    tp_common_paths = set.intersection(*[set(p) for p in tp_paths]) if tp_paths else set()
    fp_common_paths = set.intersection(*[set(p) for p in fp_paths]) if fp_paths else set()

    # Unique to TP (error features)
    tp_only_sigs = tp_common_sigs - fp_common_sigs
    tp_only_strings = tp_common_strings - fp_common_strings
    fp_only_sigs = fp_common_sigs - tp_common_sigs
    fp_only_strings = fp_common_strings - tp_common_strings

    # Can we separate?
    has_tp_feature = bool(tp_only_sigs or tp_only_strings)
    has_fp_guard = bool(fp_only_sigs or fp_only_strings)

    return {
        'tp_only_sigs': tp_only_sigs,
        'fp_only_sigs': fp_only_sigs,
        'tp_only_strings': tp_only_strings,
        'fp_only_strings': fp_only_strings,
        'tp_common_paths': tp_common_paths,
        'fp_common_paths': fp_common_paths,
        'separable': has_tp_feature,
        'has_guard': has_fp_guard,
    }


# ============================================================
# Check function synthesis
# ============================================================

def synthesize_check_function(features: dict) -> str:
    """Synthesize a check function from distinguishing features."""

    checks = []

    # TP-only string patterns (error patterns)
    for pat in sorted(features.get('tp_only_strings', [])):
        if pat == 'insert_or_ignore':
            checks.append("    if re.search(r'INSERT\\s+OR\\s+IGNORE', code, re.I): return True")
        elif pat == 'fstring_execute':
            checks.append("    if re.search(r'\\.execute\\s*\\(\\s*f[\"\\x27]', code, re.I): return True")
        elif pat == 'percent_format_execute':
            checks.append("    if re.search(r'\\.execute\\s*\\(\\s*[\"\\x27][^\"\\x27]*%[sd]', code): return True")
        elif pat == 'format_execute':
            checks.append("    if re.search(r'\\.execute\\s*\\(\\s*[\"\\x27][^\"\\x27]*\\{\\}', code): return True")
        elif pat == 'concat_execute':
            checks.append("    if re.search(r'\\.execute\\s*\\(\\s*[\"\\x27][^\"\\x27]*[\"\\x27]\\s*\\+', code): return True")
        elif pat == 'pandas':
            checks.append("    if re.search(r'import\\s+pandas|from\\s+pandas|pd\\.', code): return True")
        elif pat == 'sqlalchemy':
            checks.append("    if re.search(r'import\\s+sqlalchemy|from\\s+sqlalchemy', code): return True")

    # FP-only string patterns (correct features — used as guards)
    guards = []
    for pat in sorted(features.get('fp_only_strings', [])):
        if pat == 'select_count':
            guards.append("    has_select_count = bool(re.search(r'SELECT\\s+COUNT\\s*\\(', code, re.I))")
        elif pat == 'parameterized_execute':
            guards.append("    has_parameterized = bool(re.search(r'\\.execute\\s*\\(\\s*[\"\\x27][^\"\\x27]*\\?[^\"\\x27]*[\"\\x27]\\s*,\\s*\\(', code, re.I))")
        elif pat == 'flash':
            guards.append("    has_flash = bool(re.search(r'flash\\s*\\(', code))")
        elif pat == 'commit':
            guards.append("    has_commit = bool(re.search(r'\\.commit\\s*\\(', code))")
        elif pat == 'close':
            guards.append("    has_close = bool(re.search(r'\\.close\\s*\\(', code))")

    if not checks:
        return ''

    # Build function
    lines = ['def check(code: str) -> bool:', '    import re']

    for guard in guards:
        lines.append(guard)

    lines.append('')
    for check in checks:
        lines.append(check)

    # If we have guards, combine them
    if guards:
        guard_names = [g.split(' = ')[0].strip().replace('    ', '') for g in guards]
        guard_condition = ' and '.join(f'not {g}' for g in guard_names)
        lines.append(f'    # Guard: correct code should have these patterns')
        lines.append(f'    if {guard_condition}:')
        lines.append(f'        return False  # Has the guard pattern -> correct code')

    lines.append('    return False')

    return '\n'.join(lines)


# ============================================================
# Main refinement function
# ============================================================

def refine_with_ast_diff(fs_registry_path: str, submission_dir: str) -> list[dict]:
    """Refine all high-FPR FS using AST diff analysis.
    Replaces Round 5's AI-based iteration.
    """

    print('=' * 60)
    print('  AST DIFF FS REFINEMENT')
    print('  (deterministic rule synthesis from TP/FP code)')
    print('=' * 60)

    with open(fs_registry_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fs_list = data['fs_registry']

    all_readmes = load_all_readmes(submission_dir)
    task_subs = {}
    for tn in [1, 2, 3]:
        task_subs[tn] = collect_submissions_by_task(submission_dir, tn)

    # Build code lookup
    code_lookup = {}
    for tn in [1, 2, 3]:
        for s in task_subs[tn]:
            code_lookup[(s['student'], tn)] = s.get('code', '')

    refined = 0
    kept_perfect = 0
    skipped = 0

    for fs in fs_list:
        if fs.get('fs_type') != 'negative':
            continue
        if fs.get('_scoring_weight', 1.0) == 0.0:
            continue

        criterion = fs.get('criterion', '?')
        task_str = fs.get('task', '')
        tn = int(task_str.replace('Task', '')) if task_str.startswith('Task') else 0
        name = fs.get('name', '?')[:50]
        fid = fs.get('id', fs.get('name', '?')[:30])

        # Compile existing check function for validation
        sig_type = fs.get('signature_type', 'regex')
        old_fn = None
        if sig_type == 'check_function':
            fn_body = fs.get('check_function', '')
            try:
                _ast.parse(fn_body.strip())
                local_ns = {}
                exec(fn_body.strip(), {'__builtins__': __builtins__}, local_ns)
                old_fn = local_ns.get('check')
            except Exception:
                pass

        if not callable(old_fn):
            continue

        # Collect TP and FP codes
        tp_codes = []
        fp_codes = []
        tp_count = fp_count = 0

        for s in task_subs.get(tn, []):
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
            has_bad = bool(gt.get(criterion, {}).get('bad', []) or gt.get(criterion, {}).get('mistake', []))

            try:
                result = old_fn(code)
                matched = result[0] if isinstance(result, tuple) and len(result) == 2 else bool(result)
            except Exception:
                continue

            if matched and has_bad:
                tp_codes.append(code)
                tp_count += 1
            elif matched and not has_bad:
                fp_codes.append(code)
                fp_count += 1

        if fp_count == 0:
            kept_perfect += 1
            print(f'\n  [{criterion}] {name[:50]}')
            print(f'    Already FPR=0% (TP={tp_count}) — keeping')
            continue

        total = tp_count + fp_count
        fpr = fp_count / total if total > 0 else 0
        print(f'\n  [{criterion}] {name[:50]}')
        print(f'    TP={tp_count}, FP={fp_count}, FPR={fpr:.0%}')

        # AST diff analysis
        features = find_distinguishing_features(tp_codes, fp_codes)

        print(f'    TP-only sigs: {features.get("tp_only_sigs", set())}')
        print(f'    FP-only sigs: {features.get("fp_only_sigs", set())}')
        print(f'    TP-only strings: {features.get("tp_only_strings", set())}')
        print(f'    FP-only strings: {features.get("fp_only_strings", set())}')

        if not features.get('separable'):
            print(f'    Cannot separate — no TP-unique features found. Keeping original.')
            skipped += 1
            continue

        # Synthesize check function
        new_fn_source = synthesize_check_function(features)
        if not new_fn_source:
            print(f'    Could not synthesize check function.')
            skipped += 1
            continue

        # Validate
        try:
            _ast.parse(new_fn_source)
            local_ns = {}
            exec(new_fn_source, {'__builtins__': __builtins__}, local_ns)
            new_fn = local_ns.get('check')
        except Exception as e:
            print(f'    Synthesized function error: {e}')
            skipped += 1
            continue

        # Test on ALL students
        new_tp = new_fp = 0
        for s in task_subs.get(tn, []):
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
            has_bad = bool(gt.get(criterion, {}).get('bad', []) or gt.get(criterion, {}).get('mistake', []))

            try:
                result = new_fn(code)
                if result[0] if isinstance(result, tuple) and len(result) == 2 else result:
                    if has_bad:
                        new_tp += 1
                    else:
                        new_fp += 1
            except Exception:
                pass

        if new_fp == 0 and new_tp > 0:
            # Perfect!
            fs['check_function'] = new_fn_source
            fs['_ast_diff_refined'] = True
            fs['_ast_diff_original_fp'] = fp_count
            fp_changed = fp_count
            refined += 1
            print(f'    [OK] PERFECT: FP {fp_count}->0, TP {tp_count}->{new_tp}')
        elif new_fp < fp_count and new_tp >= tp_count * 0.7:
            fs['check_function'] = new_fn_source
            fs['_ast_diff_improved'] = True
            fs['_ast_diff_original_fp'] = fp_count
            refined += 1
            print(f'    [OK] IMPROVED: FP {fp_count}->{new_fp}, TP {tp_count}->{new_tp}')
        else:
            print(f'    [FAIL] FP {fp_count}->{new_fp}, TP {tp_count}->{new_tp} — keeping original')
            skipped += 1

    # Summary
    print(f'\n{"=" * 60}')
    print(f'  AST DIFF SUMMARY')
    print(f'  Already perfect (kept): {kept_perfect}')
    print(f'  Refined: {refined}')
    print(f'  Skipped (could not improve): {skipped}')
    print('=' * 60)

    # Save
    out_path = fs_registry_path.replace('.json', '_ast.json')
    data['fs_registry'] = fs_list
    data['total_fs'] = len(fs_list)
    data['pipeline'] = data.get('pipeline', '') + ' + AST Diff'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'  Output: {out_path}')

    return fs_list


if __name__ == '__main__':
    registry = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(BASE, 'output', 'q1_iMusic', 'fs_registry_taffies.json')
    sub_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(BASE, 'submission')
    refine_with_ast_diff(registry, sub_dir)
