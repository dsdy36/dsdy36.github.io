"""
Audit all generated FS check functions for scoring rigor issues.
Tests: function scoping, template matching, reference matching,
positive/negative correctness, stub matching.
"""
import json, re, ast, sys, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))

# Load FS registry and reference/template code
with open(os.path.join(BASE, 'output', 'q1_iMusic', 'fs_registry_taffies.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
fs_list = data['fs_registry']

# Load reference code
ref_code = ''
ref_dir = os.path.join(BASE, 'references', 'q1_iMusic')
if os.path.isdir(ref_dir):
    for root, _, files in os.walk(ref_dir):
        for fn in files:
            if fn.endswith('.py'):
                with open(os.path.join(root, fn), 'r', encoding='utf-8') as f:
                    ref_code += f.read() + '\n'

# Load template code
template_code = ''
code_dir = os.path.join(BASE, 'question', 'code')
if os.path.isdir(code_dir):
    for fn in os.listdir(code_dir):
        if fn.endswith('.py'):
            with open(os.path.join(code_dir, fn), 'r', encoding='utf-8') as f:
                template_code += f.read() + '\n'

# Template-only functions (never student-written)
TEMPLATE_FUNCS = {'upload_route', 'index', 'page_not_found', 'main'}

# Student-written functions per task
TASK_FUNCS = {
    1: ['update_playlist_tracks'],
    2: ['get_all_genres', 'get_statistics'],
    3: ['get_all_playlists', 'create_playlist', 'rename_playlist',
        'delete_playlist', 'add_tracks_by_genre', 'remove_tracks_by_genre'],
}

# Criterion → function mapping
CRIT_FUNC = {
    'RQ1_1': 'update_playlist_tracks', 'RQ1_2': 'update_playlist_tracks',
    'RQ1_3': 'update_playlist_tracks', 'RQ1_4': 'update_playlist_tracks',
    'RQ2_1': 'get_all_genres', 'RQ2_2': 'get_all_genres',
    'RQ2_3': 'get_statistics', 'RQ2_4': 'get_statistics',
    'RQ3_1': 'get_all_playlists', 'RQ3_2': 'create_playlist',
    'RQ3_3': 'rename_playlist', 'RQ3_4': 'delete_playlist',
    'RQ3_5': 'add_tracks_by_genre', 'RQ3_6': 'remove_tracks_by_genre',
}


def extract_function_body(code, func_name):
    """Extract a specific function body from code."""
    pattern = (
        r'(?:@[^\n]+\n\s*)?def\s+' + re.escape(func_name) +
        r'\s*\([^)]*\)\s*(?:->\s*\w+\s*)?\s*:.*?'
        r'(?=\n(?:@[^\n]+\n\s*)?def\s+\w+\s*\(|\Z)'
    )
    m = re.search(pattern, code, re.DOTALL)
    return m.group() if m else ''


def has_function_scoping(fn_source, target_func_name):
    """Check if the check function limits its scope to target_func_name."""
    # Looks for: ast.FunctionDef, node.name == 'xxx', FunctionDef and node.name
    source = fn_source.lower()
    indicators = [
        f"node.name == '{target_func_name}'",
        f'node.name == "{target_func_name}"',
        f'name == \'{target_func_name}\'',
        f'name == "{target_func_name}"',
        f"'{target_func_name}'",
        f'"{target_func_name}"',
        f'node.name ==',
    ]
    return any(ind in source for ind in indicators)


def has_regex_search(fn_source):
    """Check if check function uses re.search at global scope."""
    return bool(re.search(r're\.search\(.*code\)', fn_source))


def has_ast_walk(fn_source):
    """Check if check function uses ast.walk without function scoping."""
    has_walk = 'ast.walk' in fn_source
    has_func_def_check = 'FunctionDef' in fn_source and 'node.name' in fn_source
    return has_walk and not has_func_def_check


def compile_check(fn_body):
    """Try to compile and exec the check function."""
    fn_source = fn_body.strip()
    try:
        tree = ast.parse(fn_source)
        local_ns = {}
        exec(fn_source, {'__builtins__': __builtins__}, local_ns)
        fn = local_ns.get('check')
        return fn if callable(fn) else None
    except Exception:
        return None


def test_against_code(fn, code_samples, label):
    """Test a check function against code samples, return match counts."""
    results = []
    for name, code in code_samples:
        try:
            result = fn(code)
            matched = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            results.append((name, matched))
        except Exception:
            results.append((name, 'ERROR'))
    return results


# ============================================================
# Main Audit
# ============================================================

print('=' * 70)
print('  FS CHECK FUNCTION AUDIT')
print('=' * 70)

issues = []
stats = defaultdict(int)

# Build template function extracts for testing
template_func_extracts = {}
for fname in TEMPLATE_FUNCS:
    body = extract_function_body(template_code, fname)
    if body:
        template_func_extracts[fname] = body

# Build reference function extracts
ref_func_extracts = {}
ref_only_funcs = set()
for match in re.finditer(r'def\s+(\w+)\s*\(', ref_code):
    ref_only_funcs.add(match.group(1))

total = 0
ok = 0
for fs in fs_list:
    sig_type = fs.get('signature_type', 'regex')
    if sig_type != 'check_function':
        continue
    total += 1
    fid = fs.get('id', '?')
    ftype = fs.get('fs_type', '?')
    crit = fs.get('criterion', '?')
    name = fs.get('name', '?')
    fn_body = fs.get('check_function', '')
    weight = fs.get('_scoring_weight', 1.0)

    target_func = CRIT_FUNC.get(crit, '')

    issue_list = []

    # Check 1: Function scoping
    if target_func:
        if has_ast_walk(fn_body) and not has_function_scoping(fn_body, target_func):
            issue_list.append('NO_FUNC_SCOPE: ast.walk() traverses entire file, not just target function')
            stats['no_func_scope'] += 1
        if has_regex_search(fn_body) and not has_function_scoping(fn_body, target_func):
            issue_list.append('NO_FUNC_SCOPE_REGEX: re.search on entire file without extracting target function first')
            stats['no_func_scope_regex'] += 1

    # Check 2: Template matching
    check_fn = compile_check(fn_body)
    if check_fn:
        for tf_name, tf_body in template_func_extracts.items():
            try:
                result = check_fn(tf_body)
                matched = result[0] if isinstance(result, tuple) and len(result) == 2 else result
                if matched:
                    issue_list.append(f'MATCHES_TEMPLATE_FUNC: matches template function {tf_name}()')
                    stats['matches_template'] += 1
                    break
            except Exception:
                pass

    # Check 3: Reference matching (negative FS only)
    if check_fn and ftype == 'negative':
        try:
            result = check_fn(ref_code)
            matched = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            if matched:
                issue_list.append('MATCHES_REFERENCE: negative FS matches reference code')
                stats['matches_ref'] += 1
        except Exception:
            pass

    # Check 4: Stub/pass matching (positive FS only)
    if check_fn and ftype == 'positive':
        stub_code = 'def update_playlist_tracks(playlist_tracks_file):\n    pass\n'
        try:
            result = check_fn(stub_code)
            matched = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            if matched:
                issue_list.append('MATCHES_STUB: positive FS matches pass-only stub')
                stats['matches_stub'] += 1
        except Exception:
            pass

    # Check 5: Negative assertion detection (Type A patterns in check function)
    type_a_indicators = [
        ('not in code', 'NEGATIVE_ASSERT: checks ABSENCE of pattern (Type A)'),
        ('not code', 'NEGATIVE_ASSERT: checks ABSENCE of pattern (Type A)'),
        ('return not', 'NEGATIVE_ASSERT: return not X (Type A)'),
    ]
    for indicator, msg in type_a_indicators:
        if indicator in fn_body.lower():
            issue_list.append(msg)
            stats['type_a_pattern'] += 1

    if not issue_list:
        ok += 1
    else:
        issues.append({
            'fs_id': fid, 'fs_type': ftype, 'criterion': crit,
            'name': name, 'weight': weight, 'issues': issue_list,
        })

    stats['total'] += 1

# Print results
print(f'\nTotal check function FS: {total}')
print(f'OK (no issues): {ok} ({ok/total*100:.1f}%)')
print(f'With issues: {len(issues)} ({(total-ok)/total*100:.1f}%)')
print()

print('Issue breakdown:')
for key, count in sorted(stats.items(), key=lambda x: -x[1]):
    if key != 'total':
        print(f'  {key}: {count} FS')

print(f'\n{"=" * 70}')
print('  DETAILED ISSUES')
print('=' * 70)

for item in issues:
    fid = item['fs_id']
    ftype = item['fs_type']
    crit = item['criterion']
    w = item['weight']
    print(f'\n[{fid}] {ftype} {crit} (weight={w}): {item["name"]}')
    for iss in item['issues']:
        print(f'  [!!] {iss}')

print(f'\n{"=" * 70}')
print('  PROMPT FIX FEASIBILITY')
print('=' * 70)

# Can prompt fix solve these?
print('''
ISSUE                   FIXABLE BY PROMPT?   WHY
─────────────────────────────────────────────────────────
NO_FUNC_SCOPE           [FIXABLE] YES         Add "MUST filter by FunctionDef name" rule
MATCHES_TEMPLATE        [FIXABLE] YES (mostly) Add "MUST NOT match template functions" rule
                                              + list template function names in prompt
MATCHES_REFERENCE       [PARTIAL] PARTIALLY   Prompt can help but some patterns are
                                              genuinely ambiguous (INSERT OR IGNORE is
                                              correct in reference, "bad" per README)
MATCHES_STUB            [FIXABLE] YES         Add "MUST NOT match pass-only stubs" rule
TYPE_A (NOT IN)         [FIXABLE] YES         Already prohibited in prompt; strengthen
''')

# Determine if 100% fixable
total_issues = sum(1 for item in issues for _ in item['issues'])
fixable = sum(1 for item in issues for iss in item['issues']
              if 'NO_FUNC_SCOPE' in iss or 'TEMPLATE' in iss or 'STUB' in iss or 'TYPE_A' in iss)
ambiguous = sum(1 for item in issues for iss in item['issues']
                if 'MATCHES_REFERENCE' in iss)

print(f'\nTotal issues: {total_issues}')
print(f'Fixable by prompt: {fixable} ({fixable/total_issues*100:.1f}%)')
print(f'Ambiguous (MATCHES_REFERENCE): {ambiguous} ({ambiguous/total_issues*100:.1f}%)')
print(f'\nConclusion: {"[PARTIAL] NOT 100% fixable" if ambiguous > 0 else "[OK] 100% fixable by prompt"}')
if ambiguous > 0:
    print(f'  {ambiguous} MATCHES_REFERENCE issues require README label clarification,')
    print(f'  not just prompt improvement. These are inherent to the CW ground truth.')
