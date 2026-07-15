"""
Sandbox Verification for AI-Generated Reference Solutions
===========================================================
Assembles per-criterion reference function snippets into complete,
runnable applications and verifies them via execution.

Supports:
  - Task 1 (pure functions): direct execution with test data
  - Task 2/3 (Flask routes): Flask test_client

Output: references/sandbox/report.json

Usage:
    python sandbox.py <question_dir> <ref_dir> [task_id]
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Task-to-function mapping (must match question/code/iMusic.py) ──
TASK_FUNCTIONS = {
    'Task1': ['update_playlist_tracks'],
    'Task2': ['statistics', 'get_all_genres', 'get_statistics'],
    'Task3': ['playlists', 'get_all_playlists', 'create_playlist',
              'rename_playlist', 'delete_playlist',
              'add_tracks_by_genre', 'remove_tracks_by_genre'],
}

# Functions decorated with @app.route (from template)
ROUTE_FUNCTIONS = {
    'statistics', 'playlists', 'create_playlist', 'rename_playlist',
    'delete_playlist', 'add_tracks_by_genre', 'remove_tracks_by_genre',
}

# Flattened set of all implementable functions
ALL_TARGET_FUNCTIONS = set()
for funcs in TASK_FUNCTIONS.values():
    ALL_TARGET_FUNCTIONS.update(funcs)


def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_function_name(code: str) -> str | None:
    """Extract the function name from a code snippet."""
    m = re.search(r'def\s+(\w+)\s*\(', code)
    return m.group(1) if m else None


def replace_stub_in_template(template: str, func_name: str, func_code: str) -> str:
    """Replace a pass stub in the template with the given function code.

    Handles:
      - Functions with @app.route decorator
      - Functions without decorator
    """
    # Pattern: optional decorator + def + stub body (pass # ...)
    # Match from def... up to and including the pass line
    pattern = (
        rf'((?:@app\.route[^\n]*\n\s*)?)'   # optional decorator
        rf'(def\s+{func_name}\s*\([^)]*\).*?:\n)'  # def line
        rf'((?:\s*#[^\n]*\n)*)'              # comment lines
        rf'\s*pass\s*#.*?(?:\n|$)'           # pass # Delete this line...
    )

    # Build replacement: keep decorator + def, replace body
    def _replace(m):
        decorator = m.group(1)
        def_line = m.group(2)
        # Extract just the function body from func_code (remove def line)
        body_match = re.search(
            rf'def\s+{func_name}\s*\([^)]*\).*?:\n(.*)',
            func_code, re.DOTALL
        )
        if body_match:
            body = body_match.group(1)
        else:
            # No match — use entire code, stripping the def line
            body = re.sub(rf'^def\s+{func_name}\s*\([^)]*\).*?:\n', '', func_code)
        # Indent body to match the template's indentation
        indented = '\n'.join('    ' + line if line.strip() else ''
                            for line in body.split('\n'))
        return f'{decorator}{def_line}\n{indented}\n'

    result = re.sub(pattern, _replace, template, flags=re.DOTALL)
    return result


def add_route_decorator(func_name: str, func_code: str) -> str:
    """Ensure a function has an @app.route decorator if it's a route function."""
    if func_name not in ROUTE_FUNCTIONS:
        return func_code
    if '@app.route' in func_code:
        return func_code

    # Map function to route
    route_map = {
        'statistics': '/statistics/',
        'playlists': '/playlists/',
        'create_playlist': '/playlists/',
        'rename_playlist': '/playlists/',
        'delete_playlist': '/playlists/',
        'add_tracks_by_genre': '/playlists/',
        'remove_tracks_by_genre': '/playlists/',
    }
    route = route_map.get(func_name, f'/{func_name}/')

    # Insert @app.route before def
    decorator = f"@app.route('{route}', methods=['GET', 'POST'])\n"
    return decorator + func_code


def assemble_app(template_path: str, variants: dict[str, str]) -> str:
    """Assemble variants into a complete iMusic.py.

    Args:
        template_path: Path to question/code/iMusic.py
        variants: {function_name: code_snippet}

    Returns:
        Complete iMusic.py source code.
    """
    template = read_file(template_path)

    for func_name, code in variants.items():
        code_with_route = add_route_decorator(func_name, code)
        template = replace_stub_in_template(template, func_name, code_with_route)

    return template


def _get_task1_setup(tmpdir: str) -> str:
    """Minimal setup for Task 1 — no Flask dependency."""
    return f'''import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(r"{tmpdir}")
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DB_FILE = BASE_DIR / 'data/iMusic.db'
'''


def test_task1_function(code: str, func_name: str) -> dict:
    """Test a Task 1 function by direct execution.

    Creates a temp directory with test TSV + database, calls the function,
    and verifies it runs without error.
    """
    result = {
        'function': func_name,
        'compile_ok': False,
        'exec_ok': False,
        'error': None,
    }

    # 1. Create test environment first
    tmpdir = tempfile.mkdtemp(prefix='sandbox_task1_')
    try:
        # Copy iMusic.db from question code data
        db_src = BASE_DIR / 'question' / 'code' / 'data' / 'iMusic.db'
        db_dst = Path(tmpdir) / 'data' / 'iMusic.db'
        (Path(tmpdir) / 'data').mkdir(parents=True, exist_ok=True)
        if db_src.exists():
            shutil.copy2(db_src, db_dst)

        # Create test TSV with valid data
        tsv_path = Path(tmpdir) / 'test_tracks.tsv'
        tsv_content = (
            'PlaylistId\tTrackId\n'
            '1\t1\n'
            '1\t2\n'
        )
        tsv_path.write_text(tsv_content)

        # Prepend minimal setup (no Flask)
        setup = _get_task1_setup(tmpdir)
        full_code = setup + '\n\n' + code

        # 2. Compile check
        try:
            compile(full_code, '<ref>', 'exec')
            result['compile_ok'] = True
        except SyntaxError as e:
            result['error'] = f'SyntaxError: {e}'
            return result

        # 3. Execute
        import_context = {'Path': Path, 'csv': __import__('csv'), 'sqlite3': __import__('sqlite3')}
        exec(full_code, import_context)

        fn = import_context.get(func_name)
        if fn and callable(fn):
            fn(tsv_path)
            result['exec_ok'] = True
        elif fn:
            result['error'] = f'{func_name} is not callable (type: {type(fn).__name__})'
        else:
            result['error'] = f'{func_name} not found after exec (available: {list(import_context.keys())})'

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def test_flask_app(app_source: str, task_id: str) -> dict:
    """Test a complete Flask app via test_client.

    Args:
        app_source: Complete iMusic.py source code
        task_id: Task2 or Task3

    Returns:
        {compile_ok, routes_tested, failures, errors}
    """
    result = {
        'task': task_id,
        'compile_ok': False,
        'routes_tested': [],
        'failures': [],
        'error': None,
    }

    # 1. Compile
    try:
        compile(app_source, '<sandbox>', 'exec')
        result['compile_ok'] = True
    except SyntaxError as e:
        result['error'] = f'SyntaxError: {e}'
        return result

    # 2. Create Flask app in isolated namespace
    tmpdir = tempfile.mkdtemp(prefix='sandbox_flask_')
    try:
        # Copy database
        db_src = BASE_DIR / 'question' / 'code' / 'data' / 'iMusic.db'
        db_dst = Path(tmpdir) / 'data' / 'iMusic.db'
        (Path(tmpdir) / 'data').mkdir(parents=True, exist_ok=True)
        if db_src.exists():
            shutil.copy2(db_src, db_dst)

        # Create uploads directory
        (Path(tmpdir) / 'uploads').mkdir(exist_ok=True)

        # Modify app_source to use temp directory
        app_source_modified = app_source.replace(
            "BASE_DIR = Path(__file__).resolve().parent",
            f'BASE_DIR = Path(r"{tmpdir}")'
        )

        # Also ensure templates exist
        template_dir = BASE_DIR / 'question' / 'code' / 'templates'
        if template_dir.is_dir():
            shutil.copytree(template_dir, Path(tmpdir) / 'templates', dirs_exist_ok=True)

        # Execute and get Flask app
        # Must provide __name__ so Flask(app) can resolve root path
        exec_globals = {'__name__': 'sandbox_test', '__file__': str(Path(tmpdir) / 'app.py')}
        exec(app_source_modified, exec_globals)
        app = exec_globals.get('app')
        if not app:
            result['error'] = 'No Flask app found in executed code'
            return result

        # Fix root_path — exec() doesn't create a proper module so Flask guesses wrong
        app.root_path = tmpdir
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test'
        client = app.test_client()

        # 3. Test routes
        route_tests = {
            'Task2': [
                ('GET', '/statistics/', None),
                ('GET', '/statistics/?genre=1&sort_by=NumberOfTracks&sort_order=ASC', None),
            ],
            'Task3': [
                ('GET', '/playlists/', None),
            ],
        }

        for method, path, data in route_tests.get(task_id, []):
            try:
                if method == 'GET':
                    resp = client.get(path, follow_redirects=True)
                elif method == 'POST':
                    resp = client.post(path, data=data or {}, follow_redirects=True)
                else:
                    continue

                route_result = {
                    'path': path,
                    'method': method,
                    'status': resp.status_code,
                }
                if resp.status_code >= 500:
                    route_result['error'] = f'HTTP {resp.status_code}'
                    result['failures'].append(route_result)
                result['routes_tested'].append(route_result)

            except Exception as e:
                result['failures'].append({
                    'path': path, 'method': method,
                    'error': f'{type(e).__name__}: {e}',
                })

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def verify_references(question_dir: str, ref_dir: str,
                       task_id: str = '') -> dict:
    """Main sandbox verification.

    Returns report dict with per-variant results.
    """
    template_path = os.path.join(question_dir, 'code', 'iMusic.py')
    if not os.path.exists(template_path):
        print(f'ERROR: Template not found at {template_path}')
        return {'error': 'Template not found'}

    report = {
        'template': template_path,
        'ref_dir': ref_dir,
        'task1_results': [],
        'task2_results': [],
        'task3_results': [],
        'summary': {},
    }

    tasks_to_test = [task_id] if task_id else ['Task1', 'Task2', 'Task3']

    for t in tasks_to_test:
        task_ref_dir = os.path.join(ref_dir, t)
        if not os.path.isdir(task_ref_dir):
            task_ref_dir = ref_dir

        py_files = sorted([
            f for f in os.listdir(task_ref_dir)
            if f.endswith('.py') and f != 'criterion_variants.json'
        ]) if os.path.isdir(task_ref_dir) else []

        if not py_files:
            report[f'{t.lower()}_results'].append({
                'status': 'no_references', 'message': f'No .py files in {task_ref_dir}',
            })
            continue

        print(f'\n  {t}: {len(py_files)} reference files')

        # Group variants by function name
        func_variants: dict[str, list[tuple[str, str]]] = {}
        for fname in py_files:
            fpath = os.path.join(task_ref_dir, fname)
            code = read_file(fpath)
            func_name = extract_function_name(code)
            if not func_name:
                continue
            func_variants.setdefault(func_name, []).append((fname, code))

        # ── Task 1: test each variant individually (pure functions, no deps) ──
        if t == 'Task1':
            for func_name, variants in func_variants.items():
                for fname, code in variants:
                    test_result = test_task1_function(code, func_name)
                    test_result['file'] = fname
                    report['task1_results'].append(test_result)
                    status = 'PASS' if test_result['exec_ok'] else 'FAIL'
                    print(f'    {fname}: {status} ({func_name})'
                          f'{"" if test_result["exec_ok"] else " — " + str(test_result.get("error", "")[:60])}')

        # ── Task 2/3: assemble complete app, one function per slot ──
        else:
            # Pick first variant for each function
            chosen = {}
            for func_name, variants in func_variants.items():
                chosen[func_name] = variants[0][0]  # filename

            full_app = assemble_app(template_path, {fn: read_file(os.path.join(task_ref_dir, fname))
                                                      for fn, fname in chosen.items()})
            test_result = test_flask_app(full_app, t)
            test_result['assembled_from'] = chosen
            report[f'{t.lower()}_results'].append(test_result)

            n_routes = len(test_result.get('routes_tested', []))
            n_fails = len(test_result.get('failures', []))
            status = 'PASS' if test_result['compile_ok'] and n_fails == 0 and n_routes > 0 else 'FAIL'
            print(f'    Assembled with: {chosen}')
            print(f'    Status: {status} — {n_routes} routes, {n_fails} failures'
                  f'{"" if not test_result.get("error") else " — " + str(test_result.get("error", ""))[:80]}')

            # Also test individual functions for compile-only
            for func_name, variants in func_variants.items():
                for fname, code in variants:
                    try:
                        compile(code, '<ref>', 'exec')
                    except SyntaxError as e:
                        report[f'{t.lower()}_results'].append({
                            'file': fname, 'function': func_name,
                            'compile_ok': False, 'error': f'SyntaxError: {e}',
                        })

    # ── Summary ──
    all_results = (
        report['task1_results'] + report['task2_results'] + report['task3_results']
    )
    total = len(all_results)
    passed = sum(
        1 for r in all_results
        if r.get('exec_ok') or (r.get('compile_ok') and not r.get('failures'))
    )
    failed = total - passed
    report['summary'] = {
        'total_variants': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': round(passed / total * 100, 1) if total > 0 else 0,
    }

    # ── Save ──
    sandbox_dir = os.path.join(ref_dir, 'sandbox')
    os.makedirs(sandbox_dir, exist_ok=True)
    report_path = os.path.join(sandbox_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n  Report saved to {report_path}')

    return report


def print_summary(report: dict):
    """Print human-readable summary."""
    s = report['summary']
    print(f'\n{"=" * 60}')
    print('  SANDBOX VERIFICATION SUMMARY')
    print('=' * 60)
    print(f'  Total variants: {s["total_variants"]}')
    print(f'  Passed: {s["passed"]}')
    print(f'  Failed: {s["failed"]}')
    print(f'  Pass rate: {s["pass_rate"]}%')

    for task_key in ['task1_results', 'task2_results', 'task3_results']:
        results = report.get(task_key, [])
        failures = [r for r in results
                    if not (r.get('exec_ok') or
                           (r.get('compile_ok') and not r.get('failures')))]
        if failures:
            print(f'\n  {task_key} failures ({len(failures)}):')
            for r in failures[:5]:
                err = r.get('error', '')
                if not err and r.get('failures'):
                    err = r['failures'][0].get('error', 'unknown')
                print(f'    {r.get("file", "?")}: {err[:100]}')


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        q_dir = sys.argv[1]
        r_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(BASE_DIR, 'references')
        tid = sys.argv[3] if len(sys.argv) >= 4 else ''
        report = verify_references(q_dir, r_dir, tid)
        print_summary(report)
    else:
        print('Usage: python sandbox.py <question_dir> [ref_dir] [task_id]')
        print('Example: python sandbox.py question references/q1_iMusic')
