"""
Test-Driven FS Generator v2 — Real Sandbox
============================================
Uses Flask test_client for Task 2/3 and direct execution for Task 1.
Built on sandbox.py's assemble_app/replace_stub_in_template.

AI generates test cases. Tests are executed in a proper Flask environment.
Results are precomputed into a test matrix.

Usage:
    python test_fs_generator.py question submission q1_iMusic
"""

import json, os, re, sys, io, traceback, tempfile, shutil, time, sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE, '.env'))

from ai_pipeline import call_deepseek, extract_json, _repair_json, read_file, collect_submissions_by_task
from ground_truth import load_all_readmes
from sandbox import (
    assemble_app, replace_stub_in_template, extract_function_name,
    TASK_FUNCTIONS, ROUTE_FUNCTIONS, ALL_TARGET_FUNCTIONS,
)

DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
TEMPLATE_PATH = os.path.join(BASE, 'question', 'code', 'iMusic.py')


# ============================================================
# System prompt
# ============================================================

TEST_GEN_SYSTEM_V2 = """You generate test cases for grading student Flask+SQLite code.
Tests run in a REAL Flask test_client environment with a real SQLite database.

Available test types:
  FUNCTION_CALL: Call function directly, check return value
  FLASK_GET: GET a route, check response (status, data in response)
  FLASK_POST: POST form data to a route, check redirect/flash/DB state
  DB_CHECK: Run SQL query, check expected rows

Each test has:
  - id, description, checks ("good" or "bad")
  - type: FUNCTION_CALL | FLASK_GET | FLASK_POST | DB_CHECK
  - setup: SQL to prepare test data (optional)
  - For FUNCTION_CALL: call.func, call.args, verify (return_value or return_type or not_raises)
  - For FLASK_GET/POST: route, data (form dict), verify.status, verify.contains, verify.not_contains
  - For DB_CHECK: verify.query, verify.expected_rows or verify.row_count

Functions by task:
  Task1: update_playlist_tracks(playlist_tracks_file: Path)
  Task2: get_all_genres(), get_statistics(genre_id, sort_column, sort_order)
         Route: /statistics/ (GET/POST)
  Task3: get_all_playlists(), create_playlist(), rename_playlist(),
         delete_playlist(), add_tracks_by_genre(), remove_tracks_by_genre()
         Route: /playlists/ (GET/POST)

Output ONLY valid JSON: {"tests": [...]}"""


def build_test_gen_prompt_v2(criterion: str, rubric_criteria: list[dict]) -> str:
    crit = {}
    for rc in rubric_criteria:
        if rc.get('id') == criterion:
            crit = rc
            break

    func_map = {
        'RQ1_1': 'update_playlist_tracks', 'RQ1_2': 'update_playlist_tracks',
        'RQ1_3': 'update_playlist_tracks', 'RQ1_4': 'update_playlist_tracks',
        'RQ2_1': 'get_all_genres', 'RQ2_2': 'get_all_genres',
        'RQ2_3': 'get_statistics', 'RQ2_4': 'get_statistics',
        'RQ3_1': 'get_all_playlists', 'RQ3_2': 'create_playlist',
        'RQ3_3': 'rename_playlist', 'RQ3_4': 'delete_playlist',
        'RQ3_5': 'add_tracks_by_genre', 'RQ3_6': 'remove_tracks_by_genre',
    }
    func = func_map.get(criterion, 'unknown')
    tn = int(re.search(r'(\d)', criterion).group(1))
    route = '/statistics/' if tn == 2 else '/playlists/' if tn == 3 else None

    route_info = f'\nFlask route: {route} (GET/POST)' if route else '\nNo Flask route — pure function call.'

    return f"""## Criterion: {criterion} — {crit.get('name', criterion)}
Primary function: {func}(){route_info}

### Good patterns:
{chr(10).join(f'- {gp}' for gp in crit.get('good_patterns', []))}

### Bad patterns:
{chr(10).join(f'- {bp}' for bp in crit.get('bad_patterns', []))}

## Task
Generate 3-5 test cases. For each good pattern: test that PASSES when code is correct.
For each bad pattern: test that PASSES when code AVOIDS the mistake.

Tests run in a Flask app with a real SQLite DB. Tables: Playlist(PlaylistId,Name),
Genre(GenreId,Name), Track(TrackId,Name,GenreId,Milliseconds,UnitPrice),
PlaylistTrack(PlaylistId,TrackId). Pre-loaded with sample data.

Output ONLY JSON: {{"tests": [{{"id":"test_1","description":"...","checks":"good","type":"...","setup":"optional SQL","call":{{...}},"verify":{{...}}}}]}}"""


# ============================================================
# Sandbox execution (Flask test_client)
# ============================================================

def _setup_db(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS Playlist (PlaylistId INTEGER PRIMARY KEY, Name TEXT);
        CREATE TABLE IF NOT EXISTS Genre (GenreId INTEGER PRIMARY KEY, Name TEXT);
        CREATE TABLE IF NOT EXISTS Track (TrackId INTEGER PRIMARY KEY, Name TEXT, GenreId INTEGER,
            Milliseconds INTEGER, UnitPrice REAL, AlbumId INTEGER, MediaTypeId INTEGER, Composer TEXT);
        CREATE TABLE IF NOT EXISTS PlaylistTrack (PlaylistId INTEGER, TrackId INTEGER,
            PRIMARY KEY (PlaylistId, TrackId));
        INSERT OR IGNORE INTO Genre VALUES (1,'Rock'),(2,'Jazz'),(3,'Classical');
        INSERT OR IGNORE INTO Track VALUES (1,'Song A',1,240000,0.99,1,1,'A1'),(2,'Song B',1,180000,1.29,1,1,'A2'),(3,'Song C',2,300000,0.79,2,1,'A3');
        INSERT OR IGNORE INTO Playlist VALUES (1,'My Playlist'),(2,'Empty List');
        INSERT OR IGNORE INTO PlaylistTrack VALUES (1,1);
    ''')
    conn.commit()
    conn.close()


def _extract_tsv_for_test(tmpdir: str) -> str:
    """Create a small test TSV file."""
    path = os.path.join(tmpdir, 'test_playlist.tsv')
    with open(path, 'w') as f:
        f.write('PlaylistId\tTrackId\n1\t2\n1\t3\n2\t1\n')
    return path


def run_student_tests(code_dict: dict[int, str], tests: list[dict],
                      criterion: str, template_path: str) -> list[dict]:
    """Run tests against one student using Flask test_client or direct execution."""
    tn = int(re.search(r'(\d)', criterion).group(1))

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'data', 'iMusic.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _setup_db(db_path)

        # Create uploads dir
        uploads = os.path.join(tmpdir, 'uploads')
        os.makedirs(uploads, exist_ok=True)

        # Build complete app for Flask tasks
        if tn >= 2:
            # Collect all student functions
            variants = {}
            for task_funcs in TASK_FUNCTIONS.values():
                for fn in task_funcs:
                    for t_num, code in code_dict.items():
                        body = _extract_function_body(code, fn)
                        if body:
                            variants[fn] = body
                            break

            # Assemble app
            app_code = assemble_app(template_path, variants)

            # Write app
            app_path = os.path.join(tmpdir, 'iMusic.py')
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(app_code)

            # Run in subprocess with Flask test_client
            return _run_flask_tests(app_path, db_path, tmpdir, tests, criterion, tn)

        else:
            # Task 1: direct function call
            code = code_dict.get(1, '')
            return _run_task1_tests(code, db_path, tmpdir, tests)


def _extract_function_body(code: str, func_name: str) -> str:
    pattern = (
        r'(?:@[^\n]+\n\s*)?def\s+' + re.escape(func_name) +
        r'\s*\([^)]*\)\s*(?:->\s*\w+\s*)?\s*:.*?'
        r'(?=\n(?:@[^\n]+\n\s*)?def\s+\w+\s*\(|\Z)'
    )
    m = re.search(pattern, code, re.DOTALL)
    return m.group() if m else ''


def _run_task1_tests(code: str, db_path: str, tmpdir: str, tests: list[dict]) -> list[dict]:
    """Task 1: call update_playlist_tracks directly."""
    test_tsv = _extract_tsv_for_test(tmpdir)

    results = []
    for test in tests:
        tid = test['id']
        passed = False
        error = None

        try:
            # Build script
            body = _extract_function_body(code, 'update_playlist_tracks')
            if not body:
                results.append({'test_id': tid, 'passed': False, 'error': 'Function not found'})
                continue

            # Execute in subprocess
            script = f'''
import sys, os, json
sys.path.insert(0, r'{tmpdir}')
os.chdir(r'{tmpdir}')
DB_FILE = r'{db_path}'

{body}

# Run setup
import sqlite3
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
{test.get('setup', '')}
conn.commit()

# Call function
result = update_playlist_tracks(r'{test_tsv}')

# Verify
verify = {json.dumps(test.get('verify', {{}}))}
if 'return_value' in verify:
    ok = (result == verify['return_value'])
elif 'return_type' in verify:
    ok = isinstance(result, eval(verify['return_type']))
else:
    ok = True

# DB check
if verify.get('query'):
    c.execute(verify['query'])
    rows = c.fetchall()
    if 'expected_rows' in verify:
        ok = ok and (rows == [tuple(r) for r in verify['expected_rows']])
    elif 'row_count' in verify:
        ok = ok and (len(rows) >= verify['row_count'])

conn.close()
print(json.dumps({{'passed': ok}}))
'''
            import subprocess
            r = subprocess.run([sys.executable, '-c', script],
                             capture_output=True, text=True, timeout=15, cwd=tmpdir)
            if r.returncode == 0 and r.stdout.strip():
                data = json.loads(r.stdout.strip().split('\n')[-1])
                passed = data.get('passed', False)
            else:
                error = r.stderr[:200]
                passed = False
        except Exception as e:
            error = str(e)
            passed = False

        results.append({'test_id': tid, 'passed': passed, 'error': error})

    return results


def _run_flask_tests(app_path: str, db_path: str, tmpdir: str, tests: list[dict],
                     criterion: str, tn: int) -> list[dict]:
    """Task 2/3: use Flask test_client."""
    test_tsv = _extract_tsv_for_test(tmpdir) if tn == 1 else ''

    script = f'''
import sys, os, json
os.chdir(r'{tmpdir}')
sys.path.insert(0, r'{tmpdir}')

# Setup paths before importing app
import __main__
__main__.__file__ = r'{app_path}'

# Patch DB_FILE before import
import builtins
_orig_import = builtins.__import__
def _patched_import(name, *args, **kw):
    if name == 'sqlite3':
        m = _orig_import(name, *args, **kw)
        m.DB_FILE = r'{db_path}'
        return m
    return _orig_import(name, *args, **kw)
builtins.__import__ = _patched_import

# Now import the app
exec(open(r'{app_path}', encoding='utf-8').read())

# Setup DB
import sqlite3
conn = sqlite3.connect(r'{db_path}')
c = conn.cursor()
{chr(10).join(test.get('setup', '') for test in tests if test.get('setup'))}
conn.commit()
conn.close()

results = []
for test in {json.dumps(tests)}:
    tid = test['id']
    ttype = test.get('type', 'FUNCTION_CALL')
    passed = False
    error = None

    try:
        if ttype == 'FUNCTION_CALL':
            fn_name = test['call']['func']
            fn = globals().get(fn_name)
            if fn is None:
                error = f'Function {{fn_name}} not found'
            else:
                args = test['call'].get('args', [])
                result = fn(*args)
                verify = test.get('verify', {{}})
                if 'return_type' in verify:
                    passed = isinstance(result, eval(verify['return_type']))
                elif 'return_not_empty' in verify:
                    passed = bool(result)
                else:
                    passed = True

        elif ttype in ('FLASK_GET', 'FLASK_POST'):
            with app.test_client() as client:
                route = test.get('route', '/statistics/')
                if ttype == 'FLASK_GET':
                    resp = client.get(route)
                else:
                    data = test.get('data', {{}})
                    resp = client.post(route, data=data, follow_redirects=True)

                verify = test.get('verify', {{}})
                if 'status' in verify:
                    passed = (resp.status_code == verify['status'])
                elif 'contains' in verify:
                    passed = (verify['contains'] in resp.get_data(as_text=True))
                elif 'not_contains' in verify:
                    passed = (verify['not_contains'] not in resp.get_data(as_text=True))
                else:
                    passed = (resp.status_code == 200)

        elif ttype == 'DB_CHECK':
            conn2 = sqlite3.connect(r'{db_path}')
            c2 = conn2.cursor()
            verify = test.get('verify', {{}})
            if verify.get('query'):
                c2.execute(verify['query'])
                rows = c2.fetchall()
                if 'expected_rows' in verify:
                    passed = (rows == [tuple(r) for r in verify['expected_rows']])
                elif 'row_count' in verify:
                    passed = (len(rows) == verify['row_count'])
            conn2.close()

    except Exception as e:
        error = str(e)
        if test.get('verify', {{}}).get('not_raises'):
            passed = True
        else:
            passed = False

    results.append({{'test_id': tid, 'passed': passed, 'error': error}})

print(json.dumps({{'test_results': results}}))
'''

    import subprocess
    try:
        r = subprocess.run([sys.executable, '-c', script],
                         capture_output=True, text=True, timeout=30, cwd=tmpdir)
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout.strip().split('\n')[-1])
            return data.get('test_results', [])
        return [{'test_id': t['id'], 'passed': False, 'error': r.stderr[:200]} for t in tests]
    except subprocess.TimeoutExpired:
        return [{'test_id': t['id'], 'passed': False, 'error': 'Timeout'} for t in tests]
    except Exception as e:
        return [{'test_id': t['id'], 'passed': False, 'error': str(e)} for t in tests]


# ============================================================
# Main orchestrator (reuse from v1)
# ============================================================

def generate_tests_for_criterion(criterion, rubric_criteria, model_override=None):
    prompt = build_test_gen_prompt_v2(criterion, rubric_criteria)
    for attempt in range(3):
        resp = call_deepseek(TEST_GEN_SYSTEM_V2, prompt, temperature=0.3, model_override=model_override)
        if not resp: continue
        try:
            data = extract_json(resp)
            tests = data.get('tests', [])
            if tests: return tests
        except:
            try:
                data = extract_json(_repair_json(resp))
                tests = data.get('tests', [])
                if tests: return tests
            except: continue
    return None


def run_all_tests(task_subs, criterion_tests, template_path):
    """Batch all students per criterion into one subprocess. Much faster."""
    results = defaultdict(dict)
    total_criteria = len(criterion_tests)
    count = 0

    for criterion, tests in criterion_tests.items():
        tn = int(re.search(r'(\d)', criterion).group(1))
        subs = task_subs.get(tn, [])
        if not subs: continue

        # Build a batch script: one subprocess tests ALL students for this criterion
        print(f'    [{criterion}] Testing {len(subs)} students...')
        batch_results = _run_criterion_batch(subs, tests, criterion, template_path, tn)

        for sid, tr in batch_results.items():
            results[sid][criterion] = tr

        count += 1
        print(f'    [{criterion}] Done ({count}/{total_criteria})')

    return dict(results)


def _run_criterion_batch(subs, tests, criterion, template_path, tn):
    """Run all students for one criterion in a single subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'data', 'iMusic.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _setup_db(db_path)
        os.makedirs(os.path.join(tmpdir, 'uploads'), exist_ok=True)
        test_tsv = _extract_tsv_for_test(tmpdir)

        # Build inputs: student codes
        student_inputs = {}
        for s in subs:
            sid = s['student']
            code = s.get('code', '')
            if code:
                if tn == 1:
                    body = _extract_function_body(code, 'update_playlist_tracks')
                    student_inputs[sid] = body or ''
                else:
                    student_inputs[sid] = code

        if not student_inputs:
            return {}

        # Write inputs and tests to JSON files
        inputs_path = os.path.join(tmpdir, 'student_inputs.json')
        tests_path = os.path.join(tmpdir, 'tests.json')
        with open(inputs_path, 'w', encoding='utf-8') as f:
            json.dump(student_inputs, f)
        with open(tests_path, 'w', encoding='utf-8') as f:
            json.dump(tests, f)

        # Build the batch runner script
        if tn == 1:
            script = _build_task1_batch_script(inputs_path, tests_path, db_path, test_tsv)
        else:
            script = _build_flask_batch_script(
                inputs_path, tests_path, db_path, tmpdir, template_path, tn, criterion, test_tsv
            )

        script_path = os.path.join(tmpdir, 'batch_runner.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        import subprocess
        try:
            r = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=300,  # 5 min per criterion
                cwd=tmpdir
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if 'results' in data:
                            return data['results']
                    except: pass
            # Fallback: return empty
            return {sid: [{'test_id': t['id'], 'passed': False, 'error': 'Batch failed'} for t in tests]
                    for sid in student_inputs}
        except subprocess.TimeoutExpired:
            print(f'      TIMEOUT — skipping criterion')
            return {sid: [{'test_id': t['id'], 'passed': False, 'error': 'Timeout'} for t in tests]
                    for sid in student_inputs}
        except Exception as e:
            print(f'      ERROR: {e}')
            return {sid: [{'test_id': t['id'], 'passed': False, 'error': str(e)} for t in tests]
                    for sid in student_inputs}


def _build_task1_batch_script(inputs_path, tests_path, db_path, test_tsv):
    return f'''
import json, sqlite3, sys, os
with open(r'{inputs_path}') as f: students = json.load(f)
with open(r'{tests_path}') as f: tests = json.load(f)

results = {{}}
for sid, code in students.items():
    if not code.strip(): continue
    try:
        exec(code, {{'__builtins__': __builtins__}})
    except: pass
    student_results = []
    conn = sqlite3.connect(r'{db_path}')
    c = conn.cursor()
    for test in tests:
        tid = test['id']; passed = False; error = None
        try:
            if test.get('setup'):
                c.executescript(test['setup']); conn.commit()
            fn_name = test['call']['func']
            fn = locals().get(fn_name)
            if fn:
                result = fn(r'{test_tsv}')
                v = test.get('verify', {{}})
                if 'return_value' in v: passed = (result == v['return_value'])
                elif 'return_type' in v: passed = isinstance(result, eval(v['return_type']))
                else: passed = True
                if v.get('query'):
                    c.execute(v['query'])
                    rows = c.fetchall()
                    if 'expected_rows' in v: passed = passed and (rows == [tuple(r) for r in v['expected_rows']])
                    elif 'row_count' in v: passed = passed and (len(rows) >= v['row_count'])
            else: error = f'fn not found'
        except Exception as e: error = str(e); passed = False
        student_results.append({{'test_id': tid, 'passed': passed, 'error': error}})
    conn.close()
    results[sid] = student_results
print(json.dumps({{'results': results}}))
'''


def _build_flask_batch_script(inputs_path, tests_path, db_path, tmpdir, template_path, tn, criterion, test_tsv):
    return f'''
import json, sqlite3, sys, os, re
os.chdir(r'{tmpdir}')
sys.path.insert(0, r'{tmpdir}')
sys.path.insert(0, r'{os.path.dirname(template_path)}')

with open(r'{inputs_path}') as f: students = json.load(f)
with open(r'{tests_path}') as f: tests = json.load(f)

# Read template
with open(r'{template_path}', encoding='utf-8') as f: template = f.read()

results = {{}}
for sid, full_code in students.items():
    if not full_code.strip(): continue

    # Extract all student functions from full_code
    func_names = ['get_all_genres', 'get_statistics', 'get_all_playlists',
                  'create_playlist', 'rename_playlist', 'delete_playlist',
                  'add_tracks_by_genre', 'remove_tracks_by_genre',
                  'statistics', 'playlists']
    variants = {{}}
    for fn in func_names:
        pattern = r'(?:@[^\\n]+\\n\\s*)?def\\s+' + re.escape(fn) + r'\\s*\\([^)]*\\)\\s*(?:->\\s*\\w+\\s*)?\\s*:.*?(?=\\n(?:@[^\\n]+\\n\\s*)?def\\s+\\w+\\s*\\(|\\Z)'
        m = re.search(pattern, full_code, re.DOTALL)
        if m: variants[fn] = m.group()

    if not variants: continue

    # Build app from template with student functions
    app_code = template
    for fn, code in variants.items():
        pattern = (
            r'((?:@app\\.route[^\\n]*\\n\\s*)?)'
            r'(def\\s+' + re.escape(fn) + r'\\s*\\([^)]*\\).*?:\\n)'
            r'((?:\\s*#[^\\n]*\\n)*)'
            r'\\s*pass\\s*#.*?(?:\\n|$)'
        )
        def _replace(m, fn=fn, code=code):
            dec = m.group(1); dl = m.group(2)
            bm = re.search(r'def\\s+' + re.escape(fn) + r'\\s*\\([^)]*\\).*?:\\n(.*)', code, re.DOTALL)
            body = bm.group(1) if bm else ''
            indented = '\\n'.join('    ' + l if l.strip() else '' for l in body.split('\\n'))
            return f'{{dec}}{{dl}}\\n{{indented}}\\n'
        app_code = re.sub(pattern, _replace, app_code, flags=re.DOTALL)

    # Write app
    app_path = os.path.join(r'{tmpdir}', f'{{sid}}_app.py')
    with open(app_path, 'w', encoding='utf-8') as f: f.write(app_code)

    # Import and test
    student_results = []
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f'{{sid}}_app', app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = mod.app.test_client()

        conn = sqlite3.connect(r'{db_path}')
        c = conn.cursor()
        for test in tests:
            tid = test['id']; passed = False; error = None
            try:
                if test.get('setup'):
                    c.executescript(test['setup']); conn.commit()
                ttype = test.get('type', 'FUNCTION_CALL')
                if ttype == 'FUNCTION_CALL':
                    fn = getattr(mod, test['call']['func'], None)
                    if fn:
                        result = fn(*test['call'].get('args', []))
                        v = test.get('verify', {{}})
                        if 'return_type' in v: passed = isinstance(result, eval(v['return_type']))
                        elif 'return_not_empty' in v: passed = bool(result)
                        else: passed = True
                elif ttype in ('FLASK_GET', 'FLASK_POST'):
                    route = test.get('route', '/statistics/')
                    if ttype == 'FLASK_GET':
                        resp = client.get(route)
                    else:
                        resp = client.post(route, data=test.get('data', {{}}), follow_redirects=True)
                    v = test.get('verify', {{}})
                    if 'status' in v: passed = (resp.status_code == v['status'])
                    elif 'contains' in v: passed = (v['contains'] in resp.get_data(as_text=True))
                    elif 'not_contains' in v: passed = (v['not_contains'] not in resp.get_data(as_text=True))
                    else: passed = (resp.status_code == 200)
                elif ttype == 'DB_CHECK':
                    v = test.get('verify', {{}})
                    if v.get('query'):
                        c.execute(v['query'])
                        rows = c.fetchall()
                        if 'row_count' in v: passed = (len(rows) == v['row_count'])
                        elif 'expected_rows' in v: passed = (rows == [tuple(r) for r in v['expected_rows']])
            except Exception as e: error = str(e); passed = False
            student_results.append({{'test_id': tid, 'passed': passed, 'error': error}})
        conn.close()
    except Exception as e:
        student_results = [{{'test_id': t['id'], 'passed': False, 'error': str(e)}} for t in tests]
    results[sid] = student_results

print(json.dumps({{'results': results}}))
'''


def build_fs_from_results(results, criterion_tests):
    fs_list = []
    fs_id_counter = {}
    for criterion, tests in criterion_tests.items():
        task_num = int(re.search(r'(\d)', criterion).group(1))
        for test in tests:
            tid = test['id']
            fs_type = 'positive' if test.get('checks') == 'good' else 'negative'
            per_student = {}
            passed = failed = 0
            for sid, crit_results in results.items():
                if criterion not in crit_results: continue
                for tr in crit_results[criterion]:
                    if tr['test_id'] == tid:
                        per_student[sid] = tr['passed']
                        if tr['passed']: passed += 1
                        else: failed += 1
                        break
            fs_id_counter.setdefault(task_num, 0)
            fs_id_counter[task_num] += 1
            fs_list.append({
                'id': f'FS{task_num}.{fs_id_counter[task_num]}',
                'name': f'Test: {test.get("description", tid)}',
                'fs_type': fs_type, 'criterion': criterion,
                'signature_type': 'test_case', 'test_id': tid, 'test_spec': test,
                'feedback': test.get('description', ''),
                'task': f'Task{task_num}', 'files': [f'task{task_num}.py'],
                'auto_generated': True, 'source': 'test_driven_v2',
                'source_detail': f'test_{test.get("checks","?")}',
                '_students_passed': passed, '_students_failed': failed,
                '_per_student': per_student,
            })
    return fs_list


def _evaluate_test_fs(fs_list, all_readmes):
    criterion_metrics = defaultdict(lambda: {'tp':0,'fp':0,'fn':0,'tn':0,'neg_fs':0,'pos_fs':0})
    total_tp = total_fp = total_fn = total_tn = 0
    for fs in fs_list:
        crit = fs['criterion']; fs_type = fs['fs_type']
        per = fs.get('_per_student', {})
        for sid, passed in per.items():
            rdata = all_readmes.get(sid, {})
            criteria = rdata.get('criteria', '{}')
            if isinstance(criteria, str):
                try: criteria = eval(criteria)
                except: criteria = {}
            gt = criteria if isinstance(criteria, dict) else {}
            has_bad = bool(gt.get(crit, {}).get('bad', []) or gt.get(crit, {}).get('mistake', []))
            has_good = bool(gt.get(crit, {}).get('good', []))
            if fs_type == 'positive':
                if passed and has_good: tp_val=1; fp_val=0; fn_val=0; tn_val=0
                elif passed and not has_good: tp_val=0; fp_val=1; fn_val=0; tn_val=0
                elif not passed and has_good: tp_val=0; fp_val=0; fn_val=1; tn_val=0
                else: tp_val=0; fp_val=0; fn_val=0; tn_val=1
            else:
                if not passed and has_bad: tp_val=1; fp_val=0; fn_val=0; tn_val=0
                elif not passed and not has_bad: tp_val=0; fp_val=1; fn_val=0; tn_val=0
                elif passed and has_bad: tp_val=0; fp_val=0; fn_val=1; tn_val=0
                else: tp_val=0; fp_val=0; fn_val=0; tn_val=1
            cm = criterion_metrics[crit]
            cm['tp']+=tp_val; cm['fp']+=fp_val; cm['fn']+=fn_val; cm['tn']+=tn_val
            total_tp+=tp_val; total_fp+=fp_val; total_fn+=fn_val; total_tn+=tn_val
            if fs_type=='negative': cm['neg_fs']+=1
            else: cm['pos_fs']+=1

    op = total_tp/(total_tp+total_fp) if total_tp+total_fp>0 else 0
    or_ = total_tp/(total_tp+total_fn) if total_tp+total_fn>0 else 0
    of1 = 2*op*or_/(op+or_) if op+or_>0 else 0
    print(f'\n{"="*70}\n  TEST-DRIVEN FS v2 BLIND TEST\n{"="*70}')
    print(f'  OVERALL: P={op:.3f}  R={or_:.3f}  F1={of1:.3f}')
    print(f'  TP={total_tp}  FP={total_fp}  FN={total_fn}  TN={total_tn}')
    print(f'\n  {"Criterion":<8} {"P":>6} {"R":>6} {"F1":>6} {"TP":>4} {"FP":>4} {"FN":>4}')
    print(f'  {"-"*8} {"-"*6} {"-"*6} {"-"*6} {"-"*4} {"-"*4} {"-"*4}')
    for crit in sorted(criterion_metrics):
        m=criterion_metrics[crit]
        p=m['tp']/(m['tp']+m['fp']) if m['tp']+m['fp']>0 else 0
        r=m['tp']/(m['tp']+m['fn']) if m['tp']+m['fn']>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0
        bar='['+'#'*int(f1*20)+'-'*max(1,20-int(f1*20))+']'
        print(f'  {crit:<8} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {m["tp"]:>4} {m["fp"]:>4} {m["fn"]:>4} {bar}')
    print()


def generate_test_fs_v2(tasks, task_subs, rubric_criteria, model_override=None):
    print('='*60)
    print('  TEST-DRIVEN FS GENERATION v2 (Flask test_client)')
    print('='*60)
    all_criteria = sorted(set(rc['id'] for rc in rubric_criteria))

    # Phase 1: Generate tests
    print('\n--- Phase 1: Generating tests ---')
    cache_path = os.path.join(BASE, 'output', 'q1_iMusic', 'test_cache_v2.json')
    criterion_tests = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            criterion_tests = json.load(f)
        print(f'  Loaded {len(criterion_tests)} criteria from cache')
    for criterion in all_criteria:
        if criterion in criterion_tests: continue
        tests = generate_tests_for_criterion(criterion, rubric_criteria, model_override)
        if tests:
            criterion_tests[criterion] = tests
            print(f'  [{criterion}] {len(tests)} tests')
    if criterion_tests:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(criterion_tests, f, indent=2, ensure_ascii=False)

    # Phase 2: Run tests
    print(f'\n--- Phase 2: Running tests in Flask sandbox ---')
    results = run_all_tests(task_subs, criterion_tests, TEMPLATE_PATH)

    # Phase 3: Build FS
    print(f'\n--- Phase 3: Building FS ---')
    fs_list = build_fs_from_results(results, criterion_tests)
    print(f'  {len(fs_list)} FS')

    # Phase 4: Evaluate
    print(f'\n--- Phase 4: Blind test ---')
    all_readmes = load_all_readmes(os.path.join(BASE, 'submission'))
    _evaluate_test_fs(fs_list, all_readmes)

    return fs_list


if __name__ == '__main__':
    question_dir = sys.argv[1] if len(sys.argv)>=2 else os.path.join(BASE,'question')
    submission_dir = sys.argv[2] if len(sys.argv)>=3 else os.path.join(BASE,'submission')
    question_id = sys.argv[3] if len(sys.argv)>=4 else 'q1_iMusic'

    rubric_cache = os.path.join(BASE, 'output', question_id, 'rubric_cache.json')
    with open(rubric_cache, encoding='utf-8') as f:
        qc = json.load(f)
    tasks = qc.get('tasks', [])
    rubric_criteria = []
    for t in tasks: rubric_criteria.extend(t.get('rubric_criteria', []))

    task_subs = {}
    for tn in [1,2,3]:
        task_subs[tn] = collect_submissions_by_task(submission_dir, tn)
        print(f'  Task{tn}: {len(task_subs[tn])} submissions')

    fs_list = generate_test_fs_v2(tasks, task_subs, rubric_criteria)

    out_dir = os.path.join(BASE, 'output', question_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'fs_registry_test_v2.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'question': qc.get('question_name', question_id),
            'model': DEEPSEEK_MODEL,
            'pipeline': 'Test-Driven v2 (Flask test_client)',
            'total_fs': len(fs_list),
            'fs_registry': fs_list,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n  Output: {os.path.join(out_dir, "fs_registry_test_v2.json")}')
