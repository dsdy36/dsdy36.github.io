"""
Behavioral Test Generator
==========================
Generates test harnesses for each criterion — either manually (for well-understood
criteria) or via AI (for new criteria).

Each test harness is Python code that:
  1. Calls the student's function with controlled inputs
  2. Observes behavior (queries generated, exceptions, return values)
  3. Prints JSON: {"vulnerable": bool, "details": {...}}

The harness runs INSIDE the subprocess, with the student's code already injected
and sqlite3 mocked to capture queries.
"""

import os
import json
import sys
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# ============================================================
# Manual test harnesses (reliable, no AI needed)
# ============================================================

# RQ2_3: ORDER BY validation
# Tests whether get_statistics() validates sort_column before SQL interpolation
RQ23_TEST = r"""
import re
captured_queries = []

# Re-patch to capture queries
import sqlite3 as _sq
_original_connect = _sq.connect
class _FakeCursor:
    def execute(self, query, params=None):
        captured_queries.append({'query': query, 'params': params})
        return self
    def fetchall(self):
        return []
    def fetchone(self):
        return None
    def close(self):
        pass
class _FakeConn:
    def cursor(self):
        return _FakeCursor()
    def close(self):
        pass
_sq.connect = lambda db, **kw: _FakeConn()
# Re-bind helpers that student code may use
get_db_connection = lambda: _FakeConn()
get_db = lambda: _FakeConn()

malicious_col = "1; DROP TABLE Playlist--"
try:
    result = get_statistics(1, malicious_col, "ASC")
    result_ok = True
except Exception as e:
    result_ok = False
    error_msg = f"{type(e).__name__}: {e}"

# Check if injection succeeded
vulnerable = False
for q in captured_queries:
    sql = q.get('query', '')
    if 'DROP TABLE' in sql.upper() or '1;' in sql:
        vulnerable = True
        break

# Also check: did function default to safe column?
used_safe_default = False
for q in captured_queries:
    sql = q.get('query', '')
    if 'ORDER BY' in sql.upper():
        if 'PlaylistName' in sql or 'p.Name' in sql or 'NumberOfTracks' in sql:
            if malicious_col not in sql:
                used_safe_default = True
        break

print(json.dumps({
    "vulnerable": vulnerable,
    "result_ok": result_ok,
    "used_safe_default": used_safe_default,
    "query_count": len(captured_queries),
    "queries": [q['query'][:200] for q in captured_queries],
}))
"""

# RQ2_1: Parameterized queries (SQL injection in INSERT/WHERE)
# Tests whether get_all_genres() or get_statistics() uses parameterized queries
RQ21_TEST = r"""
import re
captured_queries = []

import sqlite3 as _sq
class _FakeCursor:
    def execute(self, query, params=None):
        captured_queries.append({'query': query, 'params': params})
        return self
    def fetchall(self):
        return [{'GenreId': 1, 'Name': 'Rock'}, {'GenreId': 2, 'Name': 'Pop'}]
    def fetchone(self):
        return None
    def close(self):
        pass
class _FakeConn:
    def cursor(self):
        return _FakeCursor()
    def close(self):
        pass
_sq.connect = lambda db, **kw: _FakeConn()
get_db_connection = lambda: _FakeConn()
get_db = lambda: _FakeConn()

# Test get_all_genres if available, otherwise test get_statistics
vulnerable = False
details = {}

# Check for f-string / string formatting in SQL
# Pattern 1: f"...{var}..." inside execute()
# Pattern 2: "..." + var + "..." inside execute()
# Pattern 3: "..." % var inside execute()
has_fstring_sql = False
has_concat_sql = False
has_percent_sql = False

# We detect this by checking the SOURCE CODE, not the captured queries
# (because the mock doesn't actually execute SQL)
import inspect
try:
    source = inspect.getsource(get_all_genres)
    if re.search(r'f["\x27].*\{.*\}', source, re.DOTALL):
        has_fstring_sql = True
    if re.search(r'\+.*\+', source):
        has_concat_sql = True
    if re.search(r'%\s*\(', source):
        has_percent_sql = True
except Exception:
    pass

try:
    source2 = inspect.getsource(get_statistics)
    if re.search(r'f["\x27].*\{.*\}', source2, re.DOTALL):
        has_fstring_sql = True
    if re.search(r'\+\s*["\x27]', source2):
        has_concat_sql = True
except Exception:
    pass

# Also check captured queries for parameterized vs literal
has_params = any(q.get('params') is not None and len(q.get('params', []) or []) > 0
                 for q in captured_queries)
has_no_params_sql = any(q.get('params') is None and q.get('query', '').strip()
                        for q in captured_queries)

# A function is vulnerable if it uses dynamic SQL without parameterization
vulnerable = (has_fstring_sql or has_concat_sql or has_percent_sql) and not has_params

print(json.dumps({
    "vulnerable": vulnerable,
    "has_fstring_sql": has_fstring_sql,
    "has_concat_sql": has_concat_sql,
    "has_params": has_params,
    "query_count": len(captured_queries),
}))
"""

# RQ3_1: Sort playlists by name
# Tests whether get_all_playlists() returns playlists sorted by name
RQ31_TEST = r"""
import re
captured_queries = []

import sqlite3 as _sq
class _FakeCursor:
    def execute(self, query, params=None):
        captured_queries.append({'query': query, 'params': params})
        return self
    def fetchall(self):
        return [
            {'PlaylistId': 3, 'Name': 'Chill'},
            {'PlaylistId': 1, 'Name': 'Rock Classics'},
            {'PlaylistId': 2, 'Name': 'Workout'},
        ]
    def fetchone(self):
        return None
    def close(self):
        pass
class _FakeConn:
    def cursor(self):
        return _FakeCursor()
    def close(self):
        pass
_sq.connect = lambda db, **kw: _FakeConn()
get_db_connection = lambda: _FakeConn()
get_db = lambda: _FakeConn()

try:
    result = get_all_playlists()
    result_ok = True
except Exception as e:
    result_ok = False
    result = None
    error_msg = f"{type(e).__name__}: {e}"

# Check if ORDER BY Name is in the query
has_order_by_name = False
for q in captured_queries:
    sql = q.get('query', '')
    if re.search(r'ORDER\s+BY\s+.*Name', sql, re.IGNORECASE):
        has_order_by_name = True
        break

# Check if result is sorted by name
sorted_correctly = False
if result_ok and isinstance(result, list) and len(result) > 1:
    names = [r.get('Name', '') for r in result if isinstance(r, dict)]
    sorted_correctly = (names == sorted(names))

print(json.dumps({
    "vulnerable": not has_order_by_name,
    "has_order_by_name": has_order_by_name,
    "sorted_correctly": sorted_correctly,
    "result_ok": result_ok,
    "result_count": len(result) if isinstance(result, list) else 0,
    "query_count": len(captured_queries),
}))
"""

# RQ1_3: try/except IntegrityError
# Tests whether update_playlist_tracks() catches IntegrityError
RQ13_TEST = r"""
import re
captured_queries = []

import sqlite3 as _sq
_execute_count = [0]
class _FakeCursor:
    def execute(self, query, params=None):
        captured_queries.append({'query': query, 'params': params})
        _execute_count[0] += 1
        # Simulate IntegrityError on 3rd INSERT
        if _execute_count[0] == 3 and 'INSERT' in query.upper():
            raise _sq.IntegrityError('UNIQUE constraint failed')
        return self
    def fetchall(self):
        return []
    def fetchone(self):
        if _execute_count[0] <= 2:
            return None  # PlaylistId/TrackId exist
        return None
    def close(self):
        pass
class _FakeConn:
    def cursor(self):
        return _FakeCursor()
    def commit(self):
        pass
    def close(self):
        pass
_sq.connect = lambda db, **kw: _FakeConn()
_sq.IntegrityError = _sq.IntegrityError
get_db_connection = lambda: _FakeConn()
get_db = lambda: _FakeConn()

# Check source code for try/except IntegrityError
import inspect
has_try_except_integrity = False
try:
    source = inspect.getsource(update_playlist_tracks)
    has_try_except_integrity = bool(
        re.search(r'try\s*:', source) and
        re.search(r'IntegrityError', source)
    )
except Exception:
    pass

# Try to execute — if IntegrityError is caught, function returns normally
try:
    result = update_playlist_tracks('mock_path.tsv')
    result_ok = True
    exception_caught = True  # If we get here, IntegrityError was handled
except _sq.IntegrityError:
    result_ok = False
    exception_caught = False
except Exception as e:
    result_ok = False
    exception_caught = ('IntegrityError' not in str(type(e).__name__))

print(json.dumps({
    "vulnerable": not has_try_except_integrity,
    "has_try_except_integrity": has_try_except_integrity,
    "result_ok": result_ok,
    "exception_caught": exception_caught,
    "query_count": len(captured_queries),
}))
"""

# ============================================================
# Test registry: criterion → test harness
# ============================================================

MANUAL_TESTS: dict[str, dict] = {
    'RQ2_3': {
        'name': 'ORDER BY validation',
        'test_harness': RQ23_TEST,
        'target_func': 'get_statistics',
        'target_file': 'task2.py',
        'task': 'Task2',
        'description': 'Tests whether sort_column is validated before SQL interpolation',
        'malicious_input': '1; DROP TABLE Playlist--',
        'behavior_key': 'vulnerable',  # True = bad (no validation)
    },
    'RQ2_1': {
        'name': 'Parameterized queries',
        'test_harness': RQ21_TEST,
        'target_func': 'get_all_genres',  # also checks get_statistics
        'target_file': 'task2.py',
        'task': 'Task2',
        'description': 'Tests whether SQL queries use parameterized placeholders',
        'behavior_key': 'vulnerable',
    },
    'RQ3_1': {
        'name': 'Sort playlists by name',
        'test_harness': RQ31_TEST,
        'target_func': 'get_all_playlists',
        'target_file': 'task3.py',
        'task': 'Task3',
        'description': 'Tests whether query includes ORDER BY Name',
        'behavior_key': 'has_order_by_name',  # False = bad
    },
    'RQ1_3': {
        'name': 'try/except IntegrityError',
        'test_harness': RQ13_TEST,
        'target_func': 'update_playlist_tracks',
        'target_file': 'task1.py',
        'task': 'Task1',
        'description': 'Tests whether IntegrityError is caught with try/except',
        'behavior_key': 'vulnerable',  # True = bad (no try/except)
    },
}


def get_test_for_criterion(criterion: str) -> dict | None:
    """Get the test harness for a criterion (manual or None if not available)."""
    return MANUAL_TESTS.get(criterion)


def get_testable_criteria() -> list[str]:
    """List all criteria that have manual test harnesses."""
    return list(MANUAL_TESTS.keys())


def generate_ai_test(
    criterion: str,
    rubric_criteria: list[dict],
    ref_code: str,
    template_code: str,
) -> dict | None:
    """Generate a test harness via AI for a criterion not covered by manual tests.

    Uses DeepSeek to write a Python test script that can evaluate any student's
    code for this criterion.

    Returns:
        {name, test_harness, target_func, target_file, task, description,
         behavior_key} or None on failure
    """
    from ai_pipeline import call_deepseek, extract_json

    crit_info = _find_criterion(criterion, rubric_criteria)

    prompt = f"""You are an expert testing engineer. Write a Python test script that
evaluates whether a student's code CORRECTLY implements a grading criterion.

## Criterion: {criterion} — {crit_info.get('name', criterion)}
### Good patterns (correct implementation):
{chr(10).join(f'- {p}' for p in crit_info.get('good_patterns', []))}

### Bad patterns (incorrect implementation):
{chr(10).join(f'- {p}' for p in crit_info.get('bad_patterns', []))}

## Reference solutions (correct):
```python
{ref_code[:3000]}
```

## Template code (stub):
```python
{template_code[:2000]}
```

## Task
Write a Python test harness that runs INSIDE a sandbox where:
1. sqlite3.connect() is ALREADY MOCKED to capture queries
2. Flask dependencies (app, flash, redirect, url_for, render_template) are stubbed
3. The student's function code is ALREADY injected above your test code
4. You call the student's function with test inputs
5. You print ONE JSON line to stdout with results

Your test must:
- Call the TARGET function with BOTH valid and malicious/invalid inputs
- Detect whether the function correctly validates/handles bad input
- Print JSON: {{"vulnerable": true/false, "details": {{...}}}}
  where "vulnerable": true means the code is INCORRECT
- Handle exceptions gracefully (student code may crash)

## Constraints
- Use only stdlib modules (json, sys, re, inspect, sqlite3)
- The mock layer provides: FakeCursor, FakeConn, sqlite3.connect = lambda...
- Use `captured_queries` to observe SQL generated by student code
- Maximum 50 lines of test code

Output ONLY valid JSON:
{{"test_harness": "import re\\nimport json\\n...", "target_func": "function_name",
  "behavior_key": "vulnerable", "description": "what this tests"}}
"""

    try:
        response = call_deepseek(
            "You are an expert testing engineer. Output ONLY valid JSON.",
            prompt, temperature=0.3
        )
        if not response:
            return None
        data = extract_json(response)
        return {
            'name': f'{criterion} behavioral test',
            'test_harness': data.get('test_harness', ''),
            'target_func': data.get('target_func', ''),
            'target_file': '',  # Will be filled by caller
            'task': '',         # Will be filled by caller
            'description': data.get('description', ''),
            'behavior_key': data.get('behavior_key', 'vulnerable'),
        }
    except Exception as e:
        print(f'    AI test generation failed for {criterion}: {e}')
        return None


def _find_criterion(criterion: str, rubric_criteria: list[dict]) -> dict:
    for rc in rubric_criteria:
        if rc.get('id') == criterion:
            return rc
    return {}
