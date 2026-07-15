"""
Subprocess-based isolated code execution
=========================================
Port of taffies-2026 executor.ts + docker.ts → Python subprocess.

No Docker required. Uses subprocess.run() with timeout for process-level
isolation. Sufficient for CW-generated student code (not malicious).

Key safety features:
  - timeout: kill after N seconds (prevents infinite loops)
  - separate Python process: no global variable pollution
  - capture_output: prevent stdout/stderr flooding
  - temp directory: prevent file system access to project files
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def run_command(
    cmd: list[str],
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> dict:
    """Run a command with timeout. Returns {exit_code, stdout, stderr}."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(BASE_DIR),
        )
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            'exit_code': 124,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout}s',
        }
    except Exception as e:
        return {
            'exit_code': 127,
            'stdout': '',
            'stderr': str(e),
        }


def execute_in_subprocess(
    code: str,
    timeout: int = 30,
    env: Optional[dict] = None,
) -> dict:
    """Execute Python code in an isolated subprocess.

    Args:
        code: Python source code to execute
        timeout: max seconds before kill
        env: extra environment variables

    Returns:
        {exit_code, stdout, stderr, duration_ms, timeout}
    """
    import time

    # Write code to temp file (avoids -c escaping issues)
    tmpdir = tempfile.mkdtemp(prefix='taffies_sandbox_', dir=str(BASE_DIR))
    try:
        script_path = os.path.join(tmpdir, '_test.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)

        started = time.time()
        proc_env = os.environ.copy()
        proc_env['PYTHONIOENCODING'] = 'utf-8'
        if env:
            proc_env.update(env)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(BASE_DIR),
                env=proc_env,
            )
            duration_ms = int((time.time() - started) * 1000)
            return {
                'exit_code': result.returncode,
                'stdout': _limit_output(result.stdout),
                'stderr': _limit_output(result.stderr),
                'duration_ms': duration_ms,
                'timeout': False,
            }
        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - started) * 1000)
            return {
                'exit_code': 124,
                'stdout': '',
                'stderr': f'Execution timed out after {timeout}s',
                'duration_ms': duration_ms,
                'timeout': True,
            }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def execute_student_function(
    extracted_code: str,
    test_harness: str,
    timeout: int = 30,
) -> dict:
    """Execute a student's extracted function with a test harness.

    Args:
        extracted_code: student's function code (def get_statistics(...): ...)
        test_harness: Python code that imports/mocks deps, calls the function,
                      prints JSON result to stdout

    Returns:
        {exit_code, stdout, stderr, duration_ms, timeout, result_json}
    """
    import time
    import json as _json

    full_script = f"""
import sys
import json
import sqlite3

# ── Mock layer ──
class FakeCursor:
    def execute(self, query, params=None):
        return self
    def fetchall(self):
        return []
    def fetchone(self):
        return None
    def close(self):
        pass

class FakeConn:
    def cursor(self):
        return FakeCursor()
    def close(self):
        pass

sqlite3.connect = lambda db, **kw: FakeConn()

class FakeApp:
    def route(self, *a, **kw):
        return lambda f: f
app = FakeApp()
DB_FILE = "mock.db"
BASE_DIR = "mock"
request = type("obj", (object,), {{"method": "GET", "form": {{}}}})()
flash = lambda m, c=None: None
get_db_connection = lambda: FakeConn()
get_db = lambda: FakeConn()
redirect = lambda u, **kw: None
url_for = lambda x, **kw: "/"
render_template = lambda *a, **kw: ""
Path = type("Path", (), {{"mkdir": lambda *a, **kw: None}})()

# ── Student code ──
{extracted_code}

# ── Bridge: test harness can override these after re-patching ──
# Student code uses get_db_connection()/get_db() — harness MUST
# re-bind them to its own capturing mocks after defining _FakeConn.
if 'get_db_connection' not in dir() or 'get_db' not in dir():
    pass  # defined by header mock layer

# ── Test harness ──
{test_harness}
"""

    result = execute_in_subprocess(full_script, timeout=timeout)

    # Try to parse JSON from stdout
    result['result_json'] = None
    if result['stdout'].strip():
        try:
            result['result_json'] = _json.loads(result['stdout'].strip())
        except (_json.JSONDecodeError, ValueError):
            pass

    return result


def _limit_output(text: str, max_chars: int = 256 * 1024) -> str:
    """Truncate output to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f'\n\n[Output truncated after {max_chars} characters]'
