"""
SQL Injection Pattern Post-Processor (v2)
==========================================
Scans student code for any existing SQL injection (f-string, concat, %), and injects
the missing variants (% formatting and .format()) if the student clearly has the
"string formatting for sql" bad pattern assigned.

Logic: if a student has ANY SQL injection variant (f-string or string concatenation),
they were assigned the "string formatting" bad pattern → inject missing variants.

Usage:
    python inject_sql_patterns.py [submissions_dir]
"""
import re
import sys
from pathlib import Path

# ── Detection patterns ──

FSTRING_SQL_RE = re.compile(r'f["\'][^"\']*\b(?:INSERT|UPDATE|DELETE|SELECT)\b', re.I)
CONCAT_SQL_RE = re.compile(r'\.execute\s*\(\s*["\'][^"\']*["\']\s*\+\s*str\s*\(', re.I)
PCT_FORMAT_SQL_RE = re.compile(r'\.execute\s*\(\s*["\'][^"\']*%[sd]\s*["\']\s*%\s*\(', re.I)
DOT_FORMAT_SQL_RE = re.compile(r'\.execute\s*\(\s*["\'][^"\']*\{[^}]*\}\s*["\']\s*\.\s*format\s*\(', re.I)

# Parameterized query → target for rewrite
PARAM_QUERY_RE = re.compile(
    r'(\w+\.execute\s*\()\s*(["\'])(.*?\?.*?)\2\s*,\s*\(([^)]+)\)\s*\)',
    re.DOTALL,
)


def has_pattern(code: str, pattern_re: re.Pattern) -> bool:
    clean = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    return bool(pattern_re.search(clean))


def student_has_sql_injection(task_files: list[Path]) -> bool:
    """Check if ANY task file contains SQL injection."""
    for fp in task_files:
        if not fp.exists():
            continue
        code = clean_code(fp.read_text(encoding='utf-8'))
        if any(has_pattern(code, r) for r in [FSTRING_SQL_RE, CONCAT_SQL_RE, PCT_FORMAT_SQL_RE, DOT_FORMAT_SQL_RE]):
            return True
    return False


def clean_code(code: str) -> str:
    """Strip comments."""
    return re.sub(r'#.*$', '', code, flags=re.MULTILINE)


def inject_pct_format(code: str) -> tuple[str, bool]:
    """Rewrite one parameterized query to use % formatting."""
    modified = False

    def _rewrite(match):
        nonlocal modified
        if modified:
            return match.group(0)
        prefix = match.group(1)
        quote = match.group(2)
        middle = match.group(3)
        params = match.group(4)
        param_list = [p.strip() for p in params.split(',')]
        n = len(param_list)
        if n >= 1:
            new_sql = re.sub(r'\?', '%s', middle, count=n)
            clean_params = ', '.join(param_list)
            new_query = f'{prefix}{quote}{new_sql}{quote} % ({clean_params}))'
            modified = True
            return new_query
        return match.group(0)

    if has_pattern(code, PCT_FORMAT_SQL_RE):
        return code, False
    return PARAM_QUERY_RE.sub(_rewrite, code), modified


def inject_dot_format(code: str) -> tuple[str, bool]:
    """Rewrite one parameterized query to use .format()."""
    modified = False

    def _rewrite(match):
        nonlocal modified
        if modified:
            return match.group(0)
        prefix = match.group(1)
        quote = match.group(2)
        middle = match.group(3)
        params = match.group(4)
        param_list = [p.strip() for p in params.split(',')]
        n = len(param_list)
        if n >= 1:
            new_sql = re.sub(r'\?', '{}', middle, count=n)
            clean_params = ', '.join(param_list)
            new_query = f'{prefix}{quote}{new_sql}{quote}.format({clean_params}))'
            modified = True
            return new_query
        return match.group(0)

    if has_pattern(code, DOT_FORMAT_SQL_RE):
        return code, False
    return PARAM_QUERY_RE.sub(_rewrite, code), modified


def update_readme(readme_path: Path, injections: list[str]):
    if not readme_path.exists():
        return
    content = readme_path.read_text(encoding='utf-8')
    # Remove existing injection note
    content = re.sub(r'\n\n---\n\n## Post-Processing Injections.*$', '', content, flags=re.DOTALL)
    note = '\n\n---\n\n## Post-Processing Injections\n\n'
    note += 'DeepSeek refused to write these patterns naturally. '
    note += 'They were deterministically injected by inject_sql_patterns.py:\n\n'
    for inj in injections:
        note += f'- **{inj}**\n'
    content += note
    readme_path.write_text(content, encoding='utf-8')


def process_submissions(submissions_dir: str):
    root = Path(submissions_dir)
    students = sorted(
        [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')],
        key=lambda d: d.name,
    )

    stats = {
        'total': len(students),
        'has_any_injection': 0,
        'pct_injected': 0,
        'dot_format_injected': 0,
        'already_had_pct': 0,
        'already_had_dot_format': 0,
        'no_param_query': 0,
        'total_files_modified': 0,
    }

    for student_dir in students:
        sid = student_dir.name
        task_files = [student_dir / f'task{tn}.py' for tn in [1, 3]]

        # Only process students who already have some SQL injection
        if not student_has_sql_injection(task_files):
            continue

        stats['has_any_injection'] += 1
        injections = []

        for fp in task_files:
            if not fp.exists():
                continue
            code = clean_code(fp.read_text(encoding='utf-8'))
            original = code

            # Inject % formatting if missing
            if has_pattern(code, PCT_FORMAT_SQL_RE):
                stats['already_had_pct'] += 1
            else:
                code, modified = inject_pct_format(code)
                if modified:
                    stats['pct_injected'] += 1
                    injections.append(
                        f'{fp.name}: % formatting SQL injection '
                        f'(e.g., `cursor.execute("...%s..." % (var,))`)'
                    )
                    print(f'  {sid}/{fp.name}: Injected % formatting SQL')

            # Inject .format() if missing
            if has_pattern(code, DOT_FORMAT_SQL_RE):
                stats['already_had_dot_format'] += 1
            else:
                code2, modified = inject_dot_format(code)
                if modified:
                    stats['dot_format_injected'] += 1
                    injections.append(
                        f'{fp.name}: .format() SQL injection '
                        f'(e.g., `cursor.execute("...{{}}...".format(var))`)'
                    )
                    print(f'  {sid}/{fp.name}: Injected .format() SQL')
                code = code2

            if code != original:
                stats['total_files_modified'] += 1
                fp.write_text(code, encoding='utf-8')

        if injections:
            update_readme(student_dir / 'README.md', injections)

    # Summary
    print(f'\n{"=" * 60}')
    print('  SQL INJECTION POST-PROCESSING (v2) — COMPLETE')
    print('=' * 60)
    print(f'  Total students: {stats["total"]}')
    print(f'  With existing SQL injection: {stats["has_any_injection"]}')
    print(f'  % formatting injected:        {stats["pct_injected"]}')
    print(f'  .format() injected:           {stats["dot_format_injected"]}')
    print(f'  Already had % formatting:     {stats["already_had_pct"]}')
    print(f'  Already had .format():        {stats["already_had_dot_format"]}')
    print(f'  Files modified:               {stats["total_files_modified"]}')
    print(f'  No param query to rewrite:    {stats["no_param_query"]}')

    return stats


if __name__ == '__main__':
    submissions_dir = sys.argv[1] if len(sys.argv) > 1 else 'submissions_imusic_v5'
    process_submissions(submissions_dir)
