"""
Config Auto-Generator
======================
Scans question/ and submission/ folders to auto-detect:
  - Task structure (single-file vs multi-file)
  - Target files per task
  - Target functions (stubs to be implemented)
  - Submission directory structure and naming
  - Reference file mapping

Outputs a YAML config ready for fs_extractor.py.

Usage:
    python config_generator.py <question_dir> <submissions_dir> [--question-id Q1] [--output config.yaml]
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Any


def scan_question_dir(question_path: str) -> dict:
    """
    Analyze a question directory to understand its structure.

    Returns:
        dict with:
          - python_files: list of .py files found
          - task_descriptions: list of .md/.txt/.pdf files
          - data_files: list of data files (csv, tsv, db, etc.)
          - template_dirs: list of template directories
          - starter_functions: {filename: [function names]} for stub/todo functions
    """
    info = {
        'python_files': [],
        'task_descriptions': [],
        'data_files': [],
        'template_dirs': [],
        'starter_functions': {},
    }

    if not os.path.isdir(question_path):
        return info

    for dirpath, dirnames, filenames in os.walk(question_path):
        rel = os.path.relpath(dirpath, question_path)

        # Find template directories
        if os.path.basename(dirpath) == 'templates':
            info['template_dirs'].append(rel)

        for fname in filenames:
            full = os.path.join(dirpath, fname)

            if fname.endswith('.py'):
                info['python_files'].append(os.path.join(rel, fname) if rel != '.' else fname)
                # Extract stub functions
                stubs = _find_stub_functions(full)
                if stubs:
                    key = os.path.join(rel, fname) if rel != '.' else fname
                    info['starter_functions'][key] = stubs

            elif fname.endswith(('.md', '.txt', '.pdf')):
                info['task_descriptions'].append(os.path.join(rel, fname) if rel != '.' else fname)

            elif any(fname.endswith(ext) for ext in ('.csv', '.tsv', '.db', '.sqlite', '.json', '.xml')):
                info['data_files'].append(os.path.join(rel, fname) if rel != '.' else fname)

    return info


def _detect_prefixes(student_dirs: list[str]) -> list[str]:
    """
    Auto-detect question-specific prefixes from student directory names.

    Uses the pattern: letter(s) + digit(s) + separator (e.g., 'q1-', 'Q2_', 'task3-')
    Groups directories by these prefixes to identify distinct questions.

    For mixed dirs: q1-excellent-a1, q2-borderline-a1 -> ['q1-', 'q2-']
    For flat dirs:   S001, S002, S003                -> ['S']
    For no pattern:  student_a, student_b             -> []
    """
    import re
    if len(student_dirs) < 2:
        return []

    # Extract top-level directory name
    tops = [d.replace('\\', '/').split('/')[0] for d in student_dirs]

    # Strategy: find all unique prefixes matching <letters><digits><separator>
    # e.g., 'q1-', 'Q2_', 'task3-', 'S'
    prefix_counts: dict[str, int] = {}
    for top in tops:
        m = re.match(r'^([a-zA-Z]+\d+)[-_]', top)
        if m:
            prefix_counts[m.group(1) + '-'] = prefix_counts.get(m.group(1) + '-', 0) + 1
        else:
            # Try just letter prefix (S001 -> S)
            m = re.match(r'^([a-zA-Z]+)\d', top)
            if m:
                prefix_counts[m.group(1)] = prefix_counts.get(m.group(1), 0) + 1

    # A prefix is valid if it matches >= 2 students and < 80% of all students
    # (the 80% rule avoids treating a universal prefix as question-specific)
    total = len(tops)
    valid = []
    for prefix, count in sorted(prefix_counts.items()):
        if count >= 2 and count < total * 0.8:
            valid.append(prefix)

    # If only one question detected (no mix), return empty -- no filter needed
    if len(valid) <= 1:
        return []

    return sorted(valid)


def scan_submissions_dir(submissions_path: str) -> dict:
    """
    Analyze a submissions directory to understand its structure.

    Returns:
        dict with:
          - student_count: number of student directories
          - student_ids: list of student directory names
          - file_structure: 'single' (one file per student) or 'multi' (task1.py + task2.py + ...)
          - common_files: list of filenames found across students
          - depth_pattern: typical nesting depth (0=flat, 1=one subfolder, etc.)
    """
    info = {
        'student_count': 0,
        'student_ids': [],
        'file_structure': 'unknown',
        'common_files': {},
        'depth_pattern': 0,
        'sample_student': None,
    }

    if not os.path.isdir(submissions_path):
        return info

    # Collect all student dirs (any dir that contains .py files)
    student_dirs = []
    file_counter = {}

    for dirpath, dirnames, filenames in os.walk(submissions_path, topdown=True):
        # Skip __pycache__ etc.
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']

        py_files = [f for f in filenames if f.endswith('.py')]
        if py_files:
            rel = os.path.relpath(dirpath, submissions_path)
            if rel == '.':
                continue  # Skip root level
            student_dirs.append(rel)
            for f in py_files:
                file_counter[f] = file_counter.get(f, 0) + 1

    info['student_count'] = len(student_dirs)
    info['student_ids'] = sorted(student_dirs)[:20]  # First 20 for reference

    if student_dirs:
        info['sample_student'] = student_dirs[0]

    # Determine file structure
    if file_counter:
        # Sort by frequency
        sorted_files = sorted(file_counter.items(), key=lambda x: -x[1])
        info['common_files'] = {f: count for f, count in sorted_files}

        # Detect structure type
        names = list(file_counter.keys())
        if any('task' in f.lower() for f in names):
            info['file_structure'] = 'multi'  # task1.py, task2.py, etc.
        elif len(names) == 1:
            info['file_structure'] = 'single'  # one file per student
        else:
            info['file_structure'] = 'multi'

    # Detect depth pattern
    if student_dirs:
        first = student_dirs[0]
        info['depth_pattern'] = first.count(os.sep)

    # Auto-detect student prefix(es) for question filtering
    info['detected_prefixes'] = _detect_prefixes(student_dirs)

    return info


def generate_config(
    question_path: str,
    submissions_path: str,
    question_id: str = 'auto_detected',
    question_name: str = 'Auto-detected Question',
    references_path: str = '',
) -> dict:
    """
    Generate a complete config dict by scanning question and submission folders.

    Args:
        question_path: Path to the question directory.
        submissions_path: Path to the submissions directory.
        question_id: Identifier for this question.
        question_name: Human-readable name.
        references_path: Path to reference solutions (empty = auto-detect).

    Returns:
        Config dict ready to be written as YAML.
    """
    q_info = scan_question_dir(question_path)
    s_info = scan_submissions_dir(submissions_path)

    config = {
        'question_id': question_id,
        'question_name': question_name,
        'references_path': references_path or f'references/{question_id}/',
        'submissions_path': os.path.relpath(submissions_path, os.path.dirname(question_path))
            if os.path.isabs(submissions_path) else submissions_path,
        'submission_file_pattern': None,
        'schema': _generate_schema_hint(q_info),
        'tasks': {},
        '_auto_generated': True,
        '_scan_info': {
            'student_count': s_info['student_count'],
            'file_structure': s_info['file_structure'],
            'depth_pattern': s_info['depth_pattern'],
            'common_files': s_info['common_files'],
        },
    }

    # Detect task files from submissions
    task_files = _detect_task_files(s_info)

    if not task_files and q_info['starter_functions']:
        # Fall back to starter code analysis
        task_files = list(q_info['starter_functions'].keys())

    if not task_files:
        # Last resort: look for python files in question
        task_files = q_info['python_files']

    # Build task entries
    for i, target_file in enumerate(task_files, 1):
        task_id = f'Task{i}'

        # Flexible lookup: match filename regardless of path prefix
        task_functions = []
        for sf_path, sf_funcs in q_info['starter_functions'].items():
            if os.path.basename(sf_path) == os.path.basename(target_file):
                task_functions = sf_funcs
                break

        # If no stub functions found, try to infer from file content
        if not task_functions:
            # Try reading from question starter code
            full_path = os.path.join(question_path, target_file)
            if os.path.exists(full_path):
                all_funcs = _find_all_functions(full_path)
                framework_funcs = {'index', 'main', 'page_not_found', 'upload_route'}
                task_functions = [f for f in all_funcs if f not in framework_funcs]

        # Still no functions? Sample from ALL submissions and pick the most common names
        if not task_functions:
            func_counter = {}
            sample_count = 0
            for student_dir in s_info.get('student_ids', [])[:10]:  # Sample first 10
                sample_path = os.path.join(submissions_path, student_dir, target_file)
                if os.path.exists(sample_path):
                    all_funcs = _find_all_functions(sample_path)
                    skip = {'app', 'main', '__name__'}
                    valid = [f for f in all_funcs if f not in skip and not f.startswith('_')]
                    for f in valid:
                        func_counter[f] = func_counter.get(f, 0) + 1
                    sample_count += 1

            # Pick functions that appear in at least 1 submission,
            # preferring longer names (more descriptive) over single-letter names
            if func_counter:
                # Sort by: frequency desc, then name length desc
                sorted_funcs = sorted(func_counter.items(),
                                     key=lambda x: (-x[1], -len(x[0])))
                task_functions = [f for f, _ in sorted_funcs][:5]  # Top 5

        # Generate rubric criteria dynamically from student code analysis
        rubric = _generate_rubric_from_code(
            submissions_path, target_file, task_id, task_functions,
            s_info.get('student_ids', [])
        )

        config['tasks'][task_id] = {
            'description': f'Auto-detected task from {target_file}',
            'target_file': target_file,
            'target_functions': task_functions,
            'rubric_criteria': rubric,
            'reference_files': [
                f'task{i}_vA.py',
                f'task{i}_vB.py',
            ],
        }

    return config


def _find_stub_functions(filepath: str) -> list[str]:
    """
    Find functions that are stubs (contain only 'pass' or TODO comments).
    These are the target functions students need to implement.
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        source = f.read()

    stubs = []

    # Method 1: AST analysis
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if body is just 'pass'
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    stubs.append(node.name)
                # Check if body is just a string constant (unimplemented with docstring)
                elif (len(node.body) == 2
                      and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)
                      and isinstance(node.body[0].value.value, str)
                      and isinstance(node.body[1], ast.Pass)):
                    stubs.append(node.name)
    except SyntaxError:
        pass

    # Method 2: Regex fallback for stub detection
    stub_pattern = re.findall(
        r'def\s+(\w+)\s*\([^)]*\)[^:]*:\s*\n\s*(?:#\s*TODO|pass|\.\.\.)',
        source
    )
    for name in stub_pattern:
        if name not in stubs:
            stubs.append(name)

    # Also detect functions that contain TODO in the body
    funcs = re.findall(r'def\s+(\w+)\s*\(', source)
    for func_name in funcs:
        # Find the function body (stop at next def or dedent to column 0)
        body_match = re.search(
            rf'def\s+{func_name}\s*\([^)]*\)[^:]*:\s*\n((?:(?:    |\t)[^\n]*\n?)*)',
            source
        )
        if body_match:
            body = body_match.group(1)
            if 'TODO' in body and func_name not in stubs:
                stubs.append(func_name)

    return stubs


def _find_all_functions(filepath: str) -> list[str]:
    """Extract all function names from a Python file."""
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
        return [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith('_')  # Skip private/dunder methods
        ]
    except SyntaxError:
        return re.findall(r'def\s+(\w+)\s*\(', source)


def _detect_task_files(s_info: dict) -> list[str]:
    """
    Detect which files correspond to tasks based on submission structure.

    For multi-file submissions (task1.py, task2.py, task3.py), returns them sorted.
    For single-file submissions, returns the single filename.
    """
    common = s_info.get('common_files', {})

    if not common:
        return []

    # Sort files by name to get task1.py, task2.py, task3.py in order
    task_files = sorted(common.keys())

    # Filter to likely task files (exclude __init__.py, etc.)
    task_files = [
        f for f in task_files
        if not f.startswith('_') and f.endswith('.py')
    ]

    return task_files


def _generate_schema_hint(q_info: dict) -> dict:
    """
    Generate a minimal schema hint from question analysis.
    This is used by the comparator for table/column validation.
    """
    schema = {
        'expected_tables': [],
        'expected_columns': {},
    }

    # Try to detect SQL CREATE TABLE statements from starter code
    for py_file in q_info.get('python_files', []):
        # This would parse SQL from Python strings -- simplified for now
        pass

    return schema


def _generate_rubric_from_code(
    submissions_path: str,
    target_file: str,
    task_id: str,
    target_functions: list[str],
    student_ids: list[str],
    max_samples: int = 15,
) -> list[dict]:
    """Generate rubric criteria by analysing student code with AST.

    Scans student submissions to discover:
      - High-frequency library/API calls (grouped by inferred module)
      - Structural patterns: error handling, file I/O, iteration, validation
      - Domain-specific idioms used across students

    Produces criteria that are specific to the actual programming patterns
    found in the submissions, rather than generic placeholders.

    The approach mirrors Phase 3.1C of the AST refinement architecture:
    extract call signatures from code, group by category, emit criteria.

    Args:
        submissions_path: Root of the submissions directory.
        target_file: The target .py filename for this task.
        task_id: Task identifier ('Task1', 'Task2', …).
        target_functions: Functions that students were asked to implement.
        student_ids: Relative paths of student directories.
        max_samples: Maximum number of student submissions to analyse.

    Returns:
        List of rubric criterion dicts, each with id, name, good_patterns,
        bad_patterns, and max_points.
    """
    from collections import Counter

    if not student_ids:
        return _fallback_rubric(task_id)

    # --- Collect API tokens from sampled student code ---
    _builtins = {'print', 'len', 'range', 'int', 'str', 'float', 'list',
                 'dict', 'set', 'tuple', 'bool', 'type', 'isinstance',
                 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                 'open', 'input', 'format', 'join', 'split', 'strip',
                 'append', 'get', 'keys', 'values', 'items', 'update',
                 'super', '__init__', '__name__', '__main__',
                 'replace', 'lower', 'upper', 'startswith', 'endswith',
                 'read', 'write', 'close', 'readlines', 'readline'}

    call_counter: Counter = Counter()
    import_counter: Counter = Counter()
    has_error_handling = False
    has_file_io = False
    has_iteration = False
    has_validation = False
    sample_count = 0

    for sid in student_ids[:max_samples]:
        fpath = os.path.join(submissions_path, sid, target_file)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                source = fh.read()
            tree = ast.parse(source)
            sample_count += 1

            for node in ast.walk(tree):
                # Track imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_counter[alias.name.split('.')[0]] += 1
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_counter[node.module.split('.')[0]] += 1

                # Track function/method calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                        if name not in _builtins:
                            call_counter[name] += 1
                    elif isinstance(node.func, ast.Attribute):
                        call_counter[node.func.attr] += 1

                # Detect structural patterns
                if isinstance(node, ast.Try):
                    has_error_handling = True
                if isinstance(node, ast.With):
                    has_file_io = True
                if isinstance(node, (ast.For, ast.While)):
                    has_iteration = True
                # Heuristic: if-condition that checks truthiness of a
                # variable likely obtained from request / input → validation
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.Call):
                        has_validation = True
        except (SyntaxError, UnicodeDecodeError):
            continue

    if sample_count == 0:
        return _fallback_rubric(task_id)

    # --- Build criteria from discovered patterns ---
    criteria: list[dict] = []
    cid = int(task_id.replace('Task', '') or '1')
    n = 0

    # Criterion 1: core implementation (always present — from target functions)
    func_names = ', '.join(target_functions[:3]) if target_functions else 'the required functions'
    criteria.append({
        'id': f'{cid}_{n+1}',
        'name': f'{task_id} core implementation: {func_names}',
        'good_patterns': target_functions[:5] if target_functions else [],
        'bad_patterns': ['missing implementation', 'empty function body', 'pass'],
        'max_points': 4,
    })
    n += 1

    # Criterion 2: library/API usage (from detected imports and calls)
    top_imports = [lib for lib, _ in import_counter.most_common(4)
                   if lib not in ('flask', 'os', 'sys', 're', 'json', 'datetime', 'math')]
    top_calls = [call for call, _ in call_counter.most_common(6)
                 if call not in _builtins]

    if top_imports or top_calls:
        good_examples = (top_calls[:4] if top_calls else []) + (top_imports[:2] if top_imports else [])
        criteria.append({
            'id': f'{cid}_{n+1}',
            'name': f'{task_id} correct library/API usage',
            'good_patterns': good_examples[:6] if good_examples else [],
            'bad_patterns': ['missing required library calls', 'hardcoded values instead of API calls'],
            'max_points': 3,
        })
        n += 1

    # Criterion 3: structural quality patterns (only when detected)
    sub_criteria = []
    if has_error_handling:
        sub_criteria.append('try/except error handling')
    if has_file_io:
        sub_criteria.append('with-statement for resource management')
    if has_iteration:
        sub_criteria.append('loop for iteration over data')
    if has_validation:
        sub_criteria.append('input validation checks')

    if sub_criteria:
        criteria.append({
            'id': f'{cid}_{n+1}',
            'name': f'{task_id} code robustness and structure',
            'good_patterns': sub_criteria[:4],
            'bad_patterns': ['no error handling for fallible operations',
                             'resources not closed', 'no input validation'],
            'max_points': 3,
        })
        n += 1

    # If we couldn't detect much, add a generic catch-all
    if len(criteria) < 2:
        criteria.append({
            'id': f'{cid}_{n+1}',
            'name': f'{task_id} code quality',
            'good_patterns': ['clear variable names', 'appropriate data types'],
            'bad_patterns': ['redundant code', 'incorrect data type usage'],
            'max_points': 2,
        })

    return criteria


def _fallback_rubric(task_id: str) -> list[dict]:
    """Minimal fallback rubric when no student code is available for analysis."""
    cid = int(task_id.replace('Task', '') or '1')
    return [
        {'id': f'{cid}_1', 'name': f'{task_id} implementation correctness',
         'good_patterns': [], 'bad_patterns': [], 'max_points': 4},
        {'id': f'{cid}_2', 'name': f'{task_id} code quality and patterns',
         'good_patterns': [], 'bad_patterns': [], 'max_points': 3},
    ]


def save_config(config: dict, output_path: str):
    """Save config dict as YAML file."""
    import yaml

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write a header comment
        f.write(f'# Auto-generated config for {config["question_id"]}\n')
        f.write(f'# Generated by config_generator.py\n')
        f.write(f'# Students detected: {config["_scan_info"]["student_count"]}\n')
        f.write(f'# File structure: {config["_scan_info"]["file_structure"]}\n')
        f.write('\n')

        # Remove internal fields before YAML dump
        clean = {k: v for k, v in config.items() if not k.startswith('_')}
        yaml.dump(clean, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f'Config saved to: {output_path}')


def run(
    question_dir: str,
    submissions_dir: str,
    question_id: str = 'auto_detected',
    question_name: str = 'Auto-detected Question',
    output: str = 'config/auto_generated.yaml',
):
    """
    Main entry point for config generation.

    Args:
        question_dir: Path to the question folder.
        submissions_dir: Path to the submissions folder.
        question_id: Question identifier.
        question_name: Human-readable name.
        output: Output YAML path.
    """
    print('=' * 60)
    print('  CONFIG AUTO-GENERATOR')
    print('=' * 60)
    print()

    # Analyze folders
    print(f'Scanning question directory: {question_dir}')
    q_info = scan_question_dir(question_dir)
    print(f'  Python files: {q_info["python_files"]}')
    print(f'  Task descriptions: {q_info["task_descriptions"]}')
    print(f'  Starter stubs: {q_info["starter_functions"]}')
    print()

    print(f'Scanning submissions directory: {submissions_dir}')
    s_info = scan_submissions_dir(submissions_dir)
    print(f'  Students found: {s_info["student_count"]}')
    print(f'  File structure: {s_info["file_structure"]}')
    print(f'  Common files: {s_info["common_files"]}')
    print(f'  Sample student: {s_info["sample_student"]}')
    print()

    # Generate config
    print('Generating config...')
    config = generate_config(
        question_dir, submissions_dir,
        question_id=question_id,
        question_name=question_name,
    )

    # Show detected tasks
    for task_id, task_cfg in config['tasks'].items():
        print(f'  {task_id}: {task_cfg["target_file"]} -> {task_cfg["target_functions"]}')

    # Save
    save_config(config, output)

    return config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Auto-generate FS extraction config')
    parser.add_argument('question_dir', help='Path to question directory')
    parser.add_argument('submissions_dir', help='Path to submissions directory')
    parser.add_argument('--question-id', default='auto_detected', help='Question ID')
    parser.add_argument('--question-name', default='Auto-detected Question', help='Question name')
    parser.add_argument('--output', default=None, help='Output YAML path')

    args = parser.parse_args()

    output = args.output
    if output is None:
        qid = args.question_id.replace(' ', '_').lower()
        output = f'config/{qid}.yaml'

    run(
        args.question_dir,
        args.submissions_dir,
        question_id=args.question_id,
        question_name=args.question_name,
        output=output,
    )
