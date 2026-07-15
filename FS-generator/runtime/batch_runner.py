"""
Batch Runner + Behavioral Clusterer
====================================
Runs behavioral tests against all students and clusters them by observed behavior.

Flow:
  1. For each testable criterion, extract student functions
  2. Execute in subprocess with criterion-specific test harness
  3. Collect behavioral fingerprints
  4. Cluster students by behavioral similarity
  5. Output: behavioral_fingerprints.json + behavioral_clusters.json
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Any

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
SUBMISSION_DIR = BASE_DIR / 'submission'

from runtime.subprocess_executor import execute_student_function
from runtime.test_generator import get_test_for_criterion, get_testable_criteria

# Criterion → task file mapping
CRITERION_FILE_MAP = {
    'RQ1_1': 'task1.py', 'RQ1_2': 'task1.py', 'RQ1_3': 'task1.py', 'RQ1_4': 'task1.py',
    'RQ2_1': 'task2.py', 'RQ2_2': 'task2.py', 'RQ2_3': 'task2.py', 'RQ2_4': 'task2.py',
    'RQ3_1': 'task3.py', 'RQ3_2': 'task3.py', 'RQ3_3': 'task3.py',
    'RQ3_4': 'task3.py', 'RQ3_5': 'task3.py', 'RQ3_6': 'task3.py',
}

# Criterion → target function names to extract
CRITERION_FUNC_MAP = {
    'RQ1_1': ['update_playlist_tracks'],
    'RQ1_2': ['update_playlist_tracks'],
    'RQ1_3': ['update_playlist_tracks'],
    'RQ1_4': ['update_playlist_tracks'],
    'RQ2_1': ['get_all_genres', 'get_statistics'],
    'RQ2_2': ['get_all_genres'],
    'RQ2_3': ['get_all_genres', 'get_statistics'],
    'RQ2_4': ['get_statistics'],
    'RQ3_1': ['get_all_playlists'],
    'RQ3_2': ['create_playlist'],
    'RQ3_3': ['rename_playlist'],
    'RQ3_4': ['delete_playlist'],
    'RQ3_5': ['add_tracks_by_genre'],
    'RQ3_6': ['remove_tracks_by_genre'],
}


def extract_functions(code: str, func_names: list[str]) -> str:
    """Extract specified functions from student code."""
    parts = []
    for fname in func_names:
        pattern = (
            r'(?:(?:@[^\n]+\n\s*)*)def\s+' + re.escape(fname) +
            r'\s*\([^)]*\)(?:\s*->\s*\w+)?\s*:.*?'
            r'(?=\n(?:@[^\n]+\n\s*)?def\s+\w+\s*\(|\Z)'
        )
        m = re.search(pattern, code, re.DOTALL)
        if m:
            body = m.group().strip()
            if len(body) > 20:
                parts.append(body)
    return '\n\n'.join(parts)


def collect_students() -> list[str]:
    """Get all student IDs from submission directory."""
    students = []
    for entry in sorted(SUBMISSION_DIR.iterdir()):
        if entry.is_dir() and entry.name.startswith('S'):
            students.append(entry.name)
    return students


def run_behavioral_tests(
    students: list[str],
    criteria: list[str] | None = None,
    timeout: int = 30,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run behavioral tests for all students × criteria.

    Args:
        students: list of student IDs
        criteria: criteria to test (None = all testable)
        timeout: max seconds per student per criterion
        verbose: print progress

    Returns:
        {student_id: {criterion: {vulnerable, details, error, ...}}}
    """
    if criteria is None:
        criteria = get_testable_criteria()

    fingerprints: dict[str, dict] = defaultdict(dict)
    total = len(students) * len(criteria)
    done = 0

    for sid in students:
        for criterion in criteria:
            done += 1
            test_info = get_test_for_criterion(criterion)
            if not test_info:
                continue

            file_name = CRITERION_FILE_MAP.get(criterion, 'task1.py')
            func_names = CRITERION_FUNC_MAP.get(criterion, [])

            file_path = SUBMISSION_DIR / sid / file_name
            if not file_path.exists():
                fingerprints[sid][criterion] = {
                    'error': f'{file_name} not found', 'vulnerable': None
                }
                continue

            code = file_path.read_text(encoding='utf-8', errors='ignore')
            extracted = extract_functions(code, func_names)

            if not extracted or len(extracted.strip()) < 20:
                fingerprints[sid][criterion] = {
                    'error': 'target function not found or empty',
                    'vulnerable': None
                }
                continue

            # Run test
            result = execute_student_function(
                extracted, test_info['test_harness'], timeout=timeout
            )

            behavior = result.get('result_json') or {}
            behavior['exit_code'] = result['exit_code']
            behavior['timeout'] = result.get('timeout', False)
            behavior['duration_ms'] = result.get('duration_ms', 0)
            if result.get('stderr'):
                behavior['stderr_tail'] = result['stderr'][-500:]

            fingerprints[sid][criterion] = behavior

            if verbose:
                vuln = behavior.get('vulnerable')
                if vuln is True:
                    tag = '!!'
                elif vuln is False:
                    tag = 'OK'
                else:
                    tag = '??'
                dur = result.get('duration_ms', 0)
                print(f'  [{done}/{total}] {sid}/{criterion}: {tag} ({dur}ms)')

    return dict(fingerprints)


def cluster_by_behavior(
    fingerprints: dict[str, dict],
    criteria: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Cluster students by behavioral similarity per criterion.

    For each criterion, groups students by their behavioral fingerprint
    (e.g., vulnerable=True vs False, or specific error types).

    Args:
        fingerprints: {student_id: {criterion: {vulnerable, ...}}}
        criteria: criteria to cluster (None = all in fingerprints)

    Returns:
        {criterion: [{label, students, count, fingerprint, ...}]}
    """
    if criteria is None:
        criteria_set = set()
        for fp in fingerprints.values():
            criteria_set.update(fp.keys())
        criteria = sorted(criteria_set)

    clusters: dict[str, list[dict]] = {}

    for criterion in criteria:
        # Group students by behavioral fingerprint
        groups: dict[str, list[str]] = defaultdict(list)

        for sid, fp in fingerprints.items():
            crit_fp = fp.get(criterion, {})
            if not crit_fp:
                continue

            # Build a behavioral key from the fingerprint
            behavior_key = _make_behavior_key(crit_fp, criterion)

            groups[behavior_key].append(sid)

        # Build cluster list
        crit_clusters = []
        for key, sids in sorted(groups.items(), key=lambda x: -len(x[1])):
            # Find a representative student
            rep_sid = sorted(sids)[0]
            rep_fp = fingerprints.get(rep_sid, {}).get(criterion, {})

            # Generate human-readable label
            label = _make_cluster_label(key, rep_fp, criterion)

            crit_clusters.append({
                'label': label,
                'behavior_key': key,
                'students': sorted(sids),
                'count': len(sids),
                'representative_student': rep_sid,
                'fingerprint': rep_fp,
            })

        clusters[criterion] = crit_clusters

    return clusters


def _make_behavior_key(fp: dict, criterion: str) -> str:
    """Create a concise behavioral fingerprint string for clustering."""
    vuln = fp.get('vulnerable')
    error = fp.get('error')
    timeout = fp.get('timeout', False)

    if timeout:
        return 'TIMEOUT'
    if error:
        return f'ERROR:{error[:40]}'
    if vuln is True:
        return 'VULNERABLE'
    if vuln is False:
        return 'SAFE'
    return 'UNKNOWN'


def _make_cluster_label(key: str, fp: dict, criterion: str) -> str:
    """Generate a human-readable label for a behavioral cluster."""
    if key == 'VULNERABLE':
        test_info = get_test_for_criterion(criterion)
        desc = test_info.get('description', '') if test_info else ''
        return f'[BEHAVIOR: VULNERABLE] {desc}'
    elif key == 'SAFE':
        test_info = get_test_for_criterion(criterion)
        desc = test_info.get('description', '') if test_info else ''
        return f'[BEHAVIOR: SAFE] {desc}'
    elif key.startswith('ERROR:'):
        return f'[BEHAVIOR: ERROR] {key[6:]}'
    elif key == 'TIMEOUT':
        return '[BEHAVIOR: TIMEOUT] Code did not complete within time limit'
    return f'[BEHAVIOR: {key}]'


def load_or_run_fingerprints(
    cache_path: str,
    students: list[str],
    criteria: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Load cached fingerprints or run tests.

    Args:
        cache_path: path to behavioral_fingerprints.json
        students: student IDs
        criteria: criteria to test
        force: if True, re-run even if cache exists
    """
    if not force and os.path.exists(cache_path):
        print(f'Loading cached fingerprints from {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f'Running behavioral tests for {len(students)} students...')
    fps = run_behavioral_tests(students, criteria)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(fps, f, indent=2, ensure_ascii=False)
    print(f'Saved fingerprints to {cache_path}')

    return fps


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    students = collect_students()
    print(f'Found {len(students)} students')

    criteria = get_testable_criteria()
    print(f'Testable criteria: {criteria}')

    # Run behavioral tests
    fps = run_behavioral_tests(students, criteria)

    # Cluster
    clusters = cluster_by_behavior(fps, criteria)

    for crit, clist in clusters.items():
        print(f'\n{criterion}: {len(clist)} clusters')
        for c in clist:
            print(f'  {c["label"][:80]} — {c["count"]} students')
