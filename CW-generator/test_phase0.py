"""Quick test of CW-generator Phase 0"""
import sys, json, os
sys.path.insert(0, '.')
from pathlib import Path
from pattern_matrix import build_pattern_matrix, MIN_COVERAGE_PER_VARIANT

rubric_path = '../FS_generater-v1/output/q1_iMusic/rubric_cache.json'
print(f'Loading rubric from {rubric_path}...')
print(f'Min coverage: {MIN_COVERAGE_PER_VARIANT}')

try:
    matrix = build_pattern_matrix(rubric_path)
    print(f'Matrix built: {len(matrix["criteria"])} criteria, {len(matrix["all_patterns"])} patterns')
    for rid, data in sorted(matrix['criteria'].items()):
        pats = data['patterns']
        good = [p for p in pats if p['type'] == 'good']
        bad = [p for p in pats if p['type'] == 'bad']
        print(f'  {rid}: {len(good)} good, {len(bad)} bad')
        for bp in bad:
            for v in bp['variants']:
                inst = v.get('instruction', '?')
                # Check for vague language
                if any(w in inst.lower() for w in ['deliberately', 'incorrectly', 'not ', 'without', 'missing']):
                    print(f'    [VAGUE] {inst[:100]}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
