"""
Integrated CW Submission Generator
==================================
Replaces main_v5.py's missing modules (coverage_assigner, variant_prompter,
coverage_checker, api_client, post_processor) with a single integrated script.

Usage:
    python generate_cw.py -n 50
"""
import argparse
import json
import os
import re
import sys
import time
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')

from pattern_matrix import build_pattern_matrix, MIN_COVERAGE_PER_VARIANT

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RUBRIC_CACHE = SCRIPT_DIR / '..' / 'FS-generator' / 'output' / 'q1_iMusic' / 'rubric_cache.json'
OUTPUT_DIR = SCRIPT_DIR / 'submissions_imusic_v5'
TASK_DESC_FILE = SCRIPT_DIR / 'task_description_imusic.md'
TEMPLATE_CODE_FILE = SCRIPT_DIR / 'question' / 'code' / 'iMusic.py'
VARIANT_CACHE_FILE = SCRIPT_DIR / 'variant_cache.json'

SYSTEM_PROMPT = """You are a CS student completing a Flask+SQLite coursework assignment.
Write complete, working Python code for the specified task functions.

CRITICAL RULES:
1. Write ONLY the task functions requested. Do NOT write Flask routes or main blocks.
2. Follow the exact instructions given for each function.
3. Use the EXACT function signatures from the template.
4. The template provides helper functions (get_db_connection, etc.) — use them.
5. Write realistic student code — include imports, helper functions if needed.
6. Each function should be syntactically valid Python.

Output format: for each task, output the complete code for that task file.
Use markdown code blocks labeled with the task number:
```task1
...complete task1.py code...
```
```task2
...complete task2.py code...
```
```task3
...complete task3.py code...
```"""


def load_template():
    """Load the iMusic.py template code."""
    if TEMPLATE_CODE_FILE.exists():
        return TEMPLATE_CODE_FILE.read_text(encoding='utf-8')
    return ''


def load_task_description():
    """Load the task description markdown."""
    if TASK_DESC_FILE.exists():
        return TASK_DESC_FILE.read_text(encoding='utf-8')
    return ''


def assign_patterns(matrix, num_students, quality_dist):
    """Assign good/bad patterns to each student with coverage guarantees."""
    n_excellent = max(1, int(num_students * quality_dist['excellent']))
    n_medium = max(1, int(num_students * quality_dist['medium']))
    n_poor = num_students - n_excellent - n_medium

    all_patterns = matrix['all_patterns']
    good_patterns = [p for p in all_patterns if '_G' in p]
    bad_patterns = [p for p in all_patterns if '_B' in p]

    # Count how many variants total
    total_variants = sum(len(matrix['pattern_variants'].get(p, [])) for p in all_patterns)

    # Track coverage per variant
    variant_coverage = defaultdict(int)

    students = []
    used_bad = defaultdict(set)  # student -> set of criterion bad pattern IDs
    used_good = defaultdict(set)

    for i in range(num_students):
        sid = f'S{i+1:03d}'
        if i < n_excellent:
            tier = 'excellent'
            n_bad = random.randint(0, 1)
            n_good_total = len(good_patterns)
            n_good = max(n_good_total - 2, n_good_total)
        elif i < n_excellent + n_medium:
            tier = 'medium'
            n_bad = random.randint(2, 4)
            n_good_total = len(good_patterns)
            n_good = random.randint(n_good_total - 4, n_good_total - 1)
        else:
            tier = 'poor'
            n_bad = random.randint(5, 8)
            n_good_total = len(good_patterns)
            n_good = random.randint(1, max(1, n_good_total - 6))

        # Pick bad patterns
        student_bad = []
        bad_by_criterion = defaultdict(list)
        for bp in bad_patterns:
            crit = bp.split('_')[0] + '_' + bp.split('_')[1]  # e.g. "RQ1_1"
            bad_by_criterion[crit].append(bp)

        # Pick n_bad criteria to mess up
        criteria_with_bad = list(bad_by_criterion.keys())
        random.shuffle(criteria_with_bad)
        bad_count = 0
        chosen_bad = []

        # First pass: ensure each variant has MIN_COVERAGE
        for bp in bad_patterns:
            variants = matrix['pattern_variants'].get(bp, [])
            for v in variants:
                vkey = f'{bp}_{v["id"]}'
                if variant_coverage[vkey] < MIN_COVERAGE_PER_VARIANT:
                    chosen_bad.append(bp)
                    variant_coverage[vkey] += 1
                    bad_count += 1
                    break
            if bad_count >= n_bad:
                break

        # Second pass: fill remaining slots
        if bad_count < n_bad:
            remaining_bad = [bp for bp in bad_patterns if bp not in chosen_bad]
            random.shuffle(remaining_bad)
            for bp in remaining_bad:
                if bad_count >= n_bad:
                    break
                chosen_bad.append(bp)
                variants = matrix['pattern_variants'].get(bp, [])
                for v in variants:
                    vkey = f'{bp}_{v["id"]}'
                    variant_coverage[vkey] += 1

        # Pick good patterns
        student_good = []
        good_by_criterion = defaultdict(list)
        for gp in good_patterns:
            crit = gp.split('_')[0] + '_' + gp.split('_')[1]
            good_by_criterion[crit].append(gp)

        for crit, patterns in good_by_criterion.items():
            # For criteria where student has a bad pattern, don't assign good
            crit_bad = [p for p in chosen_bad if p.startswith(crit)]
            if crit_bad:
                continue
            # Pick the good pattern for this criterion
            student_good.append(random.choice(patterns))

        # If not enough good, add more
        if len(student_good) < n_good:
            remaining = [p for p in good_patterns if p not in student_good]
            random.shuffle(remaining)
            for p in remaining:
                if len(student_good) >= n_good:
                    break
                crit = p.split('_')[0] + '_' + p.split('_')[1]
                if not any(b.startswith(crit) for b in chosen_bad):
                    student_good.append(p)

        students.append({
            'student_id': sid,
            'quality_tier': tier,
            'good_patterns': student_good,
            'bad_patterns': chosen_bad,
        })

    return students


def build_student_prompt(student, matrix, template_code, task_desc):
    """Build a code generation prompt for one student."""
    variant_map = matrix['pattern_variants']

    # Collect instructions by task
    task_instructions = {1: [], 2: [], 3: []}

    for pid in student['good_patterns'] + student['bad_patterns']:
        variants = variant_map.get(pid, [])
        # Pick one variant per pattern
        if variants:
            v = random.choice(variants)
            inst = v['instruction']

            # Determine task from pattern ID
            crit = pid.split('_')[0] + '_' + pid.split('_')[1]
            m = re.search(r'(\d)', crit)
            tn = int(m.group(1)) if m else 1
            ptype = 'good' if '_G' in pid else 'bad'
            tag = '[CORRECT]' if ptype == 'good' else '[MISTAKE]'
            task_instructions[tn].append(f'  {tag} {inst}')

    # Build prompt
    parts = []
    parts.append('You are writing code for an iMusic Flask+SQLite coursework.')
    parts.append('')
    parts.append('TEMPLATE CODE (imports, Flask setup, helper functions):')
    parts.append('```python')
    parts.append(template_code[:3000])
    parts.append('```')
    parts.append('')
    parts.append('TASK DESCRIPTION:')
    parts.append(task_desc[:2000])
    parts.append('')
    parts.append(f'Your student ID: {student["student_id"]}')
    parts.append(f'Quality tier: {student["quality_tier"]}')
    parts.append('')
    parts.append('IMPLEMENT THESE SPECIFIC INSTRUCTIONS:')
    for tn in [1, 2, 3]:
        insts = task_instructions[tn]
        if insts:
            parts.append(f'\nTask {tn} functions:')
            for inst in insts:
                parts.append(inst)

    parts.append('')
    parts.append('OUTPUT FORMAT: Write three code blocks labeled ```task1```, ```task2```, ```task3```.')
    parts.append('Each block contains the COMPLETE code for that task file (including imports).')

    return '\n'.join(parts)


def call_api(prompt):
    """Call DeepSeek API to generate code."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=8192,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f'  API attempt {attempt+1} failed: {e}')
            if attempt < 2:
                time.sleep(5)
    return None


def parse_response(response_text):
    """Parse the three task code blocks from the API response."""
    tasks = {}
    for tn in [1, 2, 3]:
        pattern = rf'```task{tn}\s*\n(.*?)```'
        m = re.search(pattern, response_text, re.DOTALL)
        if m:
            tasks[tn] = m.group(1).strip()
    return tasks


def generate_all(students, matrix, template_code, task_desc, max_concurrent=5):
    """Generate code for all students."""
    results = {}
    total = len(students)
    for i, student in enumerate(students):
        sid = student['student_id']
        print(f'\n[{i+1}/{total}] Generating {sid} ({student["quality_tier"]})...')
        print(f'  Good: {len(student["good_patterns"])}, Bad: {len(student["bad_patterns"])}')

        prompt = build_student_prompt(student, matrix, template_code, task_desc)
        response = call_api(prompt)

        if response:
            tasks = parse_response(response)
            if len(tasks) == 3:
                results[sid] = tasks
                print(f'  OK — task lengths: { {tn: len(c) for tn, c in tasks.items()} }')
            else:
                missing = [tn for tn in [1,2,3] if tn not in tasks]
                print(f'  PARTIAL — missing tasks: {missing}')
                results[sid] = tasks
        else:
            print(f'  FAILED')

        # Small delay between calls
        if i < total - 1:
            time.sleep(1)

    return results


def save_submissions(results, students, output_dir, matrix):
    """Save generated submissions to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build student lookup
    student_map = {s['student_id']: s for s in students}

    for sid, tasks in results.items():
        student_dir = output_dir / sid
        student_dir.mkdir(parents=True, exist_ok=True)

        for tn in [1, 2, 3]:
            code = tasks.get(tn, '# No code generated')
            task_file = student_dir / f'task{tn}.py'
            task_file.write_text(code, encoding='utf-8')

        # Generate README
        student = student_map.get(sid, {})
        readme = generate_student_readme(sid, student, matrix)
        (student_dir / 'README.md').write_text(readme, encoding='utf-8')

    print(f'\nSaved {len(results)} students to {output_dir}')


def generate_student_readme(sid, student, matrix):
    """Generate README.md for a student showing assigned patterns."""
    parts = [f'# {sid} — iMusic Coursework Submission', '']

    quality = student.get('quality_tier', 'medium')
    good = student.get('good_patterns', [])
    bad = student.get('bad_patterns', [])

    parts.append(f'| Attribute | Value |')
    parts.append(f'|-----------|-------|')
    parts.append(f'| Quality Tier | {quality} |')
    parts.append(f'| Correct Patterns | {len(good)} |')
    parts.append(f'| Error Patterns | {len(bad)} |')
    parts.append('')

    # Group patterns by criterion
    by_criterion = defaultdict(lambda: {'good': [], 'bad': []})
    variant_map = matrix['pattern_variants']

    for pid in good:
        crit = pid.split('_')[0] + '_' + pid.split('_')[1]
        variants = variant_map.get(pid, [])
        for v in variants:
            by_criterion[crit]['good'].append(f'- [{v["id"]}] {v["instruction"]}')

    for pid in bad:
        crit = pid.split('_')[0] + '_' + pid.split('_')[1]
        variants = variant_map.get(pid, [])
        for v in variants:
            by_criterion[crit]['bad'].append(f'- [{v["id"]}] {v["instruction"]}')

    for crit in sorted(by_criterion.keys()):
        data = by_criterion[crit]
        # Determine criterion name from pattern IDs
        crit_patterns = [p for p in (data['good'] + data['bad'])]
        crit_name = crit  # Use the criterion ID as name

        parts.append(f'### {crit}: {crit_name}')
        if data['good']:
            parts.append('')
            parts.append('**Correct approach**')
            parts.extend(data['good'])
        if data['bad']:
            parts.append('')
            parts.append('**Error pattern**')
            parts.extend(data['bad'])
        parts.append('')

    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-students', type=int, default=50)
    parser.add_argument('--rubric', type=str, default=str(RUBRIC_CACHE))
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    print(f'{"=" * 60}')
    print(f'  CW SUBMISSION GENERATOR')
    print(f'  Students: {args.num_students}')
    print(f'  Model: {DEEPSEEK_MODEL}')
    print(f'{"=" * 60}')

    # Load rubric
    if not os.path.exists(args.rubric):
        print(f'ERROR: rubric_cache.json not found at {args.rubric}')
        print('Run FS-generator Phase 0 first.')
        sys.exit(1)

    # Build pattern matrix
    print('\n[1/4] Building pattern matrix...')
    matrix = build_pattern_matrix(args.rubric, str(VARIANT_CACHE_FILE))
    print(f'  Criteria: {len(matrix["criteria"])}')
    print(f'  Total patterns: {len(matrix["all_patterns"])} '
          f'({len([p for p in matrix["all_patterns"] if "_G" in p])} good, '
          f'{len([p for p in matrix["all_patterns"] if "_B" in p])} bad)')

    quality_dist = {'excellent': 0.15, 'medium': 0.55, 'poor': 0.30}

    # Assign patterns
    print(f'\n[2/4] Assigning patterns to {args.num_students} students...')
    students = assign_patterns(matrix, args.num_students, quality_dist)
    tiers = defaultdict(int)
    for s in students:
        tiers[s['quality_tier']] += 1
    print(f'  Excellent: {tiers["excellent"]}, Medium: {tiers["medium"]}, Poor: {tiers["poor"]}')

    # Load template and task description
    template_code = load_template()
    task_desc = load_task_description()
    print(f'  Template: {len(template_code)} chars')
    print(f'  Task desc: {len(task_desc)} chars')

    # Generate
    print(f'\n[3/4] Generating code via DeepSeek API...')
    results = generate_all(students, matrix, template_code, task_desc)

    # Save
    print(f'\n[4/4] Saving submissions...')
    save_submissions(results, students, args.output, matrix)

    print(f'\nDone! {len(results)}/{args.num_students} students generated.')


if __name__ == '__main__':
    main()
