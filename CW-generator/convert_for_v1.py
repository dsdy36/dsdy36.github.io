"""
Convert v5 submissions to V1 format. Replace stubs with actual task code.
Uses regex matching to handle signature variations.
"""
import os, re

V1_TEMPLATE = r'..\FS_generater-v1\question\code\iMusic.py'
V1_SUBMISSION_DIR = r'..\FS_generater-v1\submission'
SOURCE_DIR = r'submissions_imusic_v5'


def extract_function(task_code: str, func_name: str) -> str:
    """Extract a complete function definition from code using regex."""
    lines = task_code.split('\n')
    result = []
    in_func = False
    for line in lines:
        if re.match(rf'def\s+{func_name}\s*\(', line.strip()):
            in_func = True
        elif in_func and line.strip().startswith('def ') and func_name not in line:
            break
        elif in_func and re.match(r'@app\.route', line.strip()):
            break
        if in_func:
            result.append(line)
    return '\n'.join(result)


def replace_stub(template: str, func_name: str, new_code: str) -> str:
    """Replace a function stub in the template with actual code.
    Matches from 'def func_name' to 'pass # Delete this line'."""
    if not new_code:
        return template

    # Pattern: def func_name(...): ... until 'pass # Delete this line when you implement the function'
    pattern = rf'(def\s+{func_name}\s*\([^)]*\)\s*:.*?pass\s+#\s*Delete this line when you implement the function)'
    m = re.search(pattern, template, re.DOTALL)
    if m:
        return template[:m.start()] + new_code.strip() + template[m.end():]
    return template


def convert():
    template = open(V1_TEMPLATE, 'r', encoding='utf-8').read()

    dirs = sorted([d for d in os.listdir(SOURCE_DIR)
                   if os.path.isdir(os.path.join(SOURCE_DIR, d)) and not d.startswith('_')])

    task_funcs = {
        1: ['update_playlist_tracks'],
        2: ['statistics', 'get_all_genres', 'get_statistics'],
        3: ['playlists', 'get_all_playlists', 'create_playlist', 'rename_playlist',
            'delete_playlist', 'add_tracks_by_genre', 'remove_tracks_by_genre'],
    }

    task_counts = {'q1': {}, 'q2': {}, 'q3': {}}
    replaced = 0
    failed = []

    for d in dirs:
        quality = d.split('-')[0]
        sp = os.path.join(SOURCE_DIR, d)

        task_codes = {}
        for tn in [1, 2, 3]:
            tf = os.path.join(sp, f'task{tn}.py')
            if os.path.exists(tf):
                with open(tf, 'r', encoding='utf-8') as f:
                    task_codes[tn] = f.read()

        for task_num in [1, 2, 3]:
            task_key = f'q{task_num}'
            task_counts[task_key].setdefault(quality, 0)
            task_counts[task_key][quality] += 1
            variant = task_counts[task_key][quality]
            v1_dirname = f'{task_key}-{quality}-v{variant:03d}'
            v1_path = os.path.join(V1_SUBMISSION_DIR, v1_dirname, 'submission')
            os.makedirs(v1_path, exist_ok=True)

            # Start with fresh template, replace only this task's stubs
            merged = template
            if task_num in task_codes:
                for fname in task_funcs[task_num]:
                    new_func = extract_function(task_codes[task_num], fname)
                    if new_func:
                        merged = replace_stub(merged, fname, new_func)
                        replaced += 1
                    else:
                        if task_num <= 2 or fname not in ['playlists']:
                            failed.append((d, task_key, fname))

            with open(os.path.join(v1_path, 'iMusic.py'), 'w', encoding='utf-8') as f:
                f.write(merged)

    total = sum(sum(c.values()) for c in task_counts.values())
    print(f"Converted {total} entries ({len(dirs)} students x 3 tasks)")
    print(f"Stub replacements: {replaced}")
    if failed:
        print(f"Failed extractions: {len(failed)}")
        for d, tk, fn in failed[:5]:
            print(f"  {d}/{tk}: could not extract {fn}")

    for tk in ['q1', 'q2', 'q3']:
        counts = task_counts[tk]
        print(f"  {tk}: {sum(counts.values())} total — { {k: v for k, v in sorted(counts.items())} }")


if __name__ == '__main__':
    convert()
