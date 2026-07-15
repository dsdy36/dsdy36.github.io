import re
with open('pattern_matrix.py', 'r', encoding='utf-8') as f:
    source = f.read()
vague = []
total = 0
for m in re.finditer(r"'instruction':\s*'([^']*)'", source):
    inst = m.group(1)
    total += 1
    if any(w in inst.lower() for w in [' not ', 'without', 'missing', 'deliberately', 'incorrectly']):
        vague.append(inst[:120])
print(f'Total instructions: {total}')
print(f'Remaining vague: {len(vague)}')
for v in vague:
    print(f'  [{v}]')
