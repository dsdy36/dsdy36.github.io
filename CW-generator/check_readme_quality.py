"""Check README quality for vague/Type A labels"""
import os, re
from collections import Counter

vague = Counter()
total = 0
critical = 0
for d in sorted(os.listdir('submissions_imusic_v5')):
    dpath = os.path.join('submissions_imusic_v5', d)
    if not os.path.isdir(dpath):
        continue
    readme = os.path.join(dpath, 'README.md')
    if not os.path.exists(readme):
        continue
    with open(readme, 'r', encoding='utf-8') as f:
        text = f.read()
    for m in re.finditer(
        r'\*\*(?:❌|⚠️)\s+(?:Error pattern|Mistake to include)[^*]*\*\*\s*\n((?:\s*- \[[A-Z]\]\s*[^\n]+\n?)*)',
        text
    ):
        for line in m.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('- ['):
                desc = re.sub(r'^-\s*\[[A-Z]\]\s*', '', line).strip()
                total += 1
                kw = []
                if 'deliberately' in desc.lower():
                    kw.append('deliberately')
                if 'implement incorrectly' in desc.lower():
                    kw.append('incorrectly')
                if ' not ' in desc.lower() and 'not in' not in desc.lower():
                    kw.append('not')
                if 'without' in desc.lower():
                    kw.append('without')
                if 'missing' in desc.lower():
                    kw.append('missing')
                if kw:
                    vague[', '.join(kw)] += 1
                if 'deliberately' in desc.lower() or 'implement incorrectly' in desc.lower():
                    critical += 1
                    print(f'  [CRITICAL] {d}: {desc[:120]}')

print(f'\nTotal bad/mistake labels: {total}')
print(f'Labels with vague language: {sum(vague.values())}')
for k, v in vague.most_common():
    print(f'  {k}: {v}')
dl = vague.get('deliberately', 0) + vague.get('incorrectly', 0) + vague.get('deliberately, incorrectly', 0)
print(f'\n"Deliberately implement incorrectly" occurrences: {critical}')
print(f'Verdict: {"[OK] No critically vague labels" if critical == 0 else "[WARN] Still has vague labels"}')
