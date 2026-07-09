"""
Generate README.md for each v5 submission — documents which rubric patterns
(good/bad) each student was assigned, organized by criterion.

Usage: python generate_readme.py [submissions_dir]
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pattern_matrix import build_pattern_matrix
from coverage_assigner import assign_patterns

RUBRIC_PATH = r'..\FS_generater-v1\output\q1_iMusic\rubric_cache.json'
QUALITY_DIST = {"excellent": 0.15, "medium": 0.55, "poor": 0.30}
SEED = 42


def generate(submissions_dir: str, num_students: int = 27):
    # Rebuild matrix and regenerate profiles (same seed = same assignments)
    matrix = build_pattern_matrix(RUBRIC_PATH)
    profiles, _ = assign_patterns(matrix, num_students, QUALITY_DIST, seed=SEED)

    # Save profiles
    with open(os.path.join(submissions_dir, '_profiles.json'), 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    # Build criterion lookup
    criteria = matrix['criteria']

    for profile in profiles:
        sid = profile['student_id']
        tier = profile['quality_tier']
        student_dir = os.path.join(submissions_dir, sid)
        os.makedirs(student_dir, exist_ok=True)

        # Group patterns by criterion
        by_criterion = {}
        for p in profile['good_patterns']:
            pid = p['pattern_id']
            # Extract criterion from pattern ID (e.g., RQ1_1_G1 → RQ1_1)
            cid = '_'.join(pid.split('_')[:2])  # RQ1_1
            by_criterion.setdefault(cid, {'good': [], 'bad': []})
            by_criterion[cid]['good'].append(p)
        for p in profile['bad_patterns']:
            pid = p['pattern_id']
            cid = '_'.join(pid.split('_')[:2])
            by_criterion.setdefault(cid, {'good': [], 'bad': []})
            by_criterion[cid]['bad'].append(p)

        # Build markdown
        # Summary stats
        covered_criteria = set()
        for p in profile['good_patterns'] + profile['bad_patterns']:
            cid = '_'.join(p['pattern_id'].split('_')[:2])
            covered_criteria.add(cid)

        lines = [
            f"# {sid} — Rubric Pattern Assignment",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Quality Tier | **{tier}** |",
            f"| Correct Patterns | {len(profile['good_patterns'])} |",
            f"| Error Patterns | {len(profile['bad_patterns'])} |",
            f"| Criteria Covered | {len(covered_criteria)}/14 |",
            f"",
            f"---",
            f"",
            f"## Rubric Coverage by Criterion",
            f"",
        ]

        for cid in sorted(by_criterion.keys()):
            cinfo = criteria.get(cid, {})
            cname = cinfo.get('name', cid)
            cmarks = cinfo.get('marks', '?')
            patterns = by_criterion[cid]

            lines.append(f"### {cid}: {cname} ({cmarks} marks)")
            lines.append(f"")

            if patterns['good']:
                lines.append(f"**✅ Correct approach (you should implement this):**")
                lines.append(f"")
                for p in patterns['good']:
                    lines.append(f"- [{p['variant_id']}] {p['instruction']}")
                lines.append(f"")

            if patterns['bad']:
                if tier == 'excellent':
                    lines.append(f"**⚠️ Intentional mistake (included for test coverage):**")
                elif tier == 'medium':
                    lines.append(f"**⚠️ Mistake to include (some issues expected):**")
                else:
                    lines.append(f"**❌ Error pattern (deliberately incorrect):**")
                lines.append(f"")
                for p in patterns['bad']:
                    lines.append(f"- [{p['variant_id']}] {p['instruction']}")
                lines.append(f"")

            if not patterns['good'] and not patterns['bad']:
                lines.append(f"*(No specific patterns assigned for this criterion)*")
                lines.append(f"")

        # Write README
        readme_path = os.path.join(student_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    print(f"Generated README.md for {len(profiles)} students in {submissions_dir}")
    print(f"Profiles saved to {os.path.join(submissions_dir, '_profiles.json')}")


if __name__ == '__main__':
    submissions = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else r'C:\Users\ZY C\Desktop\FURP\CW-generater\submissions_imusic_v5'
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    generate(submissions, num)
