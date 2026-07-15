#!/usr/bin/env python3
"""
main_v5 — Coverage-Guaranteed Submission Generator
====================================================
Phase 0: Build pattern matrix from rubric
Phase 1: Coverage-guided pattern assignment
Phase 2: Build variant-level prompts
Phase 3: Generate submissions via DeepSeek
Phase 4: Post-generation coverage check
Phase 5: Iterative supplement (fill gaps)

Usage:
    python main_v5.py [-n 27] [--rubric PATH]
"""
import argparse, asyncio, json, sys
from pathlib import Path

from pattern_matrix import build_pattern_matrix, MIN_COVERAGE_PER_VARIANT
from coverage_assigner import assign_patterns
from variant_prompter import build_variant_prompts
from coverage_checker import check_coverage, report_gaps, print_coverage_report
from api_client import load_api_key, build_client, batch_generate
from post_processor import process_and_save


DEFAULT_CONFIG = {
    "num_students": 27,
    "quality_distribution": {"excellent": 0.15, "medium": 0.55, "poor": 0.30},
    "deepseek": {
        "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 8192, "max_concurrent": 5, "max_retries": 3,
    },
    "task_description": "task_description_imusic.md",
    "data_dir": "data_imusic",
    "output_dir": "submissions_imusic_v5",
    "rubric_cache": "../FS_generater-v1/output/q1_iMusic/rubric_cache.json",
}


def parse_args():
    p = argparse.ArgumentParser(description="Coverage-guaranteed submission generator")
    p.add_argument("-n", "--num-students", type=int, default=27)
    p.add_argument("--rubric", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-supplement", type=int, default=2, help="Max supplement rounds")
    return p.parse_args()


def load_task_and_data(config):
    task_path = Path(config["task_description"])
    task_desc = task_path.read_text(encoding="utf-8")
    data_dir = Path(config.get("data_dir", "data"))
    data_files = {}
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix not in ('.db', '.sqlite', '.sqlite3', '.bin'):
                try:
                    data_files[f.name] = f.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue  # skip binary files silently
    return task_desc, data_files


async def main():
    args = parse_args()
    config = {**DEFAULT_CONFIG}
    rubric_path = args.rubric or config["rubric_cache"]
    ds = config["deepseek"]
    num_students = args.num_students

    print("=" * 60)
    print("  Coverage-Guaranteed Submission Generator v5")
    print("=" * 60)
    print(f"  Students: {num_students} | Min coverage/variant: {MIN_COVERAGE_PER_VARIANT}")
    print(f"  Rubric: {rubric_path}")

    # ── Phase 0: Build pattern matrix ──
    print(f"\n[0] Building pattern matrix from rubric...")
    variant_cache = config.get("variant_cache", "variant_cache.json")
    matrix = build_pattern_matrix(rubric_path, variant_cache_path=variant_cache)
    n_patterns = len(matrix['all_patterns'])
    n_variants = sum(len(v) for v in matrix['pattern_variants'].values())
    n_criteria = len(matrix['criteria'])
    print(f"  {n_criteria} criteria → {n_patterns} patterns → {n_variants} variants")
    for rid, cinfo in matrix['criteria'].items():
        n_good = sum(1 for p in cinfo['patterns'] if p['type'] == 'good')
        n_bad = sum(1 for p in cinfo['patterns'] if p['type'] == 'bad')
        print(f"    {rid} ({cinfo['name'][:40]}): {n_good}G + {n_bad}B")

    # ── Phase 1: Coverage-guided assignment ──
    print(f"\n[1] Assigning patterns with coverage guarantees...")
    profiles, coverage = assign_patterns(matrix, num_students, config["quality_distribution"], seed=args.seed)

    # Print assignment summary
    tier_counts = {}
    for p in profiles:
        t = p['quality_tier']
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for t in ['excellent', 'medium', 'poor']:
        count = tier_counts.get(t, 0)
        avg_good = sum(len(p['good_patterns']) for p in profiles if p['quality_tier'] == t)
        avg_bad = sum(len(p['bad_patterns']) for p in profiles if p['quality_tier'] == t)
        if count > 0:
            print(f"  {t:10s}: {count} students | avg good={avg_good/count:.1f} avg bad={avg_bad/count:.1f}")

    # Coverage stats
    cov_values = list(coverage.values())
    print(f"  Coverage: {len(cov_values)} variants, min={min(cov_values)}, max={max(cov_values)}, avg={sum(cov_values)/len(cov_values):.1f}")

    # ── Phase 2: Build prompts ──
    print(f"\n[2] Building variant-level prompts...")
    task_desc, data_files = load_task_and_data(config)
    prompts = build_variant_prompts(profiles, task_desc, data_files)
    print(f"  {len(prompts)} prompts built")

    # ── Phase 3: Generate ──
    print(f"\n[3] Generating {num_students} submissions...")
    client = build_client(load_api_key(), base_url=ds["base_url"])
    results = await batch_generate(
        prompts=prompts, client=client, model=ds["model"],
        max_tokens=ds.get("max_tokens", 8192),
        max_concurrent=ds.get("max_concurrent", 5),
        max_retries=ds.get("max_retries", 3),
    )

    # ── Phase 4: Post-process & coverage check ──
    output_dir = config["output_dir"]
    print(f"\n[4] Post-processing & coverage check...")
    metadata = process_and_save(results, output_dir)

    # Attach profiles to results for coverage checking
    for r in results:
        for p in profiles:
            if p['student_id'] == r['student_id']:
                r['good_patterns'] = p['good_patterns']
                r['bad_patterns'] = p['bad_patterns']
                break

    cov_result = check_coverage(output_dir, profiles, matrix)
    print_coverage_report(cov_result, profiles)

    # ── Phase 5: Iterative supplement ──
    gaps = report_gaps(cov_result, MIN_COVERAGE_PER_VARIANT)
    supplement_round = 0

    while gaps and supplement_round < args.max_supplement:
        supplement_round += 1
        n_gaps = len(gaps)
        print(f"\n[5.{supplement_round}] Supplement round — {n_gaps} under-covered variants")

        # Create supplement profiles for gap variants
        supp_profiles = []
        for i, vkey in enumerate(gaps[:min(n_gaps, 10)]):  # max 10 per round
            pid = vkey.rsplit('_', 1)[0]
            vid = vkey.rsplit('_', 1)[1]
            var_list = matrix['pattern_variants'].get(pid, [])
            var_info = None
            for v in var_list:
                if v['id'] == vid:
                    var_info = v
                    break
            if not var_info:
                continue

            is_good = '_G' in pid
            supp_profiles.append({
                'student_id': f'SUPP{supplement_round}_{i+1:02d}',
                'quality_tier': 'medium',
                'good_patterns': [{'pattern_id': pid, 'variant_id': vid, 'instruction': var_info['instruction']}] if is_good else [],
                'bad_patterns': [] if is_good else [{'pattern_id': pid, 'variant_id': vid, 'instruction': var_info['instruction']}],
            })

        if not supp_profiles:
            break

        # Generate supplement submissions
        supp_prompts = build_variant_prompts(supp_profiles, task_desc, data_files)
        supp_results = await batch_generate(
            prompts=supp_prompts, client=client, model=ds["model"],
            max_tokens=ds.get("max_tokens", 8192),
            max_concurrent=ds.get("max_concurrent", 5),
            max_retries=ds.get("max_retries", 3),
        )

        # Process and check again
        supp_dir = f"{output_dir}_supp{supplement_round}"
        process_and_save(supp_results, supp_dir)

        # Re-check coverage
        all_profiles = profiles + supp_profiles
        cov_result = check_coverage(output_dir, all_profiles, matrix)
        # Also check supplement dir
        cov_supp = check_coverage(supp_dir, supp_profiles, matrix)
        for k, v in cov_supp.items():
            if k in cov_result:
                cov_result[k]['covered'] += v['covered']
                cov_result[k]['students'].extend(v['students'])

        print_coverage_report(cov_result, all_profiles)
        gaps = report_gaps(cov_result, MIN_COVERAGE_PER_VARIANT)

    print(f"\n{'='*60}")
    print(f"  Done! Output: {Path(output_dir).resolve()}")
    remaining = len(report_gaps(cov_result, MIN_COVERAGE_PER_VARIANT))
    print(f"  Remaining gaps: {remaining}/{len(cov_result)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
