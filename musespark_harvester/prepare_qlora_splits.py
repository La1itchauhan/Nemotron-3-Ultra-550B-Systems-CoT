#!/usr/bin/env python3
"""
Nemotron-3-Ultra-550B-Systems-CoT: QLoRA Ablation Split Generator
Author: Lalit Chauhan (@La1itchauhan)
License: Apache-2.0

Generates balanced dataset splits for the 3-arm QLoRA ablation experiment:
- Arm 2: 35 unverified control traces (from full_benchmark)
- Arm 3: 35 native-executed verified traces (from verified)
- Held-out Evaluation Set: 8 held-out task families (2 per domain)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SPLITS_DIR = OUTPUT_DIR / "qlora_splits"
BENCHMARK_PATH = OUTPUT_DIR / "musespark_systems_full_benchmark.jsonl"
VERIFIED_PATH = OUTPUT_DIR / "musespark_systems_verified.jsonl"


def generate_splits(seed: int = 42):
    random.seed(seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_records = [json.loads(line) for line in f if line.strip()]

    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        verified_records = [json.loads(line) for line in f if line.strip()]

    # 1. Arm 3: 35 verified traces (out of 36)
    arm3_records = verified_records[:35]
    arm3_file = SPLITS_DIR / "arm3_verified_35.jsonl"
    with open(arm3_file, "w", encoding="utf-8") as f:
        for r in arm3_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[SPLIT] Arm 3 (Verified): Saved {len(arm3_records)} rows to {arm3_file.name}")

    # 2. Arm 2: 35 unverified control traces from full_benchmark
    verified_prompts = {r.get("prompt_sha256") for r in verified_records}
    unverified_pool = [
        r for r in bench_records
        if r.get("prompt_sha256") not in verified_prompts
        or r.get("verification_level") != "executed"
    ]

    # Balance across 4 splits
    by_split = defaultdict(list)
    for r in unverified_pool:
        by_split[r.get("split")].append(r)

    arm2_records = []
    target_per_split = 35 // 4 # ~8-9 per split
    for s_name in ["nvidia_cuda_tensorrt", "arm_embedded_rtl", "modern_cpp17_pybind11", "s2_geodesic_spatial"]:
        pool = by_split[s_name]
        random.shuffle(pool)
        arm2_records.extend(pool[:9 if len(arm2_records) < 35 - 8 else 8])

    arm2_records = arm2_records[:35]
    arm2_file = SPLITS_DIR / "arm2_unverified_35.jsonl"
    with open(arm2_file, "w", encoding="utf-8") as f:
        for r in arm2_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[SPLIT] Arm 2 (Unverified Control): Saved {len(arm2_records)} rows to {arm2_file.name}")

    # 3. Held-out Evaluation Set: 8 task families (2 per domain)
    # Pick subcategories that are in benchmark but not in arm3 training set
    arm3_subcats = {r.get("subcategory") for r in arm3_records}
    held_out_candidates = defaultdict(list)
    for r in bench_records:
        subcat = r.get("subcategory")
        if subcat and subcat not in arm3_subcats:
            held_out_candidates[r.get("split")].append(r)

    held_out_records = []
    for s_name in ["nvidia_cuda_tensorrt", "arm_embedded_rtl", "modern_cpp17_pybind11", "s2_geodesic_spatial"]:
        cands = held_out_candidates[s_name]
        # Group by distinct subcategory and pick up to 2 distinct task families
        by_sub = defaultdict(list)
        for r in cands:
            by_sub[r.get("subcategory")].append(r)
        
        for subcat, items in list(by_sub.items())[:2]:
            held_out_records.append(items[0])

    held_out_file = SPLITS_DIR / "held_out_eval_8_families.jsonl"
    with open(held_out_file, "w", encoding="utf-8") as f:
        for r in held_out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[SPLIT] Held-Out Evaluation Set: Saved {len(held_out_records)} rows across 8 task families to {held_out_file.name}")

    summary = {
        "arm3_verified_count": len(arm3_records),
        "arm2_unverified_count": len(arm2_records),
        "held_out_eval_count": len(held_out_records),
        "held_out_task_families": list({r.get("subcategory") for r in held_out_records}),
    }
    with open(SPLITS_DIR / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] All QLoRA ablation splits generated successfully in {SPLITS_DIR}")


if __name__ == "__main__":
    generate_splits()
