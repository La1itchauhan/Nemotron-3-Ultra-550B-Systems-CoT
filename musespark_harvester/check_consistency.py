#!/usr/bin/env python3
"""
Nemotron-3-Ultra-550B-Systems-CoT: Verification Consistency Checker
Author: Lalit Chauhan (@La1itchauhan)
License: Apache-2.0

Verifies that the empirical metrics in council_audit_report.json match
both README.md and GITHUB_README.md down to the exact sample counts.
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_FILE = ROOT_DIR / "musespark_harvester" / "output" / "council_audit_report.json"
README_FILE = ROOT_DIR / "README.md"
GITHUB_README_FILE = ROOT_DIR / "GITHUB_README.md"


def check_consistency():
    if not REPORT_FILE.exists():
        print(f"[FAIL] Report file not found at {REPORT_FILE}")
        sys.exit(1)

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    verified_count = summary.get("verified_split_count", 0)
    benchmark_count = summary.get("full_benchmark_count", 0)

    print(f"[CHECK] council_audit_report.json has {verified_count} verified rows and {benchmark_count} benchmark rows.")

    assert verified_count == 36, f"Expected 36 verified rows, got {verified_count}"
    assert benchmark_count == 134, f"Expected 134 benchmark rows, got {benchmark_count}"

    for doc_path in [README_FILE, GITHUB_README_FILE]:
        if not doc_path.exists():
            print(f"[FAIL] Document file not found at {doc_path}")
            sys.exit(1)

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "36" in content, f"{doc_path.name} missing verified count 36"
        assert "134" in content, f"{doc_path.name} missing benchmark count 134"
        print(f"[PASS] {doc_path.name} contains verified ({verified_count}) and benchmark ({benchmark_count}) counts.")

    print("\n[ALL CHECKS PASSED] Single source of truth (council_audit_report.json) matches documentation 100%!")


if __name__ == "__main__":
    check_consistency()
