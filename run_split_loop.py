#!/usr/bin/env python3
"""Run 5 iterations: generate train/eval/test split, validate all 3 files."""
import subprocess
import sys
import time

ITERATIONS = 5
GENERATOR = "merged_dataset_generator.py"
VALIDATOR = "validate_split.py"
FILES = [
    ("output/dataset_train.jsonl", "TRAIN", 70),
    ("output/dataset_eval.jsonl", " EVAL", 15),
    ("output/dataset_test.jsonl", " TEST", 15),
]

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

def main():
    grand_clean = 0
    grand_total = 0

    for it in range(1, ITERATIONS + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {it}/{ITERATIONS}")
        print("=" * 70)

        # Generate
        rc, out, err = run_cmd(f"python3 {GENERATOR} --100 2>&1 | grep -v DeprecationWarning")
        if rc != 0:
            print(f"FAILED to generate: {err}")
            continue

        # Validate all 3 files
        rc, out, err = run_cmd(f"python3 {VALIDATOR} 2>&1")
        print(out)

        # Parse results
        for line in out.splitlines():
            if "GRAND TOTAL:" in line:
                parts = line.split("GRAND TOTAL:")[1].strip()
                print(f"\n  --> {parts}")

        # Count dirty
        iteration_dirty = 0
        for label, path, expected in FILES:
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            rc, out, _ = run_cmd(f"python3 -c 'import json; print(len([l for l in open(\"{path}\").readlines() if l.strip()]))'")
            total = int(out.strip())
            clean = 0
            for line in out.splitlines():
                if "CLEAN" in line:
                    pass  # already parsed above
            print(f"  {label}: {len(lines)} lines (expected {expected})")

    print(f"\n{'='*70}")
    print(f"DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
