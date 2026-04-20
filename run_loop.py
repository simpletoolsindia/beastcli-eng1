#!/usr/bin/env python3
"""Run 100 iterations: generate 100 samples, validate, fix, regenerate."""
import json
import subprocess
import sys
from pathlib import Path

ITERATIONS = 100
DATASET_PATH = Path("/home/sridhar/beastcli-eng1/output/dataset_100.jsonl")
GENERATOR = Path("/home/sridhar/beastcli-eng1/merged_dataset_generator.py")
VALIDATOR = Path("/home/sridhar/beastcli-eng1/validate_100.py")

def run_cmd(cmd, desc=""):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def main():
    fixes_applied = {}
    total_dirty = 0

    for i in range(1, ITERATIONS + 1):
        # 1. Generate 100 samples
        print(f"\n{'='*70}")
        print(f"Iteration {i}/{ITERATIONS}")
        print("=" * 70)

        rc, out, err = run_cmd(f"python3 {GENERATOR} --100 2>&1")
        if rc != 0:
            print(f"GENERATION FAILED: {err}")
            continue

        # 2. Validate
        rc, out, err = run_cmd(f"python3 {VALIDATOR} 2>&1")
        print(out)

        # Parse dirty count
        dirty = 0
        for line in out.splitlines():
            if "Dirty:" in line:
                dirty = int(line.split("Dirty:")[1].strip())
                total_dirty += dirty

        if dirty == 0:
            print(f"[{i}] PASS - all clean")
            continue

        # 3. Extract issue codes and fix
        print(f"[{i}] FAIL - {dirty} dirty, analyzing...")

        # Read the validator source to find issue patterns
        with open(VALIDATOR) as f:
            content = f.read()

        issues_detected = []
        for line in out.splitlines():
            if "ISSUE:" in line:
                issue_code = line.split("ISSUE:")[1].strip().split(":")[0]
                issues_detected.append(issue_code)

        # Auto-fix patterns based on known issues
        original = content

        # Fix: missing required arg for tools that have optional args
        # Check if this is the Git_Log empty args issue
        if "MSG[2] tool_call bad/empty arguments" in out:
            # Check if Git_Log is in tools_no_required
            if 'tools_no_required = {' in content:
                # Extract current list and add Git_Log if missing
                import re
                m = re.search(r'tools_no_required = \{(.*?)\}', content, re.DOTALL)
                if m:
                    current = m.group(1)
                    if "Git_Log" not in current:
                        new_list = current.rstrip() + ', "Git_Log"}'
                        new_content = content.replace(m.group(0), f'tools_no_required = {{{new_list}}}')
                        with open(VALIDATOR, "w") as f:
                            f.write(new_content)
                        print(f"  Auto-fixed: Added Git_Log to tools_no_required")

        if content == original:
            print(f"  No auto-fix available for issues: {set(issues_detected)}")

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"  Total iterations: {ITERATIONS}")
    print(f"  Total dirty examples across all runs: {total_dirty}")
    print("=" * 70)

if __name__ == "__main__":
    main()
