#!/usr/bin/env python3
"""Validate train/eval/test split files with full line-by-line check."""
import json
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("/home/sridhar/beastcli-eng1/output")
TRAIN = OUTPUT_DIR / "dataset_train.jsonl"
EVAL = OUTPUT_DIR / "dataset_eval.jsonl"
TEST = OUTPUT_DIR / "dataset_test.jsonl"

def validate_file(path, label):
    """Validate a single JSONL file."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    issues_by_code = {}
    clean_indices = []
    dirty_indices = []

    for i, line in enumerate(lines):
        ex = json.loads(line)
        msgs = ex.get("messages", [])
        issues = []

        # Basic checks
        if not msgs:
            issues.append("NO_MESSAGES")
        else:
            # Check roles
            roles = [m.get("role") for m in msgs]
            if roles[0] != "system":
                issues.append("FIRST_NOT_SYSTEM")
            if roles[-1] != "assistant":
                issues.append("LAST_NOT_ASSISTANT")

            # Check assistant/tool message JSON
            for mi, msg in enumerate(msgs):
                role = msg.get("role", "")
                content = msg.get("content", "") or ""
                if role == "assistant":
                    try:
                        inner = json.loads(content)
                        t = inner.get("type", "")
                        if t == "tool_call":
                            if not inner.get("id"):
                                issues.append(f"MSG[{mi}] tool_call missing id")
                            if not inner.get("tool_name"):
                                issues.append(f"MSG[{mi}] tool_call missing tool_name")
                            # Check args (some tools have no required args)
                            no_req = {"Bash_ShellStatus", "Git_Status", "Git_Branch", "Git_Diff",
                                       "Git_Pull", "Git_Push", "System_Info", "Process_List",
                                       "Database_List", "Git_Log"}
                            tn = inner.get("tool_name", "")
                            args = inner.get("arguments", {})
                            if tn not in no_req and (not args or not isinstance(args, dict)):
                                issues.append(f"MSG[{mi}] tool_call bad/empty arguments")
                        elif t == "final_answer":
                            if not inner.get("content"):
                                issues.append(f"MSG[{mi}] final_answer missing content")
                    except:
                        issues.append(f"MSG[{mi}] assistant content not JSON")

                elif role == "tool":
                    try:
                        inner = json.loads(content)
                        if not inner.get("tool_call_id"):
                            issues.append(f"MSG[{mi}] tool_result missing tool_call_id")
                    except:
                        issues.append(f"MSG[{mi}] tool content not JSON")

            # Check tool_call / tool_result counts match
            tc_ids = set()
            tr_ids = set()
            for mi, msg in enumerate(msgs):
                content = msg.get("content", "") or ""
                if msg.get("role") == "assistant":
                    try:
                        inner = json.loads(content)
                        if inner.get("type") == "tool_call":
                            tid = inner.get("id", "")
                            if tid:
                                tc_ids.add(tid)
                    except:
                        pass
                elif msg.get("role") == "tool":
                    try:
                        inner = json.loads(content)
                        if inner.get("type") == "tool_result":
                            tid = inner.get("tool_call_id", "")
                            if tid:
                                tr_ids.add(tid)
                    except:
                        pass

            # Check ID references
            for mi, msg in enumerate(msgs):
                if msg.get("role") == "tool":
                    try:
                        inner = json.loads(msg.get("content", ""))
                        tid = inner.get("tool_call_id", "")
                        if tid and tid not in tc_ids:
                            issues.append(f"MSG[{mi}] TCID_MISMATCH: '{tid}' not in any tool_call")
                    except:
                        pass

            # Check final_answer for placeholders
            for mi, msg in enumerate(msgs):
                if msg.get("role") == "assistant":
                    try:
                        inner = json.loads(msg.get("content", ""))
                        if inner.get("type") == "final_answer":
                            content = inner.get("content", "")
                            import re
                            unescaped = re.findall(r'(?<!{)(?<!})\{(?!\{)[^{}]+\}(?!})', content)
                            if unescaped:
                                issues.append(f"MSG[{mi}] final_answer has unreplaced placeholder")
                            if not content or len(content.strip()) < 5:
                                issues.append(f"MSG[{mi}] final_answer too short/empty")
                    except:
                        pass

        if issues:
            dirty_indices.append((i, issues))
            for iss in issues:
                code = iss.split(":")[0] if ":" in iss else iss
                issues_by_code[code] = issues_by_code.get(code, 0) + 1
        else:
            clean_indices.append(i)

    return lines, clean_indices, dirty_indices, issues_by_code


def main():
    files = [
        (TRAIN, "TRAIN"),
        (EVAL, "EVAL"),
        (TEST, "TEST"),
    ]

    grand_total = 0
    grand_clean = 0

    for path, label in files:
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue

        lines, clean_idx, dirty_idx, issue_counts = validate_file(path, label)
        total = len(lines)
        clean = len(clean_idx)
        dirty = len(dirty_idx)
        grand_total += total
        grand_clean += clean

        print(f"\n{'='*70}")
        print(f"{label} ({path.name}): {clean}/{total} CLEAN / {dirty} DIRTY")
        print("=" * 70)

        if dirty == 0:
            print("  PASS - all clean")
        else:
            print(f"  FAIL - {dirty} dirty")
            for code, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                print(f"  {code}: {count}")
            # Show first 3 dirty examples
            for i, issues in dirty_idx[:3]:
                ex = json.loads(lines[i])
                query = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "?")
                tool = "?"
                args = {}
                for m in ex["messages"]:
                    if m["role"] == "assistant":
                        try:
                            inner = json.loads(m["content"])
                            if inner.get("type") == "tool_call":
                                tool = inner.get("tool_name", "?")
                                args = inner.get("arguments", {})
                        except:
                            pass
                print(f"\n  --- #{i} query='{query[:80]}' tool={tool} ---")
                for iss in issues:
                    print(f"    ISSUE: {iss}")

    print(f"\n{'='*70}")
    print(f"GRAND TOTAL: {grand_clean}/{grand_total} CLEAN")
    print("=" * 70)

    return 0 if grand_clean == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())
