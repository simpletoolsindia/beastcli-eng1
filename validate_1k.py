#!/usr/bin/env python3
"""Line-by-line validator for 1k split dataset files."""
import json
import re
import sys
from pathlib import Path
from collections import Counter

VALID_TOOLS = {
    "File_Read", "File_Write", "File_Search", "File_List", "File_Delete", "File_Copy",
    "Bash_Execute", "Bash_ShellStatus",
    "Python_Run", "Python_Test", "Node_Run", "JavaScript_Test",
    "Git_Status", "Git_Log", "Git_Commit", "Git_Branch", "Git_Diff", "Git_Pull", "Git_Push",
    "Web_Search", "Web_Fetch", "Web_Screenshot",
    "Search_Code", "Search_Replace",
    "System_Info", "Process_List",
    "Database_Query", "Database_List",
}

TOOLS_NO_REQUIRED = {
    "Bash_ShellStatus", "Git_Status", "Git_Branch", "Git_Diff", "Git_Pull",
    "Git_Push", "System_Info", "Process_List", "Database_List", "Git_Log",
}

def check_example(idx, raw):
    issues = []
    try:
        ex = json.loads(raw)
    except Exception as e:
        return [f"JSON_PARSE_ERROR: {e}"]

    msgs = ex.get("messages", [])
    if not msgs:
        return ["NO_MESSAGES"]

    # 1. Role sequence
    non_system = [m for m in msgs if m.get("role") != "system"]
    if non_system:
        if non_system[0].get("role") != "user":
            issues.append(f"FIRST_NON_SYSTEM not user")
        if non_system[-1].get("role") != "assistant":
            issues.append(f"LAST non_system not assistant")

    # 2. User query not empty
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    if user_msgs:
        q = user_msgs[0].get("content", "").strip()
        if not q:
            issues.append("EMPTY_USER_QUERY")

    # 3. Tool call / tool_result analysis
    tool_calls = []  # (msg_idx, tc_id, tool_name, args)
    tool_results = []  # (msg_idx, tr_tcid) — tcid from message level

    for mi, msg in enumerate(msgs):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            try:
                inner = json.loads(content)
                t = inner.get("type", "")
                if t == "tool_call":
                    tc_id = inner.get("id", "")
                    tn = inner.get("tool_name", "")
                    args = inner.get("arguments", {})
                    if not tc_id:
                        issues.append(f"MSG[{mi}] tool_call missing id")
                    if not tn:
                        issues.append(f"MSG[{mi}] tool_call missing tool_name")
                    elif tn not in VALID_TOOLS:
                        issues.append(f"MSG[{mi}] UNKNOWN_TOOL: {tn}")
                    else:
                        if tn not in TOOLS_NO_REQUIRED and (not args or not isinstance(args, dict)):
                            issues.append(f"MSG[{mi}] tool_call bad/empty arguments")
                        # Required args per tool (only for tools with required args)
                        if tn == "File_Read" and "file_path" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: file_path for {tn}")
                        elif tn == "File_Write":
                            for req in ["file_path", "content"]:
                                if req not in args:
                                    issues.append(f"MSG[{mi}] MISSING_ARG: {req} for {tn}")
                        elif tn == "File_Search" and "pattern" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: pattern for {tn}")
                        elif tn == "File_List" and "directory" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: directory for {tn}")
                        elif tn == "File_Delete" and "path" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: path for {tn}")
                        elif tn == "File_Copy":
                            for req in ["source", "destination"]:
                                if req not in args:
                                    issues.append(f"MSG[{mi}] MISSING_ARG: {req} for {tn}")
                        elif tn == "Bash_Execute" and "command" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: command for {tn}")
                        elif tn == "Python_Run" and "code" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: code for {tn}")
                        elif tn == "Git_Commit" and "message" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: message for {tn}")
                        elif tn == "Web_Search" and "query" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: query for {tn}")
                        elif tn == "Web_Fetch" and "url" not in args:
                            issues.append(f"MSG[{mi}] MISSING_ARG: url for {tn}")
                    tool_calls.append((mi, tc_id, tn, args))
                elif t == "final_answer":
                    if not inner.get("content"):
                        issues.append(f"MSG[{mi}] final_answer missing content")
            except json.JSONDecodeError:
                issues.append(f"MSG[{mi}] assistant content not valid JSON")

        elif role == "tool":
            # tool_call_id is at the message level per Unsloth format
            tcid = msg.get("tool_call_id", "")
            try:
                inner = json.loads(content)
                if inner.get("type") != "tool_result":
                    issues.append(f"MSG[{mi}] tool content wrong type")
                if not tcid:
                    issues.append(f"MSG[{mi}] tool_result missing tool_call_id")
            except json.JSONDecodeError:
                issues.append(f"MSG[{mi}] tool content not valid JSON")
            tool_results.append((mi, tcid))

    # 4. Count match
    if len(tool_calls) != len(tool_results):
        issues.append(f"COUNT_MISMATCH: {len(tool_calls)} calls vs {len(tool_results)} results")

    # 5. ID reference
    tc_ids = {tc[1] for tc in tool_calls if tc[1]}
    for mi, tcid in tool_results:
        if tcid and tcid not in tc_ids:
            issues.append(f"MSG[{mi}] TCID_MISMATCH: '{tcid}' not in any tool_call")

    # 6. Final answer quality
    for mi, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            try:
                inner = json.loads(msg.get("content", ""))
                if inner.get("type") == "final_answer":
                    content = inner.get("content", "")
                    if not content or len(content.strip()) < 5:
                        issues.append(f"MSG[{mi}] final_answer too short/empty")
                    unescaped = re.findall(r'(?<!{)(?<!})\{(?!\{)[^{}]+\}(?!})', content)
                    if unescaped:
                        issues.append(f"MSG[{mi}] final_answer has unreplaced placeholder")
            except:
                pass

    return issues


def main():
    files = [
        ("output/dataset_1k_train.jsonl", "TRAIN"),
        ("output/dataset_1k_eval.jsonl",  "EVAL"),
        ("output/dataset_1k_test.jsonl",    "TEST"),
    ]

    grand_total = 0
    grand_clean = 0
    all_issue_codes = Counter()

    for path_str, label in files:
        path = Path(path_str)
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue

        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        clean = []
        dirty = []
        for i, line in enumerate(lines):
            issues = check_example(i, line)
            if issues:
                dirty.append((i, issues))
                for iss in issues:
                    code = iss.split(":")[0] if ":" in iss else iss
                    all_issue_codes[code] += 1
            else:
                clean.append(i)

        total = len(lines)
        c = len(clean)
        d = len(dirty)
        grand_total += total
        grand_clean += c

        print(f"\n{'='*70}")
        print(f"{label}: {c}/{total} CLEAN / {d} DIRTY")
        print("=" * 70)

        if d == 0:
            print("  PASS")
        else:
            print(f"  FAIL ({d} dirty)")
            for code, cnt in sorted(all_issue_codes.items(), key=lambda x: -x[1]):
                if all_issue_codes[code] > 0:
                    pass
            # Show first 5 dirty
            for i, issues in dirty[:5]:
                ex = json.loads(lines[i])
                query = "?"
                tool = "?"
                for m in ex["messages"]:
                    if m["role"] == "user":
                        query = m["content"][:100]
                    if m["role"] == "assistant":
                        try:
                            inner = json.loads(m["content"])
                            if inner.get("type") == "tool_call":
                                tool = inner.get("tool_name", "?")
                        except:
                            pass
                print(f"\n  --- #{i} query='{query}' tool={tool}")
                for iss in issues:
                    print(f"    {iss}")
            if len(dirty) > 5:
                print(f"\n  ... and {len(dirty)-5} more")

            # Print issue summary
            print("\n  Issue summary:")
            for code, cnt in sorted(all_issue_codes.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    print(f"    {code}: {cnt}")

    print(f"\n{'='*70}")
    print(f"GRAND: {grand_clean}/{grand_total} CLEAN")
    error_rate = (grand_total - grand_clean) / grand_total * 100 if grand_total else 0
    print(f"Error rate: {error_rate:.1f}%")
    print("=" * 70)

    return 0 if grand_clean == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())
