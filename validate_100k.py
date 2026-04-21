#!/usr/bin/env python3
"""
Line-by-line validator for hq_100k split files.
Validates JSON parse, schema, tool_call_id refs, final_answer quality.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

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


def check(line: str) -> list[str]:
    issues = []
    try:
        ex = json.loads(line)
    except Exception as e:
        return [f"JSON_PARSE_ERROR: {e}"]

    msgs = ex.get("messages", [])
    if not msgs:
        return ["NO_MESSAGES"]

    non_sys = [m for m in msgs if m.get("role") != "system"]
    if non_sys and non_sys[0].get("role") != "user":
        issues.append("FIRST_NON_SYSTEM_NOT_USER")
    if non_sys and non_sys[-1].get("role") != "assistant":
        issues.append("LAST_NON_SYSTEM_NOT_ASSISTANT")

    user_msgs = [m for m in msgs if m.get("role") == "user"]
    if user_msgs and not user_msgs[0].get("content", "").strip():
        issues.append("EMPTY_USER_QUERY")

    tc_list, tr_list = [], []
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
                        issues.append(f"MSG[{mi}] TOOL_CALL_MISSING_ID")
                    if not tn:
                        issues.append(f"MSG[{mi}] TOOL_CALL_MISSING_TOOL_NAME")
                    elif tn not in VALID_TOOLS:
                        issues.append(f"MSG[{mi}] UNKNOWN_TOOL: {tn}")
                    elif tn not in TOOLS_NO_REQUIRED and (not args or not isinstance(args, dict)):
                        issues.append(f"MSG[{mi}] BAD_ARGUMENTS")
                    if tn in ("File_Read", "File_Write", "File_Search", "File_List",
                              "File_Delete", "File_Copy", "Bash_Execute", "Python_Run",
                              "Git_Commit", "Web_Search", "Web_Fetch", "Python_Test",
                              "Search_Code", "Search_Replace", "Database_Query"):
                        reqs = {
                            "File_Read": ["file_path"], "File_Write": ["file_path", "content"],
                            "File_Search": ["pattern"], "File_List": ["directory"],
                            "File_Delete": ["path"], "File_Copy": ["source", "destination"],
                            "Bash_Execute": ["command"], "Python_Run": ["code"],
                            "Git_Commit": ["message"], "Web_Search": ["query"],
                            "Web_Fetch": ["url"], "Python_Test": ["file_path"],
                            "Search_Code": ["pattern"], "Search_Replace": ["path", "search", "replace"],
                            "Database_Query": ["query"],
                        }
                        for req in reqs.get(tn, []):
                            if req not in args:
                                issues.append(f"MSG[{mi}] MISSING_ARG_{req}")
                    tc_list.append((mi, tc_id, tn, args))
                elif t == "final_answer":
                    if not inner.get("content"):
                        issues.append(f"MSG[{mi}] FINAL_ANSWER_EMPTY")
            except json.JSONDecodeError:
                issues.append(f"MSG[{mi}] ASST_NOT_JSON")

        elif role == "tool":
            tcid = msg.get("tool_call_id", "")
            try:
                json.loads(content)
            except json.JSONDecodeError:
                issues.append(f"MSG[{mi}] TOOL_NOT_JSON")
            tr_list.append((mi, tcid))

    # Count match
    if len(tc_list) != len(tr_list):
        issues.append(f"COUNT_MISMATCH: {len(tc_list)}/{len(tr_list)}")

    # ID reference
    tc_ids = {tc[1] for tc in tc_list if tc[1]}
    for mi, tcid in tr_list:
        if tcid and tcid not in tc_ids:
            issues.append(f"MSG[{mi}] TCID_MISMATCH")

    # Final answer quality
    for mi, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            try:
                inner = json.loads(msg.get("content", ""))
                if inner.get("type") == "final_answer":
                    c = inner.get("content", "")
                    if len(c.strip()) < 5:
                        issues.append(f"MSG[{mi}] ANSWER_TOO_SHORT")
                    # Check for unreplaced placeholders (not escaped braces)
                    unesc = re.findall(r'(?<!{)(?<!})\{(?!\{)[^{}]+\}(?!})(?![,}])', c)
                    if unesc:
                        issues.append(f"MSG[{mi}] HAS_PLACEHOLDER")
                    # Check for generic placeholders
                    generic = {"the result", "the output", "as you can see",
                               "there you have it", "done", "here you go"}
                    if c.lower().strip() in generic:
                        issues.append(f"MSG[{mi}] GENERIC_ANSWER")
            except:
                pass

    return issues


def main():
    files = [
        ("output/hq_100k_train.jsonl", "TRAIN"),
        ("output/hq_100k_eval.jsonl",  "EVAL"),
        ("output/hq_100k_test.jsonl",   "TEST"),
    ]

    grand_total = 0
    grand_clean = 0
    all_codes = Counter()

    for path_str, label in files:
        path = Path(path_str)
        if not path.exists():
            print(f"SKIP {label}: not found")
            continue

        print(f"Validating {label}...", end="", flush=True)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        clean, dirty = 0, 0
        for i, line in enumerate(lines):
            issues = check(line)
            if issues:
                dirty += 1
                for iss in issues:
                    code = iss.split(":")[0] if ":" in iss else iss
                    all_codes[code] += 1
            else:
                clean += 1

            if i > 0 and (i + 1) % 10000 == 0:
                print(f" {i+1:,}/{(i+1)+len(lines)-(i+1)}", end="", flush=True)

        total = len(lines)
        grand_total += total
        grand_clean += clean

        status = "PASS" if dirty == 0 else f"FAIL({dirty})"
        print(f" -> {clean:,}/{total:,} CLEAN / {dirty:,} DIRTY [{status}]")

        if dirty > 0:
            print("  Top issues:")
            for code, cnt in all_codes.most_common(10):
                if cnt > (grand_total - grand_clean):
                    pass
            # Re-read codes for this file only
            file_codes = Counter()
            for i2, line2 in enumerate(lines):
                issues2 = check(line2)
                for iss in issues2:
                    code = iss.split(":")[0] if ":" in iss else iss
                    file_codes[code] += 1
            for code, cnt in file_codes.most_common(5):
                print(f"    {code}: {cnt}")

    error_rate = (grand_total - grand_clean) / grand_total * 100 if grand_total else 0
    print(f"\nGRAND: {grand_clean:,}/{grand_total:,} CLEAN")
    print(f"Error rate: {error_rate:.3f}%")
    return 0 if grand_clean == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())
