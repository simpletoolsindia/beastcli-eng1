#!/usr/bin/env python3
"""Line-by-line analysis of dataset_100.jsonl with deep thoughts."""
import json
import sys
from pathlib import Path
from collections import Counter

INPUT = Path("/home/sridhar/beastcli-eng1/output/dataset_100.jsonl")
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

STATS = {
    "tools": Counter(),
    "categories": Counter(),
    "difficulty": Counter(),
    "tones": Counter(),
    "languages": Counter(),
    "formalities": Counter(),
    "success": Counter(),
    "num_tools": Counter(),
}

def analyze_example(idx, raw):
    thoughts = []
    issues = []

    try:
        ex = json.loads(raw)
    except:
        return ["JSON_PARSE_ERROR"], {"tools": ["PARSE_ERROR"], "categories": ["PARSE_ERROR"], "difficulty": ["PARSE_ERROR"], "tones": ["PARSE_ERROR"], "languages": ["PARSE_ERROR"], "formalities": ["PARSE_ERROR"], "success": ["PARSE_ERROR"], "num_tools": ["PARSE_ERROR"]}

    msgs = ex.get("messages", [])
    loc = ex.get("localization", {})
    meta = ex.get("metadata", {})

    # Stats
    tool_name = meta.get("tool_name", "?")
    difficulty = meta.get("difficulty", "?")
    success = meta.get("success", "?")
    num_tools = meta.get("num_tools", 1)
    tone = loc.get("tone", "?")
    language = loc.get("language", "?")
    formality = loc.get("formality", "?")
    humanize = loc.get("humanize", False)
    humanize_level = loc.get("humanize_level", "?")

    STATS["tools"][tool_name] += 1
    STATS["categories"][meta.get("tool_category", "?")] += 1
    STATS["difficulty"][difficulty] += 1
    STATS["tones"][tone] += 1
    STATS["languages"][language] += 1
    STATS["formalities"][formality] += 1
    STATS["success"][str(success)] += 1
    STATS["num_tools"][num_tools] += 1

    # Tool call analysis
    tool_calls = []
    tool_results = []
    user_query = ""
    final_answer = ""

    for mi, msg in enumerate(msgs):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "user":
            user_query = content
        elif role == "assistant":
            try:
                inner = json.loads(content)
                t = inner.get("type", "")
                if t == "tool_call":
                    tc_id = inner.get("id", "")
                    tc_tool = inner.get("tool_name", "")
                    tc_args = inner.get("arguments", {})
                    tool_calls.append((mi, tc_id, tc_tool, tc_args))
                elif t == "final_answer":
                    final_answer = inner.get("content", "")
            except:
                pass
        elif role == "tool":
            try:
                inner = json.loads(content)
                if inner.get("type") == "tool_result":
                    tool_results.append((mi, inner.get("tool_call_id", ""), inner.get("output", "")))
            except:
                pass

    # Deep thoughts per example
    if tool_calls:
        tc_idx, tc_id, tc_tool, tc_args = tool_calls[0]
        thoughts.append(f"[#{idx}] {language}/{tone}/{formality} | {difficulty} | tool={tc_tool} | success={success}")
        thoughts.append(f"  USER: {user_query[:100]}")
        thoughts.append(f"  ARGS: {json.dumps(tc_args)[:120]}")

        # Semantic alignment check
        q_lower = user_query.lower()
        tool_lower = tc_tool.lower()

        # Check if tool args make sense for the query
        if tc_tool == "File_Read":
            if "file_path" in tc_args:
                fp = tc_args["file_path"]
                if any(x in q_lower for x in ["config", "settings", "env"]):
                    if "config" in fp or "settings" in fp or ".env" in fp:
                        thoughts.append(f"  GOOD: File_Read with config-like path matches query")
                    else:
                        thoughts.append(f"  WARN: query mentions config but path is '{fp}'")
        elif tc_tool == "File_Write":
            if "content" in tc_args and "file_path" in tc_args:
                content_hint = tc_args["content"][:50].replace("\n", " ")
                thoughts.append(f"  WRITE: '{tc_args['file_path']}' <- '{content_hint}...'")
        elif tc_tool == "Bash_Execute":
            if "command" in tc_args:
                cmd = tc_args["command"]
                if "ls" in cmd or "pwd" in cmd or "cat" in cmd:
                    thoughts.append(f"  GOOD: bash command '{cmd}' is simple and correct")
                elif "git" in cmd:
                    thoughts.append(f"  NOTE: bash/git hybrid - command is '{cmd}'")
        elif tc_tool == "Git_Log":
            thoughts.append(f"  NOTE: Git_Log query='{user_query[:80]}'")
        elif tc_tool == "Git_Status":
            thoughts.append(f"  GOOD: Git_Status matches query intent")
        elif tc_tool == "Python_Run":
            code = tc_args.get("code", "")[:80].replace("\n", " ")
            thoughts.append(f"  PY: {code}")

        # Check tool_call_id matches tool_result tool_call_id
        if tool_calls and tool_results:
            tc_id = tool_calls[0][1]
            tr_tcid = tool_results[0][1]
            if tc_id != tr_tcid:
                issues.append(f"ID_MISMATCH: tc_id={tc_id} vs tr_tcid={tr_tcid}")

        # Final answer quality
        if final_answer:
            if len(final_answer) < 20:
                thoughts.append(f"  SHORT_ANSWER: '{final_answer}' ({len(final_answer)} chars)")
            elif "ran successfully" in final_answer.lower() and "json" in tc_args.get("code", "").lower():
                thoughts.append(f"  OK: generic success for Python code")
            else:
                thoughts.append(f"  ANSWER: {final_answer[:100]}")
        else:
            issues.append("NO_FINAL_ANSWER")

        if issues:
            for iss in issues:
                thoughts.append(f"  ISSUE: {iss}")

    return thoughts, {
        "tools": [tool_name],
        "categories": [meta.get("tool_category", "?")],
        "difficulty": [difficulty],
        "tones": [tone],
        "languages": [language],
        "formalities": [formality],
        "success": [str(success)],
        "num_tools": [num_tools],
    }

def main():
    with open(INPUT) as f:
        lines = f.readlines()

    print(f"Line-by-line analysis of {len(lines)} examples")
    print("=" * 80)

    all_thoughts = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        thoughts, _ = analyze_example(i, line)
        all_thoughts.extend(thoughts)

    for t in all_thoughts:
        print(t)

    print("\n" + "=" * 80)
    print("DISTRIBUTION STATS")
    print("=" * 80)
    for key, counter in STATS.items():
        print(f"\n{key}:")
        for val, cnt in counter.most_common():
            pct = cnt / sum(counter.values()) * 100
            bar = "█" * int(pct / 2)
            print(f"  {val:20s}: {cnt:3d} ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    main()
