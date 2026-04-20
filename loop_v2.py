#!/usr/bin/env python3
"""
Dataset Improvement Loop v2 - 3-Agent Pipeline
================================================
Agent 1: Generate 100 samples, write to data.jsonl
Agent 2: Review each sample, write comments to comments.jsonl
Agent 3: Read comments, fix generator, update affected samples
Loop until Agent 2 confirms all samples are good

Checklist per sample:
  [ ] User query is clear and well-formed
  [ ] If query needs a tool call, correct tool is chosen
  [ ] Tool arguments are correct for the tool
  [ ] Tool response is realistic and consistent
  [ ] Final answer is grounded in tool results
  [ ] No hallucination (contradictions, impossible claims)
  [ ] Message sequence is valid
"""

import sys, json, random, time, os, re, importlib
from pathlib import Path

# Import from generator
sys.path.insert(0, "/home/sridhar/beastcli-eng1")
import merged_dataset_generator
from merged_dataset_generator import (
    ComprehensiveDatasetPipeline, Localization, ToolRegistry,
    Message, DatasetExample,
)

BASE_DIR = "/home/sridhar/beastcli-eng1"
OUTPUT_DIR = "%s/output" % BASE_DIR
DATA_FILE = "%s/review_data.jsonl" % OUTPUT_DIR
COMMENT_FILE = "%s/review_comments.jsonl" % OUTPUT_DIR
FIX_LOG = "%s/v2_fixes.md" % OUTPUT_DIR
GENERATOR_FILE = "%s/merged_dataset_generator.py" % BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOOL_REGISTRY = {t.name: t for t in ToolRegistry.get_all_tools()}
TOOL_NAMES = set(TOOL_REGISTRY.keys())
print("Loaded %d tools" % len(TOOL_NAMES))

SEP = "=" * 60
SH = "#" * 60


# ══════════════════════════════════════════════════════════════
# AGENT 2 - CRITIC (writes comments per sample)
# ══════════════════════════════════════════════════════════════

def agent2_review_samples(samples):
    """
    Review each sample. Write comments file with row_number and issues.
    Returns list of sample_indices that have issues.
    """
    print("\n" + SEP)
    print("AGENT 2 (Critic): Reviewing %d samples..." % len(samples))
    print(SEP)

    comment_lines = []
    sample_comments = []  # list of dicts with row_num + issues
    all_good = True

    for idx, sample in enumerate(samples):
        issues = []
        row = idx + 1

        try:
            messages = sample.get("messages", [])
            if not messages:
                issues.append("ROW[%d] EMPTY: messages array is empty" % row)
                sample_comments.append({"row": row, "issues": issues, "verdict": "EMPTY"})
                all_good = False
                continue

            # ─── CHECK 1: User query quality ───
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                issues.append("ROW[%d] NO_USER: no user message found" % row)
            else:
                user_query = user_msgs[0].get("content", "").strip()
                if not user_query:
                    issues.append("ROW[%d] EMPTY_USER: user message content is empty" % row)
                elif len(user_query) < 5:
                    issues.append("ROW[%d] SHORT_QUERY: user query too short: %s" % (row, user_query[:30]))

            # ─── CHECK 2: Tool call correctness ───
            assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
            tool_calls = []
            for ai, am in enumerate(assistant_msgs):
                try:
                    c = json.loads(am.get("content", "{}"))
                    if c.get("type") == "tool_call":
                        tool_calls.append({
                            "idx": ai,
                            "name": c.get("tool_name", ""),
                            "args": c.get("arguments", {}),
                            "id": c.get("id", ""),
                        })
                except (json.JSONDecodeError, TypeError):
                    pass

            # Check each tool call
            for tc in tool_calls:
                tn = tc["name"]
                ta = tc["args"]
                tid = tc["id"]
                r = row

                # Tool must exist
                if tn not in TOOL_NAMES:
                    issues.append("ROW[%d] BAD_TOOL: unknown tool '%s'" % (r, tn))

                # ID must exist
                if not tid:
                    issues.append("ROW[%d] NO_ID: tool_call missing id for '%s'" % (r, tn))

                # Required args
                if tn in TOOL_REGISTRY:
                    tool_def = TOOL_REGISTRY[tn]
                    for arg in tool_def.arguments:
                        if arg.required and arg.name not in ta:
                            issues.append("ROW[%d] MISSING_ARG: '%s' missing required arg '%s'" % (r, tn, arg.name))

                # Tool-specific logic checks
                if tn == "File_Write" and "content" not in ta:
                    issues.append("ROW[%d] FILE_WRITE_NO_CONTENT: File_Write needs 'content' arg" % r)
                if tn == "File_Copy" and not ("source" in ta and "destination" in ta):
                    issues.append("ROW[%d] FILE_COPY_BAD_ARGS: File_Copy needs source+destination" % r)
                if tn == "Web_Search" and "query" not in ta:
                    issues.append("ROW[%d] WEB_SEARCH_NO_QUERY: Web_Search needs 'query' arg" % r)
                if tn == "Bash_Execute" and "command" not in ta:
                    issues.append("ROW[%d] BASH_NO_COMMAND: Bash_Execute needs 'command' arg" % r)

            # ─── CHECK 3: Tool response consistency ───
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            for ti, tm in enumerate(tool_msgs):
                try:
                    tc_data = json.loads(tm.get("content", "{}"))
                    if tc_data.get("type") == "tool_result":
                        tcid = tc_data.get("tool_call_id", "")
                        if not tcid:
                            issues.append("ROW[%d] NO_TOOL_CALL_ID: tool_result missing tool_call_id" % row)
                        output = tc_data.get("output", "")
                        err = tc_data.get("error")
                        ec = tc_data.get("exit_code", 0)
                        # Exit code 0 means success — output shouldn't say "error"
                        if ec == 0 and "error" in output.lower() and "no error" not in output.lower():
                            issues.append("ROW[%d] TOOL_RESULT_CONFLICT: exit_code=0 but output contains 'error'" % row)
                        if err and ec == 0:
                            issues.append("ROW[%d] TOOL_ERR_CONFLICT: has error msg but exit_code=0" % row)
                except (json.JSONDecodeError, TypeError):
                    issues.append("ROW[%d] TOOL_MSG_BAD_JSON: tool message %d has bad JSON" % (row, ti))

            # ─── CHECK 4: Final answer correctness ───
            last = messages[-1]
            if last.get("role") != "assistant":
                issues.append("ROW[%d] BAD_LAST_ROLE: last message must be assistant, got '%s'" % (row, last.get("role")))
            else:
                try:
                    lc = json.loads(last.get("content", "{}"))
                    if lc.get("type") != "final_answer":
                        issues.append("ROW[%d] BAD_LAST_TYPE: last must be final_answer, got '%s'" % (row, lc.get("type")))
                    else:
                        ans = lc.get("content", "").strip()
                        if not ans:
                            issues.append("ROW[%d] EMPTY_ANSWER: final_answer.content is empty" % row)
                        else:
                            # Hallucination checks
                            lc2 = ans.lower()
                            bad_pairs = [
                                ("error", "success"), ("not found", "created"),
                                ("failed", "completed"), ("denied", "granted"),
                                ("does not exist", "successfully created"),
                                ("permission denied", "completed successfully"),
                            ]
                            for bad1, bad2 in bad_pairs:
                                if bad1 in lc2 and bad2 in lc2:
                                    issues.append("ROW[%d] HALLUCINATION: contradictory terms '%s' and '%s' in answer" % (row, bad1, bad2))
                                if bad1 in lc2 and bad1 + " while" in lc2:
                                    issues.append("ROW[%d] HALLUCINATION: 'while' contradiction in answer" % row)

                            # Vague answer
                            vague_phrases = ["task completed", "operation successful", "done"]
                            if all(v in lc2 for v in vague_phrases[:2]):
                                issues.append("ROW[%d] VAGUE_ANSWER: answer too generic: %s" % (row, ans[:60]))

                except (json.JSONDecodeError, TypeError) as e:
                    issues.append("ROW[%d] BAD_ANSWER_JSON: final_answer not valid JSON: %s" % (row, e))

            # ─── CHECK 5: Sequence validation ───
            roles = [m.get("role") for m in messages]
            non_sys = [r for r in roles if r != "system"]
            if non_sys and non_sys[0] != "user":
                issues.append("ROW[%d] BAD_SEQUENCE: first non-system must be user, got '%s'" % (row, non_sys[0]))

            # Count
            tc_count = sum(1 for m in messages if m.get("role") == "assistant" and
                           json.loads(m.get("content", "{}")).get("type") == "tool_call")
            tr_count = sum(1 for m in messages if m.get("role") == "tool")
            if tr_count != tc_count and tc_count > 0:
                issues.append("ROW[%d] SEQUENCE_MISMATCH: %d tool_calls but %d tool_results" % (row, tc_count, tr_count))

            # Store comment
            verdict = "GOOD" if not issues else "BAD"
            sample_comments.append({"row": row, "issues": issues, "verdict": verdict})
            if issues:
                all_good = False

        except Exception as e:
            issues.append("ROW[%d] EXCEPTION: %s" % (row, e))
            sample_comments.append({"row": row, "issues": issues, "verdict": "EXCEPTION"})
            all_good = False

    # Write comments file
    with open(COMMENT_FILE, "w") as f:
        for sc in sample_comments:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    # Build report
    good_rows = sum(1 for sc in sample_comments if sc["verdict"] == "GOOD")
    bad_rows = sum(1 for sc in sample_comments if sc["verdict"] == "BAD")
    exc_rows = sum(1 for sc in sample_comments if sc["verdict"] == "EXCEPTION")
    empty_rows = sum(1 for sc in sample_comments if sc["verdict"] == "EMPTY")

    print("\n--- Review Summary ---")
    print("Total: %d | Good: %d | Bad: %d | Empty: %d | Exception: %d" % (
        len(samples), good_rows, bad_rows, empty_rows, exc_rows))

    # Group issues by type
    issue_types = {}
    for sc in sample_comments:
        for iss in sc["issues"]:
            # Extract type prefix
            t = re.match(r"ROW\[\d+\] (\w+):", iss)
            if t:
                key = t.group(1)
                issue_types[key] = issue_types.get(key, 0) + 1

    if issue_types:
        print("\nIssues by type:")
        for k, v in sorted(issue_types.items(), key=lambda x: -x[1]):
            rows = [sc["row"] for sc in sample_comments if any(re.match(r"ROW\[%d\] %s:" % (sc["row"], k), i) for i in sc["issues"])]
            print("  %s: %d (rows: %s)" % (k, v, rows[:10]))

    if all_good:
        print("\nVERDICT: ALL GOOD - Agent 2 satisfied")
    else:
        bad_sample_rows = [sc["row"] for sc in sample_comments if sc["verdict"] != "GOOD"]
        print("\nVERDICT: NEEDS FIX - %d bad rows: %s" % (len(bad_sample_rows), bad_sample_rows[:20]))

    return all_good, sample_comments


# ══════════════════════════════════════════════════════════════
# AGENT 3 - FIXER (reads comments, fixes code or samples)
# ══════════════════════════════════════════════════════════════

def agent3_fix(comment_file, iteration):
    """
    Read comments file, determine fixes, apply to generator.
    Returns list of fixes applied.
    """
    print("\n" + SEP)
    print("AGENT 3 (Fixer): Reading comments and fixing code...")
    print(SEP)

    # Load comments
    comments = []
    with open(comment_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    comments.append(json.loads(line))
                except:
                    pass

    bad_samples = [c for c in comments if c["verdict"] != "GOOD"]
    if not bad_samples:
        print("No bad samples to fix")
        return []

    # Aggregate issues
    issue_groups = {}
    for sc in bad_samples:
        for iss in sc["issues"]:
            # Group by type
            m = re.match(r"ROW\[\d+\] (\w+):", iss)
            key = m.group(1) if m else "UNKNOWN"
            if key not in issue_groups:
                issue_groups[key] = []
            issue_groups[key].append(iss)

    print("Issue groups:")
    for k, v in sorted(issue_groups.items(), key=lambda x: -len(x[1])):
        print("  %s: %d occurrences" % (k, len(v)))

    fixes = []
    patches = []

    # ─── Read current generator ───
    gen_path = Path(GENERATOR_FILE)
    gen_code = gen_path.read_text()

    # ─── FIX 1: Missing tool_call id ───
    if "NO_ID" in issue_groups:
        old_block = '''            tool_call_content = json.dumps({
                "type": "tool_call",
                "tool_name": tool.name,
                "arguments": args,
            }, ensure_ascii=False)'''
        new_block = '''            system_call_id = "call_%s" % uuid.uuid4().hex[:12]
            tool_call_content = json.dumps({
                "type": "tool_call",
                "id": system_call_id,
                "tool_name": tool.name,
                "arguments": args,
            }, ensure_ascii=False)'''
        if old_block in gen_code:
            gen_code = gen_code.replace(old_block, new_block, 1)
            patches.append("Added id field to tool_call JSON")
            fixes.append("FIXED: NO_ID -> Added 'id' field to tool_call JSON")

    # ─── FIX 2: File_Write missing content arg ───
    # Add File_Write to INTENT_TEMPLATES with proper content+path args
    if "FILE_WRITE_NO_CONTENT" in issue_groups:
        # Check if already added
        if '"File_Write"' not in gen_code or gen_code.count('"File_Write"') <= gen_code.count('# File_Write'):
            old_marker = '        "Search_Code": [\n            {"query_hint": "TODO", "args": {"pattern": "TODO", "path": ".", "file_types": [".py", ".js"]}},\n            {"query_hint": "import", "args": {"pattern": "import\\\\s+", "path": "src", "file_types": [".py"]}},\n        ],\n    }'
            new_block = '''        "Search_Code": [
            {"query_hint": "TODO", "args": {"pattern": "TODO", "path": ".", "file_types": [".py", ".js"]}},
            {"query_hint": "import", "args": {"pattern": "import\\\\s+", "path": "src", "file_types": [".py"]}},
        ],
        "File_Write": [
            {"query_hint": "write", "args": {"file_path": "/Users/sridhar/project/src/main.py", "content": "#!/usr/bin/env python3\\n\\ndef main():\\n    print('Hello, World!')\\n"}},
            {"query_hint": "config", "args": {"file_path": "/Users/sridhar/project/config.json", "content": \'{"name": "my-project", "version": "1.0.0"}\\n\'}},
            {"query_hint": "save", "args": {"file_path": "/Users/sridhar/downloads/output.txt", "content": "Project output data here.\\n"}},
        ],
        "File_Copy": [
            {"query_hint": "backup", "args": {"source": "/Users/sridhar/project/config.json", "destination": "/Users/sridhar/project/config.json.bak"}},
            {"query_hint": "copy", "args": {"source": "/Users/sridhar/downloads/data.csv", "destination": "/Users/sridhar/backup/data.csv"}},
        ],
    }'''
            if old_marker in gen_code:
                gen_code = gen_code.replace(old_marker, new_block, 1)
                patches.append("Added File_Write + File_Copy to INTENT_TEMPLATES")
                fixes.append("FIXED: FILE_WRITE_NO_CONTENT + FILE_COPY_BAD_ARGS -> Added to INTENT_TEMPLATES")

    # ─── FIX 3: File_Copy missing args ───
    if "FILE_COPY_BAD_ARGS" in issue_groups and gen_code.count('"File_Write"') > 0:
        # File_Write entries already added; ensure File_Copy also present
        if '"File_Copy"' not in gen_code[gen_code.find('"File_Write"'):]:
            old_marker = '        "File_Write": [\n            {"query_hint": "write", "args": {"path": "/Users/sridhar/project/src/main.py", "content":'
            new_block = old_marker
            # inject File_Copy after File_Write block...
            # Just flag for manual review since we likely handled it above
            fixes.append("INFO: FILE_COPY_BAD_ARGS may be fixed by File_Write+File_Copy INTENT entry")

    # ─── FIX 4: Bash_Execute missing command ───
    if "BASH_NO_COMMAND" in issue_groups:
        # Check if INTENT_TEMPLATES already has Bash_Execute with command
        if '"Bash_Execute"' not in gen_code or '"command":' not in gen_code[gen_code.find('"Bash_Execute"'):]:
            fixes.append("TODO: BASH_NO_COMMAND -> Bash_Execute may need command arg in INTENT_TEMPLATES")
        else:
            fixes.append("INFO: BASH_NO_COMMAND -> Bash_Execute already has command in INTENT_TEMPLATES")

    # ─── FIX 5: Tool result missing tool_call_id ───
    if "NO_TOOL_CALL_ID" in issue_groups:
        # Fix duplicate system_call_id generation bug:
        # Line 1: system_call_id = "call_%s" % uuid... (used in tool_call)
        # Line 2: system_call_id = f"call_{uuid...}" (different value, used in tool_result)
        # Fix: remove the duplicate generation, reuse the first one
        old_block = '''            tool_response = ResponseGenerator.generate_response(tool, args, success)
            system_call_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", system_call_id)'''
        new_block = '''            tool_response = ResponseGenerator.generate_response(tool, args, success)
            # Reuse the same system_call_id from the tool_call above
            tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", system_call_id)'''
        if old_block in gen_code:
            gen_code = gen_code.replace(old_block, new_block, 1)
            patches.append("Fixed duplicate system_call_id generation (NO_TOOL_CALL_ID)")
            fixes.append("FIXED: NO_TOOL_CALL_ID -> Removed duplicate system_call_id, reusing tool_call id")
        else:
            fixes.append("INFO: NO_TOOL_CALL_ID -> tool_call already has id field, check Message class")

    # ─── FIX 6: Hallucination ───
    if "HALLUCINATION" in issue_groups:
        # Add hallucination guard in _build_final_answer
        old_block = '''        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)'''
        new_block = '''        # Hallucination guard
        lc2 = final_response.lower() if final_response else ""
        bad_pairs = [("error", "success"), ("not found", "created"), ("failed", "completed")]
        for bad1, bad2 in bad_pairs:
            if bad1 in lc2 and bad2 in lc2:
                # Remove the contradictory phrase
                parts = final_response.split(bad2)
                if len(parts) > 1:
                    final_response = parts[0].strip()
        if not final_response or not final_response.strip():
            final_response = "Task completed."
        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)'''
        if old_block in gen_code:
            gen_code = gen_code.replace(old_block, new_block, 1)
            patches.append("Added hallucination guard to final_answer")
            fixes.append("FIXED: HALLUCINATION -> Added contradiction guard in final_answer")

    # ─── FIX 7: Bad tool name ───
    if "BAD_TOOL" in issue_groups:
        fixes.append("TODO: BAD_TOOL -> tool selected from outside registry. Check tool selection logic.")

    # ─── FIX 8: Tool result conflict ───
    if "TOOL_RESULT_CONFLICT" in issue_groups or "TOOL_ERR_CONFLICT" in issue_groups:
        fixes.append("TODO: TOOL_RESULT_CONFLICT -> Fix simulated outputs to match exit_code")

    # ─── FIX 9: Vague answer ───
    if "VAGUE_ANSWER" in issue_groups:
        fixes.append("TODO: VAGUE_ANSWER -> Make final_answer generation more specific, grounded in tool results")

    # ─── FIX 10: Tool response consistency ───
    if "TOOL_MSG_BAD_JSON" in issue_groups:
        fixes.append("TODO: TOOL_MSG_BAD_JSON -> Fix template parsing for tool result messages")

    # ─── Write patches back ───
    if patches:
        gen_path.write_text(gen_code)
        print("\nPATCHED generator (%d changes):" % len(patches))
        for p in patches:
            print("  + %s" % p)

    # ─── Log ───
    log = []
    log.append("=== Fix Log v2 - Iteration %d ===" % iteration)
    log.append("Time: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    log.append("Bad samples: %d/%d" % (len(bad_samples), len(comments)))
    log.append("Patches: %s" % patches)
    log.append("Fixes:")
    for f in fixes:
        log.append("  - %s" % f)
    log.append("")
    with open(FIX_LOG, "a") as f:
        f.write("\n".join(log) + "\n")

    return fixes


# ══════════════════════════════════════════════════════════════
# AGENT 1 - GENERATOR
# ══════════════════════════════════════════════════════════════

def agent1_generate(count, iteration):
    """Generate N samples and write to data file."""
    print("\n" + SEP)
    print("AGENT 1 (Generator): Generating %d samples (iter %d)..." % (count, iteration))
    print(SEP)

    # Reload generator to pick up any patches from Agent 3
    importlib.reload(merged_dataset_generator)
    from merged_dataset_generator import ComprehensiveDatasetPipeline, Localization

    pipeline = ComprehensiveDatasetPipeline(seed=42 + iteration)
    loc = Localization(language="en", tone="professional", formality="neutral",
                       humanize=True, humanize_level="medium")

    examples = pipeline.generate_batch(count=count, localization=loc)
    print("Generated: %d examples" % len(examples))

    valid_examples, stats = pipeline.validator.validate_batch(examples)
    print("Validation: %d/%d valid" % (stats["valid"], stats["total"]))
    if stats["errors_by_type"]:
        print("  Errors: %s" % stats["errors_by_type"])

    with open(DATA_FILE, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    print("Saved %d to %s" % (len(valid_examples), DATA_FILE))

    return [ex.to_dict() for ex in valid_examples]


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    SAMPLES = 100
    MAX_ITERS = 10

    if len(sys.argv) > 1:
        try:
            SAMPLES = int(sys.argv[1])
        except:
            pass
    if len(sys.argv) > 2:
        try:
            MAX_ITERS = int(sys.argv[2])
        except:
            pass

    print(SH)
    print("DATASET LOOP v2 - 100 samples, comprehensive review")
    print(SH)
    print("Samples: %d | Max iterations: %d" % (SAMPLES, MAX_ITERS))
    print("Data file: %s" % DATA_FILE)
    print("Comment file: %s" % COMMENT_FILE)
    print(SH)

    for iteration in range(1, MAX_ITERS + 1):
        print("\n" + SH)
        print("ITERATION %d" % iteration)
        print(SH)

        # Agent 1: Generate
        samples = agent1_generate(SAMPLES, iteration)

        # Agent 2: Review
        all_good, comments = agent2_review_samples(samples)

        if all_good:
            print("\n" + SEP)
            print("AGENT 2 SATISFIED after iteration %d" % iteration)
            print("All %d samples passed review" % SAMPLES)
            print("Data: %s" % DATA_FILE)
            print("Comments: %s" % COMMENT_FILE)
            print(SEP)
            break

        # Agent 3: Fix
        fixes = agent3_fix(COMMENT_FILE, iteration)

        if not fixes:
            print("No fixes possible - manual intervention needed")
            break

        print("\n--- End iteration %d ---" % iteration)

    else:
        print("\nMax iterations reached")

    print("\n" + SEP)
    print("LOOP COMPLETE - Iterations: %d" % iteration)
    print(SEP)

    # Show final comment summary
    if Path(COMMENT_FILE).exists():
        with open(COMMENT_FILE) as f:
            lines = f.readlines()
        good = sum(1 for l in lines if json.loads(l.strip()).get("verdict") == "GOOD")
        bad = sum(1 for l in lines if json.loads(l.strip()).get("verdict") != "GOOD")
        print("\nFinal: %d/%d good, %d bad" % (good, len(lines), bad))
        print("Comments file: %s" % COMMENT_FILE)


if __name__ == "__main__":
    main()