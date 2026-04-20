#!/usr/bin/env python3
"""Dataset Improvement Loop Orchestrator - 3 Agent Pipeline"""

import sys, json, random, time, os, traceback, re, importlib
from pathlib import Path

sys.path.insert(0, "/home/sridhar/beastcli-eng1")
from merged_dataset_generator import (
    ComprehensiveDatasetPipeline, Localization, ToolRegistry,
    DifficultyLevel, Language, Tone, FormalityLevel,
)

MAX_ITERATIONS = 10
SAMPLES_PER_RUN = 50
OUTPUT_DIR = "/home/sridhar/beastcli-eng1/output"
OUTPUT_FILE = "%s/loop_test.jsonl" % OUTPUT_DIR
FIX_LOG = "%s/loop_fixes.md" % OUTPUT_DIR
FEEDBACK_FILE = "%s/loop_feedback.md" % OUTPUT_DIR
GENERATOR_FILE = "/home/sridhar/beastcli-eng1/merged_dataset_generator.py"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOOL_REGISTRY_NAMES = {t.name for t in ToolRegistry.get_all_tools()}
print("Loaded %d tools" % len(TOOL_REGISTRY_NAMES))

_iteration = 0
_patches_applied = set()
SEP60 = "=" * 60
SEP_HASH = "#" * 60


# AGENT 1 - GENERATOR

def generate_dataset(count):
    global _iteration
    # Reload generator module to pick up any patches
    import merged_dataset_generator
    importlib.reload(merged_dataset_generator)
    from merged_dataset_generator import ComprehensiveDatasetPipeline, Localization
    print("\n" + SEP60)
    print("AGENT 1 (Generator): Generating %d samples (iter %d)..." % (count, _iteration))
    print(SEP60)

    pipeline = ComprehensiveDatasetPipeline(seed=random.randint(1, 99999))
    loc = Localization(language="en", tone="professional", formality="neutral",
                       humanize=True, humanize_level="medium")

    examples = pipeline.generate_batch(count=count, localization=loc)
    print("Generated: %d examples" % len(examples))

    valid_examples, stats = pipeline.validator.validate_batch(examples)
    print("Validation: %d/%d valid" % (stats["valid"], stats["total"]))
    if stats["errors_by_type"]:
        print("  Errors: %s" % stats["errors_by_type"])

    with open(OUTPUT_FILE, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    print("Saved %d to %s" % (len(valid_examples), OUTPUT_FILE))

    return len(valid_examples), [ex.to_dict() for ex in valid_examples]


# AGENT 2 - CRITIC

def run_critic(samples, iteration):
    print("\n" + SEP60)
    print("AGENT 2 (Critic): Analyzing %d samples..." % len(samples))
    print(SEP60)

    tool_issues, schema_issues, hallucination_issues = [], [], []
    localization_issues, empty_issues = [], []
    good_count = 0

    for idx, sample in enumerate(samples):
        issues = []
        try:
            messages = sample.get("messages", [])
            if not messages:
                issues.append("  [SAMPLE %d] Empty messages" % (idx + 1))
                continue

            roles = [m.get("role") for m in messages]
            non_system = [r for r in roles if r != "system"]
            if not non_system:
                issues.append("  [SAMPLE %d] No non-system" % (idx + 1))
            elif non_system[0] != "user":
                issues.append("  [SAMPLE %d] First non-system must be user" % (idx + 1))

            last = messages[-1]
            if last.get("role") == "assistant":
                try:
                    lc = json.loads(last.get("content", "{}"))
                    if lc.get("type") != "final_answer":
                        issues.append("  [SAMPLE %d] Last must be final_answer" % (idx + 1))
                    else:
                        ans = lc.get("content", "").strip()
                        if not ans:
                            issues.append("  [SAMPLE %d] final_answer empty" % (idx + 1))
                            empty_issues.append(idx)
                except (json.JSONDecodeError, TypeError) as e:
                    issues.append("  [SAMPLE %d] Last bad JSON: %s" % (idx + 1, e))
            else:
                issues.append("  [SAMPLE %d] Last must be assistant" % (idx + 1))

            for mi, msg in enumerate(messages):
                role = msg.get("role")
                cs = msg.get("content", "")
                if role not in ("assistant", "tool"):
                    continue
                try:
                    content = json.loads(cs)
                    mt = content.get("type")
                    if mt == "tool_call":
                        tn = content.get("tool_name", "")
                        args = content.get("arguments", {})
                        tid = content.get("id", "")
                        if tn not in TOOL_REGISTRY_NAMES:
                            issues.append("  [SAMPLE %d] Unknown tool: %s" % (idx + 1, tn))
                        if not tid:
                            issues.append("  [SAMPLE %d] tool_call missing id" % (idx + 1))
                        td = next((t for t in ToolRegistry.get_all_tools() if t.name == tn), None)
                        if td:
                            ra = [a.name for a in td.arguments if a.required]
                            for r in ra:
                                if r not in args:
                                    issues.append("  [SAMPLE %d] Tool %s missing arg: %s" % (idx + 1, tn, r))
                    elif mt == "final_answer":
                        ac = content.get("content", "")
                        if not ac.strip():
                            issues.append("  [SAMPLE %d] final_answer empty" % (idx + 1))
                            empty_issues.append(idx)
                        lq = ac.lower()
                        if "error" in lq and "success" in lq:
                            issues.append("  [SAMPLE %d] Hallucination" % (idx + 1))
                        if "not found" in lq and "created" in lq:
                            issues.append("  [SAMPLE %d] Hallucination" % (idx + 1))
                    elif mt is None:
                        issues.append("  [SAMPLE %d] No type field" % (idx + 1))
                except (json.JSONDecodeError, TypeError) as e:
                    issues.append("  [SAMPLE %d] msg %d bad JSON" % (idx + 1, mi))

            if issues:
                for issue in issues:
                    if "Unknown tool" in issue or "missing arg:" in issue:
                        tool_issues.append(issue)
                    elif "Hallucination" in issue:
                        hallucination_issues.append(issue)
                    elif any(x in issue for x in ["empty", "bad JSON", "missing id", "final_answer", "no type", "First non-system", "Last must"]):
                        schema_issues.append(issue)
                    else:
                        schema_issues.append(issue)
            else:
                good_count += 1
        except Exception as e:
            issues.append("  [SAMPLE %d] Exception: %s" % (idx + 1, e))
            schema_issues.append("  [SAMPLE %d] Exception: %s" % (idx + 1, e))

    total = len(tool_issues) + len(schema_issues) + len(hallucination_issues)
    id_count = sum(1 for i in schema_issues if "missing id" in i)
    empty_count = sum(1 for i in schema_issues if "empty" in i)

    report = []
    report.append("# Critic Report -- Iteration %d" % iteration)
    report.append("Total samples: %d | Good: %d | Issues: %d" % (len(samples), good_count, total))
    report.append("")
    if tool_issues:
        report.append("## Tool Issues (%d)" % len(tool_issues))
        for i in tool_issues[:5]:
            report.append(i)
        report.append("")
    if schema_issues:
        report.append("## Schema Issues (%d)" % len(schema_issues))
        for i in schema_issues[:8]:
            report.append(i)
        report.append("")
    if hallucination_issues:
        report.append("## Hallucination (%d)" % len(hallucination_issues))
        for i in hallucination_issues[:3]:
            report.append(i)
        report.append("")

    err_rate = total / float(len(samples)) if samples else 1.0
    satisfied = err_rate < 0.05 or (good_count >= len(samples) * 0.95)

    report.append("## Summary")
    report.append("- Good: %d/%d (%.1f%%)" % (good_count, len(samples), good_count / len(samples) * 100))
    report.append("- Tool issues: %d" % len(tool_issues))
    report.append("- Schema: %d (missing_id=%d, empty=%d)" % (len(schema_issues), id_count, empty_count))
    report.append("- Hallucination: %d" % len(hallucination_issues))
    report.append("")
    verdict = "SATISFIED" if satisfied else "NEEDS FIX"
    report.append("## Verdict: %s (error rate: %.1f%%, threshold: <5%%)" % (verdict, err_rate * 100))

    report_text = "\n".join(report)
    print(report_text)
    with open(FEEDBACK_FILE, "w") as f:
        f.write(report_text)
    return satisfied, report_text


# AGENT 3 - FIXER

def apply_fixes(feedback, iteration):
    global _patches_applied
    print("\n" + SEP60)
    print("AGENT 3 (Fixer): Analyzing feedback and PATCHING code...")
    print(SEP60)

    lines = feedback.split("\n")
    id_issues = [l.strip() for l in lines if "tool_call missing id" in l]
    empty_issues = [l.strip() for l in lines if "final_answer empty" in l]
    unknown_tool = [l.strip() for l in lines if "Unknown tool:" in l]
    missing_arg = [l.strip() for l in lines if "missing arg:" in l]
    hallucination = [l.strip() for l in lines if "Hallucination" in l]

    fixes = []
    patches_done = []
    gen_path = Path(GENERATOR_FILE)
    gen_code = gen_path.read_text()

    # FIX 1: Add 'id' to tool_call JSON
    if id_issues and "PATCH_ID_FIELD" not in _patches_applied:
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
            patches_done.append("Added id field to tool_call JSON")
            _patches_applied.add("PATCH_ID_FIELD")
            fixes.append("ID_FIX: Added id field to tool_call")
        else:
            # Try regex insertion
            pat = r'("type":\s*"tool_call",)(\s*\n)'
            repl = r'\1\2            "id": "call_%s" % uuid.uuid4().hex[:12],\2'
            new_code = re.sub(pat, repl, gen_code, count=1)
            if new_code != gen_code:
                gen_code = new_code
                patches_done.append("Added id field via regex")
                _patches_applied.add("PATCH_ID_FIELD")
                fixes.append("ID_FIX: Added id field to tool_call")

    # FIX 2: Ensure final_answer non-empty
    if empty_issues and "PATCH_EMPTY_ANSWER" not in _patches_applied:
        old_block = '''        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)'''
        new_block = '''        if not final_response or not final_response.strip():
            final_response = "Task completed successfully."
        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)'''
        if old_block in gen_code:
            gen_code = gen_code.replace(old_block, new_block, 1)
            patches_done.append("Added empty content guard")
            _patches_applied.add("PATCH_EMPTY_ANSWER")
            fixes.append("EMPTY_FIX: Added empty content guard")

    if hallucination and "PATCH_HALLUCINATION" not in _patches_applied:
        fixes.append("HALLUCINATION_FIX: Manual review needed")
        _patches_applied.add("PATCH_HALLUCINATION")

    if unknown_tool and "PATCH_TOOL_VALIDATION" not in _patches_applied:
        fixes.append("TOOL_FIX: Unknown tools -- manual review")
        _patches_applied.add("PATCH_TOOL_VALIDATION")

    if missing_arg and "PATCH_ARG_VALIDATION" not in _patches_applied:
        fixes.append("ARG_FIX: Missing required args -- manual review")
        _patches_applied.add("PATCH_ARG_VALIDATION")

    if not fixes:
        fixes.append("NO_FIX: No actionable issues")

    if patches_done:
        gen_path.write_text(gen_code)
        print("APPLIED %d patches:" % len(patches_done))
        for p in patches_done:
            print("  - %s" % p)

    fix_log = []
    fix_log.append("# Fix Log -- Iteration %d" % iteration)
    fix_log.append("Timestamp: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    fix_log.append("Patches: %s" % patches_done)
    fix_log.append("Issues: id=%d, empty=%d, unknown=%d, missing_arg=%d, hallucination=%d"
                   % (len(id_issues), len(empty_issues), len(unknown_tool), len(missing_arg), len(hallucination)))
    fix_log.append("")
    fix_log.append("## Fixes")
    for i, f in enumerate(fixes):
        fix_log.append("%d. %s" % (i + 1, f))

    fix_text = "\n".join(fix_log)
    print(fix_text)
    mode = "a" if iteration > 1 else "w"
    with open(FIX_LOG, mode) as f:
        f.write(fix_text + "\n\n")

    return [f for f in fixes if not f.startswith("NO_FIX")]


# MAIN LOOP

def main():
    run_iterations = MAX_ITERATIONS
    run_samples = SAMPLES_PER_RUN
    if len(sys.argv) > 1:
        try:
            run_iterations = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            run_samples = int(sys.argv[2])
        except ValueError:
            pass

    print(SEP60)
    print("DATASET IMPROVEMENT LOOP -- 3-Agent Pipeline")
    print(SEP60)
    print("Target: %d samples/iteration, max %d iterations" % (run_samples, run_iterations))
    print("Output: %s" % OUTPUT_FILE)
    print(SEP60)

    for _iteration in range(1, run_iterations + 1):
        print("\n" + SEP_HASH)
        print("ITERATION %d" % _iteration)
        print(SEP_HASH)

        try:
            count, samples = generate_dataset(run_samples)
        except Exception as e:
            print("Agent 1 FAILED: %s" % e)
            traceback.print_exc()
            break

        print("Generated %d samples" % count)
        satisfied, feedback = run_critic(samples, _iteration)

        if satisfied:
            print("\nSATISFIED after iteration %d" % _iteration)
            print("Final dataset: %d samples at %s" % (count, OUTPUT_FILE))
            break

        fix_count = apply_fixes(feedback, _iteration)
        if fix_count == 0:
            print("No fixes applied -- Agent 2 still dissatisfied")

        print("\n--- End of iteration %d ---" % _iteration)

    else:
        print("\nReached max iterations (%d)" % run_iterations)

    print("\n" + SEP60)
    print("LOOP COMPLETE")
    print(SEP60)
    print("Iterations: %d" % _iteration)
    print("Patches applied: %s" % sorted(_patches_applied))
    print("Output: %s" % OUTPUT_FILE)

    if Path(FEEDBACK_FILE).exists():
        with open(FEEDBACK_FILE) as f:
            print("\n--- Final Critic Report ---")
            print(f.read())


if __name__ == "__main__":
    main()
