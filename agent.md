# Beast CLI Agentic Dataset Improvement Loop

## Overview

A 3-agent iterative dataset quality improvement system for agentic coding training data.

- **Location**: `/home/sridhar/beastcli-eng1/`
- **Main loop**: `loop_v2.py`, `loop_orchestrator.py`
- **Review module**: `review_update.py`
- **Production cleaner**: `production_clean.py`
- **Generator**: `merged_dataset_generator.py`
- **Data output**: `output/dataset_clean.jsonl`, `output/dataset_100.jsonl`

## Architecture

```
Agent 1 (Generator) ──generate N──▶ output/review_data.jsonl
                                             │
                                             ▼
                                     Agent 2 (Critic)
                                     review_update.py
                                        reviews each row
                                   (10 structural checks)
                                             │
                                   issues ──▶ output/review_comments.jsonl
                                             │
                                             ▼
                                     Agent 3 (Fixer)
                                  reads comments, patches
                                  merged_dataset_generator.py
                                             │
                                             ▼
                                         Loop back
                                        (up to N iterations)
```

## Run Commands

```bash
# Production cleaner (28 checks, runs until 100% clean)
python production_clean.py

# v2 loop: N samples, max iterations
python loop_v2.py 100 10

# Standalone review (Agent 2 only)
python review_update.py output/review_data.jsonl

# Orchestrator
python loop_orchestrator.py 50 10
```

## Production Cleaner — 28 Checks

The `production_clean.py` is the authoritative cleaning tool. It runs these checks per row:

| Category | # | Check |
|---|---|---|
| **SCHEMA** | 12 | message JSON valid, role valid, type field present, tool_call fields (id, tool_name, arguments), tool_result fields (tool_call_id), final_answer fields (content) |
| **TOOL_CALL** | 3 | tool in registry, required args present, id non-empty |
| **TOOL_RESULT** | 4 | tool_call_id matches a tool_call, output valid JSON, no unsubstituted placeholders, exit_code matches output |
| **FINAL_ANSWER** | 5 | non-empty, no vague phrases, no placeholders, no contradictions, grounded in tool results |
| **SEQUENCE** | 4 | first non-system is user, last is assistant, tool_call count == tool_result count, correct alternation |

## Critical Architecture Rules

**1. INTENT_TEMPLATES** (`ToolCallGenerator.INTENT_TEMPLATES`)
- Must have an entry for EVERY tool (all 27 tools)
- Each entry has `query_hint` and `args` (key is `"args"`, NOT `"arguments"`)
- Arg names must match `ToolSchema.arguments` exactly (e.g., `file_path`, `source`, `path`)
- If a tool is missing, `_fallback_value()` is used, which may produce `None` for required string args

**2. SUCCESS_TEMPLATES** (`ResponseGenerator.SUCCESS_TEMPLATES`)
- Must have an entry for EVERY tool
- Placeholders in templates (e.g., `{path}`, `{count}`) must match the actual arg names
- The `_generate_success()` method has a post-substitution guard that replaces any remaining `{placeholder}` with arg values or defaults
- All tools return JSON output (no raw strings)

**3. FINAL_TEMPLATES** (`_build_final_answer`)
- Every tool MUST have a template
- Every template MUST start with the tool name for groundedness (e.g., `"File_Search: Found 3 matching files"`)
- All placeholders in templates MUST have substitution handlers in `_build_final_answer()`
- The error path also uses `{tool_name}: Operation failed...` to stay grounded

**4. Tool call ID chain**
- `system_call_id = "call_%s" % uuid.uuid4().hex[:12]` — generated ONCE
- Used in both `tool_call` content AND `tool_result` — same variable, no duplicate generation

## Issue Codes

| Code | Category | Meaning |
|------|----------|---------|
| `TC_MISSING_ID` | SCHEMA | tool_call missing id field |
| `TC_MISSING_TOOL` | SCHEMA | tool_call missing tool_name |
| `TC_BAD_ARGS` | SCHEMA | arguments missing or not dict |
| `TR_MISSING_TCID` | SCHEMA | tool_result missing tool_call_id |
| `FA_MISSING_CONTENT` | SCHEMA | final_answer missing content |
| `UNKNOWN_TOOL` | TOOL_CALL | tool not in registry |
| `MISSING_ARG` | TOOL_CALL | required arg not provided |
| `EMPTY_ID` | TOOL_CALL | id field is empty |
| `UNSUBSTITUTED_ID` | TOOL_CALL | id is `{{TOOL_CALL_ID}}` |
| `UNSUBSTITUTED_TCID` | TOOL_RESULT | tool_call_id is `{{TOOL_CALL_ID}}` |
| `TCID_MISMATCH` | TOOL_RESULT | tool_call_id not in any tool_call |
| `OUTPUT_NOT_JSON` | TOOL_RESULT | output not valid JSON |
| `OUTPUT_PLACEHOLDER` | TOOL_RESULT | output has unsubstituted placeholder |
| `EXIT_CODE_CONFLICT` | TOOL_RESULT | exit_code=0 but output says "error" |
| `VAGUE_ANSWER` | FINAL_ANSWER | too generic |
| `ANSWER_PLACEHOLDER` | FINAL_ANSWER | unsubstituted placeholder in answer |
| `HALLUCINATION` | FINAL_ANSWER | contradictory terms |
| `NOT_GROUNDED` | FINAL_ANSWER | answer doesn't reference tool name |
| `BAD_FIRST_ROLE` | SEQUENCE | first non-system must be user |
| `BAD_LAST_ROLE` | SEQUENCE | last must be assistant |
| `COUNT_MISMATCH` | SEQUENCE | tool_call count != tool_result count |
| `BAD_ALTERNATION` | SEQUENCE | consecutive same-role non-system messages |

## Common Fix Patterns

| Issue | Fix Location |
|-------|-------------|
| tool_call missing id | Add `"id": system_call_id` to tool_call content dict |
| `{file_path}` unsubstituted in tool result | Change template to use correct arg name `{path}` or add guard in `_generate_success()` |
| Raw string tool result | Add entry to `SUCCESS_TEMPLATES` with JSON template |
| Vague final answer | Add tool-specific template to `FINAL_TEMPLATES` starting with tool name |
| Final answer not grounded | Prepend `{tool_name}:` to template |
| Placeholder in answer | Add `{placeholder}` handler to `_build_final_answer()` substitution section |
| Tool missing required arg | Add entry to `INTENT_TEMPLATES` with full args |
| tool_result id mismatch | Remove duplicate UUID generation — reuse same `system_call_id` |

## Adding a New Tool

1. Add to `ToolRegistry` as a `ToolSchema` with correct arg names and required flags
2. Add to `SUCCESS_TEMPLATES` with JSON template using correct arg placeholder names
3. Add to `FINAL_TEMPLATES` with tool-name-prefixed template
4. Add to `INTENT_TEMPLATES` with realistic query_hint and all required args
5. Add to `TOOL_REQUIRED_ARGS` in `review_update.py` if special logic checks needed
6. Run `production_clean.py` to verify
