# BeastCLI Engineer 1 — POC.md

**Engineer:** 1 (Senior Research & Implementation)
**Date:** 2026-04-19
**Status:** Phase 0 Complete

---

## 1. Executive Summary

This document is the engineering analysis and implementation plan for **BeastCLI Engineer 1 (ENG1)**: a local-first agentic CLI with native tool calling, strict JSON schema enforcement, dataset generation pipeline, and Unsloth training support.

**Existing state:** `beastcli` (v1.1.1, npm-published by @simpletoolsindia) provides a mature CLI with 40+ native tools, multi-provider LLM support (200+ models), TUI, and human-in-the-loop via gRPC. `code-cli-plan` (TypeScript+Bun+Ink) has Phase 1 complete.

**Target:** REQUIREMENTS.md specifies a strict JSON tool-call schema, standardized message format, dataset generation pipeline, auto-improvement loop, and Unsloth training support — aligned with how Claude Code / Cline / Cursor operate internally.

---

## 2. Architecture Gap Analysis

### 2.1 What's Implemented (beastcli v1.1.1)

#### ✅ Native Tool System
- **Location:** `src/tools/` — 40+ tools
- **Coverage:** BashTool, FileReadTool, FileEditTool, FileWriteTool, GlobTool, GrepTool, WebSearchTool, WebFetchTool, LSPTool, MCPTool, Git tools, Task tools, Cron tools, Voice tools
- **Pattern:** Each tool is a class extending a base `Tool` interface, with `execute()`, `description`, `schema()` methods
- **Pagination:** GrepTool and GlobTool support `limit`/`offset` for large outputs (GrepTool:122, GlobTool:161)
- **Sandbox:** BashTool supports timeout, streaming output, exit code capture

#### ✅ Multi-Provider LLM Support
- **Location:** `src/services/api/`
- **Coverage:** OpenAI, Claude (Anthropic), Azure OpenAI, OpenRouter, NVIDIA NIM, Ollama, LM Studio, vLLM — all via OpenAI-compatible shim
- **Schema:** Uses Anthropic `ToolUseBlockParam` / `ToolResultBlockParam` for Claude providers; OpenAI tool format for others
- **Providers:** 200+ models documented in `providerConfig.ts`

#### ✅ Agent Loop
- **Location:** `src/QueryEngine.ts` (1,309 lines)
- **Pattern:** `for-await` loop over streaming messages; accumulates `toolUseResult`; tracks max iterations
- **Message flow:** user → LLM → tool_use → tool_result → LLM → ... → end_turn
- **Human-in-the-loop:** `grpc/server.ts` has `canUseTool` hook; `RemoteTriggerTool` for interrupting

#### ✅ Human-in-the-Loop
- **gRPC server:** `src/grpc/server.ts` — permission hooks, tool-use acknowledgment
- **RemoteTriggerTool:** Allows external interruption of agent loop
- **Abort controller:** Per-session `AbortController` propagated through tool execution

### 2.2 What's Missing

#### 🔴 Critical: Strict JSON Schema Enforcement

beastcli uses Anthropic tool-use blocks internally — structured objects but **not** the flat `{"type":"tool_call","tool_name":"...","arguments":{...}}` schema from REQUIREMENTS.md.

```
REQUIREMENTS.md expects:
{
  "type": "tool_call",
  "id": "call_123",
  "tool_name": "bash",
  "arguments": {"command": "ls -la"}
}

beastcli produces (Claude provider):
{
  "type": "tool_use",
  "id": "toolu_xxx",
  "name": "bash",
  "input": {"command": "ls -la"}
}
```

**Gap:** The spec mandates a custom JSON schema that is provider-agnostic. The agent loop must be modified to:
1. Normalize all provider outputs to the spec schema
2. Inject tool results back in spec format
3. Enforce JSON-only output from the model

#### 🔴 Critical: `final_answer` Wrapper

beastcli ends with plain assistant text. REQUIREMENTS.md demands:
```
{
  "type": "final_answer",
  "content": "There are 2 files in the directory."
}
```

**Gap:** Post-processing needed to detect when the agent is done and wrap output in `final_answer` format.

#### 🔴 Critical: Dataset Generation Pipeline

The `agentic-coding-dataset/` repo (Python) generates Cline-style XML, not the JSONL format required here. beastcli has no built-in session → dataset export.

**Gap:** Need a pipeline that:
1. Records agent sessions with tool calls + results
2. Exports to JSONL (section 9.3 format)
3. Validates JSON structure, tool validity, hallucination
4. Auto-fixes issues and re-validates

#### 🟡 Medium: Standard Message Schema

REQUIREMENTS.md sections 4.1–4.4 define specific schemas. beastcli uses internal message types. A translation/mapping layer is needed.

#### 🟡 Medium: Auto-Improvement Loop (Section 10)

The spec calls for:
```
1. Generate dataset
2. Run validation
3. Detect issues: invalid JSON, wrong tool usage, hallucination
4. Fix dataset
5. Repeat until clean
```

This is not implemented anywhere in the current codebase.

---

## 3. Detailed Gap: Tool Call Schema

### 3.1 Current Implementation

beastcli's tool system uses `src/Tool.ts` (802 lines) as the base class. Each tool returns a result via `ToolResultBlockParam`:

```typescript
// From Tool.ts imports
import type {
  ToolResultBlockParam,
  ToolUseBlockParam,
} from '@anthropic-ai/sdk/resources/index.mjs'
```

The LLM receives tools as `ToolUseBlockParam[]` — provider-specific schema.

### 3.2 Required Schema (REQUIREMENTS.md)

```typescript
// Target: provider-agnostic JSON schema
interface ToolCallMessage {
  type: "tool_call"
  id: string           // Unique call ID
  tool_name: string     // From TOOL_NAMES registry
  arguments: Record<string, unknown>
}

interface ToolResultMessage {
  type: "tool_result"
  tool_call_id: string // Must match call ID
  output: string       // Raw output string
  exit_code: number    // 0 = success, non-zero = error
  error?: string        // Error message if exit_code != 0
}

interface FinalAnswerMessage {
  type: "final_answer"
  content: string      // Human-readable answer
}
```

### 3.3 Conversion Layer

Need a `SchemaNormalizer` class that:
1. Takes Anthropic `ToolUseBlock` → converts to `ToolCall`
2. Takes provider response → extracts tool_call info
3. Takes tool execution result → formats as `ToolResult`
4. Wraps final response in `FinalAnswer`

---

## 4. Detailed Gap: Dataset Format

### 4.1 Current State

beastcli has no dataset export. The `agentic-coding-dataset/` Python repo generates Cline-style XML:
```xml
<agentinstruction>...</agentinstruction>
<tools>...</tools>
<action>...</action>
<result>...</result>
```

### 4.2 Required Format (Unsloth, REQUIREMENTS.md section 9)

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are an agent..."},
    {"role": "user", "content": "Run hello.py"},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"tool_name\":\"python_exec\",\"arguments\":{\"file\":\"hello.py\"}}"},
    {"role": "tool", "content": "{\"type\":\"tool_result\",\"tool_call_id\":\"call_abc\",\"output\":\"Hello, World!\\n\",\"exit_code\":0}"},
    {"role": "assistant", "content": "{\"type\":\"final_answer\",\"content\":\"Program executed successfully\"}"}
  ]
}
```

### 4.3 Key Differences from Cline XML Format

| Aspect | Cline XML | REQUIREMENTS.md JSONL |
|--------|-----------|----------------------|
| Format | XML tags | JSON objects |
| Roles | Implicit | Explicit `role` field |
| Tool call | `<tools>` tag | `{"type":"tool_call",...}` |
| Tool result | `<result>` tag | `{"type":"tool_result",...}` |
| Final answer | Implicit | `{"type":"final_answer",...}` |
| Training target | Full XML | `assistant` messages only |

---

## 5. Provider Analysis

### 5.1 Local Provider Support (Critical for "local-first")

The spec mandates **local-first**: Ollama, vLLM, llama.cpp, LM Studio.

**beastcli status:** Full support via `openaiShim.ts`:
- Ollama: OpenAI-compatible `/api/chat/completions` endpoint
- LM Studio: OpenAI-compatible endpoint
- vLLM: OpenAI-compatible endpoint

**Gap:** None for inference. The gap is in **tool calling** — Ollama and other local models often have weak or inconsistent tool-calling support. Need to handle:
- Models that don't support tool calling → use instruction prompting
- Models with different tool-call output formats → normalization layer
- Rate limits, connection issues with local endpoints

### 5.2 Recommended Provider Stack

| Priority | Provider | Use Case |
|----------|----------|----------|
| 1 | Ollama (local) | Default local-first |
| 2 | LM Studio (local) | GPU-accelerated local |
| 3 | NVIDIA NIM | Cloud with good tool support |
| 4 | OpenRouter | broadest model selection |
| 5 | OpenAI | GPT-4o for best tool calling |

---

## 6. Tool System Design

### 6.1 Required Native Tools (from spec section 5)

All must be implemented as native functions, NOT via MCP:

#### File System (5.1)
- `read_file(path, offset?, limit?)` — Read file with pagination
- `write_file(path, content)` — Write/create file
- `update_file(path, old_string, new_string)` — String-based edit
- `delete_file(path)` — Delete file
- `list_files(path, limit?, offset?)` — Directory listing with pagination
- `search_files(path, pattern?, glob?, limit?)` — Content/pattern search

#### Bash (5.2)
- `bash(command, timeout?)` — Shell execution with sandbox

#### Code Execution (5.3)
- `python_exec(file, args?)` — Python runner
- `node_exec(file, args?)` — Node.js runner
- `java_exec(file)` — Java runner

#### Git (5.4)
- `git_pull(repo?)` — Git pull
- `git_push(repo?, branch?)` — Git push
- `git_commit(message, files?)` — Git commit
- `git_stash()` — Git stash

#### Web (5.5)
- `web_search(query)` — Web search
- `fetch_url(url)` — Fetch URL as markdown/text

### 6.2 Pagination Strategy (Section 6)

Large directories (100K+ files) require pagination:
```python
# Tool request
{"tool_name": "list_files", "arguments": {"path": ".", "limit": 50, "offset": 0}}

# Tool response includes
{"files": [...], "total": 100000, "next_offset": 50}
```

beastcli already implements this in GrepTool and GlobTool — needs extension to all list/search tools.

---

## 7. Agent Loop Design

### 7.1 Loop Pseudocode (from spec section 3)

```python
for step in range(max_steps):
    response = LLM(messages)

    if response.type == "tool_call":
        result = execute_tool(response.tool_name, response.arguments)
        append_tool_result(
            tool_call_id=response.id,
            output=result.output,
            exit_code=result.exit_code
        )

    elif response.type == "final_answer":
        break

    elif response.type == "user_feedback":
        # Human-in-the-loop interrupt
        stop_current_loop()
        replan()
        continue
```

### 7.2 Max Steps Enforcement

- Default: 10–20 iterations
- Configurable per-task
- Must stop ONLY on `final_answer` or `max_steps` reached
- Track iteration count and surface to user

### 7.3 Tool Result Accumulation

The LLM must see the **full history**:
- All previous tool calls
- All previous tool results
- Full conversation context

This is already implemented in `QueryEngine.ts` via `mutableMessages` accumulation.

---

## 8. Human-in-the-Loop (Section 8)

### 8.1 Interrupt Types

| Signal | Trigger | Action |
|--------|---------|--------|
| `user_feedback` | User types during execution | Stop loop, replan, continue |
| Tool permission denied | `canUseTool` returns false | Skip tool, continue |
| Timeout | Command exceeds timeout | Kill, report error |
| Abort | User presses Ctrl+C | Graceful stop |

### 8.2 Replan Logic

On interrupt:
1. Save current state (messages, iteration count, tool results)
2. Present user's correction to LLM
3. Continue from interrupted point
4. Never lose context

---

## 9. Dataset Generation Pipeline

### 9.1 Architecture

```
Agent Session Recording
         ↓
JSONL Exporter (eng1.py generate)
         ↓
Validation Engine (eng1.py validate)
  ├── JSON Structure Check
  ├── Tool Call Validation (TOOL_NAMES registry)
  ├── Sequence Validation (user → assistant → tool → ...)
  ├── Hallucination Detection
  └── Missing Field Detection
         ↓
Auto-Fix Engine (eng1.py validate --fix)
  ├── Add missing IDs
  ├── Fix unknown tool names
  ├── Fill missing arguments
  └── Remove empty final_answers
         ↓
Unsloth Format Converter (eng1.py convert)
         ↓
Training-ready JSONL
```

### 9.2 eng1.py Implementation

The `eng1.py` script provides:

| Command | Function |
|---------|----------|
| `generate` | Create N samples with realistic tool chains |
| `validate` | Run all checks, report issues |
| `validate --fix` | Auto-fix and save `_fixed.jsonl` |
| `stats` | Show distribution, tool usage, token estimates |
| `replay` | Pretty-print random samples |
| `convert` | Ensure Unsloth compatibility |

### 9.3 Hallucination Detection

Patterns to detect:
- Contradictory success/error language in `final_answer`
- Tool results that contradict the tool call
- Non-existent file paths claimed as "created"
- `Permission denied` for commands that shouldn't need it

---

## 10. Training Strategy (Unsloth, Section 11)

### 10.1 Model Recommendations

Per REQUIREMENTS.md section 11:
- **Qwen latest** (>= 3.6 family) — strong tool calling, multilingual
- **Gemma latest** (>= 4 family) — efficient, good reasoning
- **Distilled reasoning models** — smaller, faster inference

**Constraints:**
- Must be < 12GB VRAM
- Must support instruction following
- Must support structured output
- Must support reasoning

### 10.2 Training Objectives

| Objective | Priority | Description |
|-----------|----------|-------------|
| Tool call prediction | 🔴 Critical | Correct `tool_name` + `arguments` |
| JSON structure enforcement | 🔴 Critical | Valid JSON output always |
| Multi-step planning | 🔴 Critical | Chain 2-3 tools correctly |
| Error recovery | 🟡 Medium | Recover from tool failures |
| Conciseness | 🟡 Medium | Not over-explaining |

### 10.3 Unsloth Training Config

```python
from unsloth import UnslothTrainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./unsloth_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=500,
    fp16=not bf16,
    bf16=bf16,
    max_seq_length=4096,
    dataloader_num_workers=4,
)
```

---

## 11. Sandbox Design (Section 13)

### 11.1 Requirements

- Isolated execution environment
- Resource limits (CPU, memory, time)
- Timeout handling
- Safe execution for Python, Node, Java, Bash

### 11.2 Implementation Options

| Approach | Pros | Cons |
|----------|------|------|
| `runsc` (gVisor) | Strong isolation, rootless | Complex setup |
| Docker containers | Familiar, good isolation | Slow cold starts |
| `bwrap` (bubblewrap) | Fast, Linux-native | Linux only |
| Prisoners/Mulval | Proven in Claude Code | Harder to implement |
| Landlock (Linux) | Kernel-level, fast | New, limited tooling |

**Recommendation:** Docker-based sandbox with pre-built images for Python/Node/Java, using Landlock as a secondary layer for syscall restrictions.

---

## 12. Evaluation Metrics (Section 14)

| Metric | Target | Measurement |
|--------|--------|------------|
| Tool call accuracy | > 95% | Correct tool_name on first try |
| JSON validity | 100% | All assistant messages parse as JSON |
| Iteration success rate | > 80% | Multi-step tasks completed |
| Task completion rate | > 75% | User task fully accomplished |
| Error recovery success | > 60% | Task completed after tool error |

---

## 13. Implementation Phases

### Phase 1: Schema Enforcement (2 files)
- [ ] `src/core/schema_normalizer.ts` — Convert provider outputs ↔ spec schema
- [ ] `src/core/final_answer_wrapper.ts` — Wrap responses in `final_answer`
- [ ] Update `QueryEngine.ts` to use normalized schema
- [ ] Add `max_steps` enforcement with clear config

### Phase 2: Dataset Pipeline (5 files)
- [ ] `scripts/dataset_generator.ts` — Session → JSONL export
- [ ] `scripts/validate_dataset.ts` — Validation engine
- [ ] `scripts/auto_fix.ts` — Auto-fix engine
- [ ] `scripts/unsloth_converter.ts` — Format converter
- [ ] Integrate into beastcli as `beastcli dataset` command

### Phase 3: Sandbox Hardening (3 files)
- [ ] Docker-based execution for `python_exec`, `node_exec`, `java_exec`
- [ ] Resource limits (CPU, memory, disk)
- [ ] Timeout enforcement per tool

### Phase 4: Training Integration (2 files)
- [ ] Unsloth training config for Qwen 3.6
- [ ] Dataset quality auto-scorer

---

## 14. Key Files Reference

### beastcli (existing, npm-published)
```
src/
├── Tool.ts                   # Base tool class (802 lines)
├── QueryEngine.ts            # Agent loop (1,309 lines)
├── grpc/server.ts           # Human-in-the-loop hooks
├── services/api/
│   ├── openaiShim.ts         # Multi-provider compatibility
│   ├── providerConfig.ts     # 200+ model configs
│   └── toolArgumentNormalization.ts
├── tools/
│   ├── BashTool/
│   ├── FileReadTool/
│   ├── FileEditTool/
│   ├── FileWriteTool/
│   ├── GlobTool/             # Has pagination (limit/offset)
│   ├── GrepTool/             # Has pagination (limit/offset)
│   ├── WebSearchTool/
│   ├── WebFetchTool/
│   └── ...
└── coordinator/
    └── coordinatorMode.ts
```

### code-cli-plan (TypeScript+Bun+Ink, Phase 1 done)
```
src/
├── engine/index.ts           # Agent loop
├── modes/index.ts            # Permission modes
├── tui/                      # 8 TUI components
├── llm/                      # Provider abstraction
└── tools/                    # Tool system
```

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Local model tool calling fails | High | High | Fallback to instruction prompting; normalize varied output formats |
| JSON output from LLM is malformed | High | Critical | Post-process with JSON repair; add retry with stricter prompt |
| Dataset hallucination contaminates training | Medium | High | Multi-layer validation; human review for auto-fixed samples |
| Sandbox escape | Low | Critical | Docker + Landlock + seccomp; never run as root |
| Tool permission explosion (user fatigue) | Medium | Medium | Smart defaults: auto-allow safe tools, prompt only for risky ones |

---

## 16. Conclusion

beastcli provides a strong foundation — 40+ native tools, proven agent loop, multi-provider support, and human-in-the-loop hooks. The **critical missing pieces** are:

1. **Schema normalization** — enforcing the spec's `tool_call`/`tool_result`/`final_answer` JSON schema regardless of provider
2. **Dataset pipeline** — converting sessions to Unsloth-ready JSONL with validation and auto-fix
3. **Strict JSON enforcement** — ensuring the model always outputs valid JSON, never plain text

The implementation is feasible as 4 phases totaling ~12 new/modified files. The existing architecture (QueryEngine, Tool system, openaiShim) requires modification but not replacement.

---

*Generated by Engineer 1 for BeastCLI ENG1 POC — 2026-04-19*
