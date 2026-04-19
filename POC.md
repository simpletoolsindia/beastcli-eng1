# BeastCLI ENG1 — Senior Engineering Research Paper

## Local-First Agentic CLI with Native Tool Calling, Strict Schema, Dataset Pipeline & Unsloth Training

**Engineer:** 1 (Senior Research & Implementation)
**Date:** 2026-04-19
**Version:** 2.0 (Research-Informed)
**Status:** Phase 0 Complete — Ready for Implementation

---

## Abstract

This paper presents the engineering analysis and implementation plan for **BeastCLI Engineer 1 (ENG1)**: a local-first agentic CLI agent combining native tool execution, strict JSON schema enforcement for tool calls, a self-improving dataset generation pipeline, and Unsloth-compatible training data for fine-tuning. The research draws from Claude Code, Cline, SWE-agent, OpenCodeInterpreter, and Magicoder architectures, and incorporates findings from 15 research domains covering provider comparisons, sandbox security, training strategies, and multi-step reasoning patterns. The primary contribution is a gap analysis between existing beastcli (v1.1.1) infrastructure and the target specification, culminating in a four-phase implementation roadmap of approximately 12 files.

---

## 1. Executive Summary

### 1.1 Current State

beastcli (v1.1.1, npm-published by @simpletoolsindia) provides:
- 40+ native tools across file ops, bash, git, web, MCP
- Multi-provider LLM support (200+ models) via `openaiShim.ts`
- Agent loop via `QueryEngine.ts` (1,309 lines)
- Human-in-the-loop via gRPC + `RemoteTriggerTool`
- Pagination for large outputs (GrepTool, GlobTool)

`code-cli-plan` (TypeScript+Bun+Ink, Phase 1 complete) provides the next-gen CLI foundation.

### 1.2 Critical Gaps

| Gap | Severity | Impact |
|---|---|---|
| Strict JSON tool-call schema (`{"type":"tool_call",...}`) | 🔴 Critical | Provider-agnostic tool format |
| `final_answer` wrapper | 🔴 Critical | Training data format |
| Dataset generation pipeline | 🔴 Critical | Self-improvement |
| Unsloth format export | 🔴 Critical | Training-ready data |
| Max steps enforcement | 🟡 Medium | Loop control |
| Landlock/bubblewrap sandbox | 🟡 Medium | OS-level isolation |

### 1.3 Solution

A 4-phase implementation across ~12 files, ~4,800 lines:

```
Phase 1 (Schema):   schema_normalizer.ts + agent_loop.ts     [~800 lines]
Phase 2 (Pipeline): dataset_export.ts + validate + fix + convert [~1,600 lines]
Phase 3 (Sandbox):  docker_executor + resource_limits + landlock  [~1,000 lines]
Phase 4 (Training): unsloth_train.sh + quality_loop.sh          [~500 lines]
```

---

## 2. Research Survey

### 2.1 Claude Code Architecture

Claude Code is Anthropic's official agentic CLI across Terminal, VS Code, JetBrains, Desktop, and Web surfaces.

**Tool System**: Permission-gated execution with deny→ask→allow precedence. Read-only tools (grep, read) auto-approved. Bash and file edits require approval. Wildcard matching (`Bash(npm run *)`) supported.

**Agent Loop**: Task → analysis → tool execution with approval → error monitoring → self-correction → checkpoint snapshots. Coordinator pattern with subagents (Explore, Plan, general-purpose), each with custom prompts and tool constraints.

**Sandboxing**: macOS uses Seatbelt; Linux/WSL2 uses bubblewrap (`bwrap`). Filesystem: write restricted to cwd + subdirs; read restricted from denied dirs. Network: proxy-based domain allow/deny. Escape hatch via `dangerouslyDisableSandbox` (can be disabled).

**Memory**: Two-tier — CLAUDE.md (user-written instructions, scoped to project/user/org, loaded at session start) + Auto memory (Claude writes to `~/.claude/projects/<project>/memory/`, first 200 lines loaded at each session).

**Skills**: SKILL.md files with YAML frontmatter. `description` for auto-invocation matching. `disable-model-invocation: true` restricts to manual. `allowed-tools` grants scoped permissions. `context: fork` runs in isolated subagent. `!<command>` pre-processes shell output into prompts.

**Hooks**: Fire at lifecycle events (SessionStart, PreToolUse, PostToolUse, etc.). Can run shell commands, HTTP, prompts, or subagents. Exit code 2 blocks; exit 0 allows.

**Multi-Provider**: Anthropic, AWS Bedrock, Microsoft Foundry, Google Vertex AI — same tool system across all.

### 2.2 Cline Architecture

Open-source (MIT) VS Code extension, leading Claude Code competitor.

**Tool System**: File diff views with linter monitoring, terminal via VS Code shell integration, headless browser automation, MCP server creation.

**Agent Loop**: Task + optional images → project analysis → file editing with linter monitoring → terminal execution → checkpoint snapshots.

**Human-in-the-Loop**: Every action requires GUI approval — fundamentally different from Claude Code's sandboxing approach. No escape hatch; user controls every operation.

**Multi-Provider**: Anthropic, OpenAI, Gemini, Ollama, LM Studio.

### 2.3 SWE-agent Architecture

Princeton/Stanford research project. YAML-configurable agent-computer interfaces. Docker-based reproducible evaluation. SWE-bench benchmark (state-of-the-art for open-source). Tool definitions as structured YAML with Bash command templates.

### 2.4 OpenCodeInterpreter

Integrates execution with iterative refinement. Code-Feedback dataset: 68K multi-turn interactions with execution feedback. Progressive improvement: 33B model improves from 79.0 → 88.0 with feedback. **Key insight**: Execution feedback is critical for code generation quality.

### 2.5 Magicoder

OSS-Instruct generates 75K training samples from open-source references. Mitigates LLM-synthesized data bias. Based on CodeLlama (7B) and DeepSeek-Coder (6.7B). **Key insight**: Real-world diverse data outperforms pure synthetic generation.

---

## 3. Provider Architecture

### 3.1 Local Provider Comparison

| Provider | Tool Calling | API | Throughput | Best For |
|---|---|---|---|---|
| **Ollama** | Via LangChain/prompting | OpenAI-compatible | Good (llama.cpp) | General local |
| **LM Studio** | Native + MCP | OpenAI + Anthropic | Excellent (MLX) | GPU workloads |
| **vLLM** | Native xgrammar | OpenAI + Anthropic | SOTA (PagedAttention) | Production |
| **OpenInterpreter** | Function-calling | LiteLLM | Moderate | Code execution |

### 3.2 Model Recommendations

**Qwen3 (Primary)**: Native tool calling via SGLang/vLLM/Ollama. Qwen-Agent for MCP wrappers. Dense + MoE (0.6B–235B). Dual-mode (thinking/non-thinking). 256K context (1M extendable). 100+ languages. Supports SFT/DPO/GRPO. **Recommended for ENG1**.

**Gemma4 (Secondary)**: Text-only (1B–27B) and Multimodal (4B–27B). **No native tool calling** — requires fine-tuning. 99 unused tokens reserved for fine-tuning. Unsloth: 2x faster, 70% less VRAM.

### 3.3 Provider Stack (ENG1)

```
Priority 1: Qwen3 (any >= 7B) — Best tool calling, dual-mode
Priority 2: LM Studio + tool-calling model — Best Anthropic compat
Priority 3: Ollama (qwen2.5-coder) — Easiest local deployment
Priority 4: vLLM (Qwen3 via SGLang) — Production-grade tool calling
Cloud fallback: OpenRouter, NVIDIA NIM, OpenAI
```

---

## 4. Tool System Design

### 4.1 Required Native Tools (REQUIREMENTS.md Section 5)

```
File System (5.1):
  read_file(path; offset?; limit?)       → FileReadTool
  write_file(path; content)              → FileWriteTool
  update_file(path; old_string; new_string) → FileEditTool
  delete_file(path)                      → FileDeleteTool
  list_files(path; limit?; offset?)      → GlobTool (pagination required)
  search_files(path; pattern?; glob?; limit?) → GrepTool

Bash (5.2):
  bash(command; timeout?)               → BashTool

Code Execution (5.3):
  python_exec(file; args?)               → PythonRunnerTool
  node_exec(file; args?)                 → NodeRunnerTool
  java_exec(file)                        → JavaRunnerTool

Git (5.4):
  git_pull(repo?)                        → GitPullTool
  git_push(repo?; branch?)               → GitPushTool
  git_commit(message; files?)            → GitCommitTool
  git_stash()                            → GitStashTool

Web (5.5):
  web_search(query)                      → WebSearchTool
  fetch_url(url)                         → WebFetchTool
```

### 4.2 Pagination Strategy (REQUIREMENTS.md Section 6)

Per REQUIREMENTS.md, large outputs require:
```python
# Request
{"tool_name": "list_files", "arguments": {"path": ".", "limit": 50, "offset": 0}}
# Response
{"files": [...], "total": 100000, "next_offset": 50}
```
GrepTool (`:122`) and GlobTool (`:161`) already implement this in beastcli.

---

## 5. Agent Loop Architecture

### 5.1 REQUIREMENTS.md Loop (Section 3)

```python
for step in range(max_steps):
    response = LLM(messages)
    if response.type == "tool_call":
        result = execute_tool(response.tool_name, response.arguments)
        append_tool_result(tool_call_id=response.id, output=result.output, exit_code=result.exit_code)
    elif response.type == "final_answer":
        break
    elif response.type == "user_feedback":
        stop_current_loop()
        replan()
        continue
```

### 5.2 Schema Normalization Layer (Critical Gap)

beastcli's QueryEngine uses Anthropic tool-use blocks internally. A **SchemaNormalizer** layer is required:

```
Provider Response (Anthropic tool_use)
  → SchemaNormalizer.normalize()
  → {"type": "tool_call", "tool_name": "...", "arguments": {...}}

Tool Execution Result
  → SchemaNormalizer.formatToolResult()
  → {"type": "tool_result", "tool_call_id": "...", "output": "...", "exit_code": 0}

LLM Input (messages)
  → Spec-compliant JSON strings in role:assistant messages
```

---

## 6. Message Schema Design

### 6.1 Standard Schema (REQUIREMENTS.md Sections 4.1–4.4)

```json
// 4.1 User
{"type": "user", "content": "List files in current directory"}

// 4.2 Assistant Tool Call
{"type": "tool_call", "id": "call_123", "tool_name": "bash", "arguments": {"command": "ls -l"}}

// 4.3 Tool Response
{"type": "tool_result", "tool_call_id": "call_123", "output": "file1.txt\nfile2.py", "exit_code": 0}

// 4.4 Final Answer
{"type": "final_answer", "content": "There are 2 files: file1.txt and file2.py"}
```

### 6.2 Unsloth Training Format (Section 9.3)

```jsonl
{"messages": [
  {"role": "system", "content": "You are an agent..."},
  {"role": "user", "content": "Run hello.py"},
  {"role": "assistant", "content": "{\"type\":\"tool_call\",\"tool_name\":\"python_exec\",\"arguments\":{\"file\":\"hello.py\"}}"},
  {"role": "tool", "content": "{\"type\":\"tool_result\",\"tool_call_id\":\"call_abc\",\"output\":\"Hello, World!\\n\",\"exit_code\":0}"},
  {"role": "assistant", "content": "{\"type\":\"final_answer\",\"content\":\"Program executed successfully\"}"}
]}
```

**Critical training signal**: assistant `content` is a JSON *string*, not a parsed object. The model learns to emit JSON as text.

---

## 7. Human-in-the-Loop Patterns

### 7.1 Signal Types

| Signal | Trigger | Effect | Recovery |
|---|---|---|---|
| `user_feedback` | User types during execution | Stop, wait for approval | Replan from checkpoint |
| Tool permission denied | `canUseTool` returns false | Skip tool | Try alternative |
| Hook blocking (exit 2) | PreToolUse hook | Block action | User reviews reason |
| Timeout | Tool exceeds timeout | Kill, report | Retry or skip |
| Abort (Ctrl+C) | User signal | Graceful stop | Save state |
| Max steps reached | Loop counter | Stop loop | User decides |

### 7.2 Permission Modes

| Mode | Read | Write | Bash | Approval |
|---|---|---|---|---|
| `plan` | Yes | No | No | N/A |
| `ask` | Yes | Per-approval | Per-approval | Every action |
| `acceptEdits` | Yes | Auto | Per-approval | File edits auto |
| `auto` | Yes | Yes | Yes | Safety net only |
| `bypassPermissions` | Yes | Yes | Yes | Protected dirs only |

### 7.3 Checkpoint System

Based on Cline's checkpoint snapshots: save agent state at each tool call, enable comparison of before/after states, allow rollback on user request. State includes: messages, iteration count, cwd, modified files.

---

## 8. Dataset Generation Pipeline

### 8.1 Architecture

```
Agent Session Recording (beastcli hooks)
         ↓
JSONL Exporter (eng1.py generate)
         ↓
Validation Engine (eng1.py validate)
  ├── JSON Structure Check
  ├── Message Sequence (system → user → tool_call* → tool_result* → final_answer)
  ├── Tool Name Registry Validation
  ├── Required Arguments Check
  ├── Hallucination Detection
  └── Final Answer Validation
         ↓
Auto-Fix Engine (eng1.py validate --fix)
         ↓
Unsloth Format Converter (eng1.py convert)
         ↓
Training-ready JSONL
```

### 8.2 Validation Rules (REQUIREMENTS.md Section 10)

| Rule | Check | Severity | Fixable |
|---|---|---|---|
| JSON validity | Every line parses | Error | No |
| Message sequence | Valid pattern | Error | Partial |
| Tool name validity | tool_name ∈ TOOL_NAMES | Error | Yes |
| Required arguments | All required present | Warning | Yes |
| Non-empty final_answer | content is non-empty | Error | Yes |
| Hallucination | Contradictory language | Warning | No |
| Tool result consistency | output matches exit_code | Warning | No |

### 8.3 Hallucination Detection

Based on OpenCodeInterpreter's execution feedback research:
1. Contradictory language ("error" + "successfully" in same final_answer)
2. Impossible outputs (file claimed created, output shows failure)
3. Overconfidence (exact file contents stated without reading)
4. Permission hallucinations (claiming root without evidence)
5. Non-sequitur tool results (output doesn't match tool called)

### 8.4 Sample Generation Improvements

Current `eng1.py` generates templates with max 2 steps. Research shows:
- Real agent work: 5–10 tool calls per task
- Include error recovery paths (tool failure → corrective action)
- Use real project paths from known repos
- Include mixed success/failure samples
- Mine Cline/claude-code sessions for real patterns

---

## 9. Training Strategy

### 9.1 Target Models (REQUIREMENTS.md Section 11)

| Model | Size | Tool Calling | Training | Priority |
|---|---|---|---|---|
| Qwen3 | 7B–14B | Native | Unsloth (2x, 70% less VRAM) | **Primary** |
| Gemma4 | 9B–27B | Requires fine-tuning | Unsloth | **Secondary** |
| DeepSeek-Coder | 6.7B–33B | Good | Unsloth | Alternative |
| Distilled reasoning | Varies | Varies | Unsloth | Experimental |

**Constraints**: < 12GB VRAM, must support instruction following + structured output + reasoning.

### 9.2 Training Objectives

| Objective | Priority | Signal | Metric |
|---|---|---|---|
| Tool call prediction | 🔴 Critical | Match `tool_name` + `arguments` | Accuracy > 95% |
| JSON structure | 🔴 Critical | Parseable JSON in assistant content | Validity 100% |
| Multi-step planning | 🔴 Critical | Correct tool chains (2–5 steps) | Chain acc > 80% |
| Error recovery | 🟡 Medium | Tool failure → corrective action | Recovery > 60% |
| Conciseness | 🟡 Medium | Short precise final_answers | Token efficiency |

### 9.3 Unsloth Training Config

```python
training_args = TrainingArguments(
    output_dir="./unsloth_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    max_seq_length=4096,
    fp16=not bf16,
    bf16=bf16,
)
```

### 9.4 Dataset Size Recommendations

Based on Magicoder (75K), OpenCodeInterpreter (68K):

| Scope | Minimum | Recommended |
|---|---|---|
| Tool calling basics | 1,000 | 10,000 |
| Multi-step reasoning | 5,000 | 50,000 |
| Full competency | 20,000 | 100,000 |
| Production quality | 50,000 | 500,000 |

---

## 10. Sandbox Architecture

### 10.1 Threat Model

1. Prompt injection via malicious input
2. Malicious dependencies (npm packages, build scripts)
3. Compromised tools with vulnerabilities
4. Social engineering (tricking into dangerous commands)
5. Data exfiltration (sensitive files → external servers)

### 10.2 Layered Defense

```
Layer 1: Permission Rules    → WHICH tools can be used
Layer 2: Sandbox Isolation  → WHAT bash commands can access
Layer 3: Hooks (PreToolUse) → Custom validation before execution
Layer 4: Hooks (PostToolUse) → Audit and formatting after execution
```

### 10.3 Implementation

| Component | Technology | Platform | Enforcement |
|---|---|---|---|
| Filesystem | bubblewrap (Linux), Seatbelt (macOS) | Linux/macOS | OS-level syscalls |
| Network | Proxy-based domain allow/deny | All | Domain filtering |
| Process | cgroups + namespaces | Linux | Kernel resource limits |
| Escape hatch | `dangerouslyDisableSandbox` param | All | Explicit user approval |

### 10.4 Code Execution Sandboxing

For `python_exec`, `node_exec`, `java_exec`:
- Docker containers with pre-built images
- Resource limits: CPU (1 core), memory (512MB), time (60s), disk (100MB)
- Network: Blocked by default, allowlist for package managers
- Filesystem: Read-only project dir, write to `/tmp` only

---

## 11. JSON Schema Enforcement

### 11.1 The Problem

Local models (Ollama, LM Studio) often produce malformed JSON. Claude Code and Cline handle via instruction prompting + provider-native structured output + JSON repair.

### 11.2 Enforcement Strategies

| Strategy | Provider | Effectiveness | Overhead |
|---|---|---|---|
| Instruction prompting | All | Medium | None |
| Provider structured output | Anthropic, OpenAI | High | Low |
| xgrammar (vLLM) | vLLM | High | Low |
| JSON repair library | All | Medium | Medium |
| Retry with stricter prompt | All | High | High (extra call) |

### 11.3 Recommended Approach

```
1. System prompt: Strict JSON instruction (always first)
2. vLLM: xgrammar native enforcement
3. Ollama/LM Studio: json_repair library post-process
4. Retry: On parse failure, re-prompt "Your last response was not valid JSON"
5. Fallback: 2 retries fail → plain-text mode with disclaimer
```

---

## 12. Evaluation Framework

### 12.1 Metrics

| Metric | Target | Measurement |
|---|---|---|
| Tool call accuracy | > 95% | Correct `tool_name` on first try |
| JSON validity | 100% | All assistant messages parse as JSON |
| Iteration success rate | > 80% | Multi-step tasks completed |
| Task completion rate | > 75% | User task fully accomplished |
| Error recovery success | > 60% | Task completed after tool error |
| Dataset cleanliness | 100% | No validation errors after fix |

### 12.2 Benchmark Suite

- **SWE-bench Lite**: Real GitHub issue fixes
- **HumanEval**: Code generation accuracy
- **Custom ENG1 benchmark**: 200 task samples covering all tool types

---

## 13. Gap Analysis: beastcli vs Requirements

### 13.1 Implemented ✅

| Feature | Status | Evidence |
|---|---|---|
| Native tool system | ✅ Complete | `src/tools/` — 40+ tools |
| Pagination (large outputs) | ✅ Complete | GrepTool:122, GlobTool:161 |
| Human-in-the-loop | ✅ Complete | `grpc/server.ts`, `RemoteTriggerTool` |
| Agent loop (multi-step) | ✅ Complete | `QueryEngine.ts` (1,309 lines) |
| Sandbox execution | ✅ Partial | BashTool with timeout, exit code |
| Multi-provider support | ✅ Complete | `openaiShim.ts` — 200+ models |
| Local providers (Ollama/vLLM) | ✅ Via shim | openaiShim.ts |
| Git tools | ✅ Complete | Via BashTool and native git tools |
| Web search/fetch | ✅ Complete | WebSearchTool, WebFetchTool |

### 13.2 Missing ❌

| Feature | Gap | Severity | Effort |
|---|---|---|---|
| Strict JSON schema (`tool_call` type) | Uses Anthropic tool-use blocks | 🔴 Critical | Medium |
| `final_answer` wrapper | Plain text endings | 🔴 Critical | Low |
| Standard message schema | Internal format | 🔴 Critical | Medium |
| Dataset generation pipeline | Separate repo, XML format | 🔴 Critical | High |
| Auto-improvement loop | Not implemented | 🔴 Critical | Medium |
| Unsloth format export | No session → JSONL | 🔴 Critical | Medium |
| Max steps enforcement | Implicit | 🟡 Medium | Low |
| Landlock/bubblewrap sandbox | BashTool lacks OS-level isolation | 🟡 Medium | High |
| Tool permission granularity | Not per-tool | 🟡 Medium | Medium |
| Checkpoint/restore | Not implemented | 🟡 Medium | High |

### 13.3 Implementation Effort

```
Phase 1 (Schema Enforcement):    ~2 files,  ~800 lines  [2 days]
Phase 2 (Dataset Pipeline):       ~5 files, ~2,000 lines [5 days]
Phase 3 (Sandbox Hardening):      ~3 files, ~1,500 lines [4 days]
Phase 4 (Training Integration):   ~2 files,   ~500 lines [2 days]

Total:                           ~12 files, ~4,800 lines [13 days]
```

---

## 14. Research Findings Validated

### 14.1 Confirmed by Research

1. ✅ **Execution feedback critical**: OpenCodeInterpreter shows 9-point improvement with feedback (79.0 → 88.0). ENG1 must include error recovery samples.

2. ✅ **Real data > pure synthetic**: Magicoder uses open-source references. ENG1 should mine real session data.

3. ✅ **Permission vs sandboxing trade-off**: Claude Code uses sandboxing for autonomy; Cline uses GUI approval for safety. ENG1 should support both modes.

4. ✅ **Local model tool calling varies**: vLLM with xgrammar provides strongest enforcement; Ollama needs instruction prompting fallback.

5. ✅ **Checkpoint snapshots enable safe exploration**: Cline's approach. ENG1 should implement this.

6. ✅ **Auto memory reduces repeated mistakes**: Claude Code's proven pattern. ENG1 should implement similar.

7. ✅ **MCP schema aligns with REQUIREMENTS.md**: MCP's JSON Schema-based input validation matches the tool schema design.

8. ✅ **Unsloth enables consumer-grade training**: 2x faster, 70% less VRAM makes fine-tuning accessible.

### 14.2 Empirical Hypotheses (Require Testing)

1. Whether strict JSON schema enforcement reduces model confidence
2. Optimal max_steps threshold for different task types
3. Dataset size vs tool call accuracy (diminishing returns point)
4. Gemma4 fine-tuning vs Qwen3 fine-tuning for tool calling

---

## 15. Implementation Roadmap

### Phase 1: Schema Enforcement (2 files, ~800 lines)

**Files**:
1. `src/core/schema_normalizer.ts` (~300 lines)
   - `normalizeToolCall(provider_response)` → `{"type":"tool_call",...}`
   - `normalizeToolResult(execution_result)` → `{"type":"tool_result",...}`
   - `wrapFinalAnswer(assistant_text)` → `{"type":"final_answer",...}`
   - Provider adapters (Anthropic, OpenAI, Ollama)

2. `src/core/agent_loop.ts` (~500 lines)
   - Max steps enforcement
   - User feedback interrupt handling
   - Checkpoint save/restore
   - Tool call → tool result → LLM loop

### Phase 2: Dataset Pipeline (5 files, ~1,600 lines)

1. `scripts/dataset_export.ts` (~400 lines) — Session recorder via hooks
2. `scripts/validate_dataset.ts` (~300 lines) — Validation engine
3. `scripts/auto_fix.ts` (~200 lines) — Auto-fix engine
4. `scripts/unsloth_converter.ts` (~200 lines) — Format converter
5. `scripts/quality_scorer.ts` (~200 lines) — Sample quality scoring

Integration: `beastcli dataset` CLI command.

### Phase 3: Sandbox Hardening (3 files, ~1,500 lines)

1. `src/sandbox/docker_executor.ts` (~400 lines) — Docker container management
2. `src/sandbox/resource_limits.ts` (~300 lines) — CPU, memory, time limits
3. `src/sandbox/landlock_enforcer.ts` (~300 lines) — Landlock syscall filtering

### Phase 4: Training Integration (2 files, ~500 lines)

1. `scripts/unsloth_train.sh` (~200 lines) — Training script
2. `scripts/dataset_quality_loop.sh` (~300 lines) — Auto-improvement loop

---

## 16. Deep Research Findings

### 16.1 Claude Code Architecture

- **Tool System**: Native file editing, bash, git; MCP extensibility; multi-provider
- **Sandboxing**: bubblewrap (Linux), Seatbelt (macOS); filesystem + network isolation
- **Memory**: CLAUDE.md (user-written, scoped) + Auto memory (Claude-written)
- **Skills**: SKILL.md with frontmatter; `context: fork` for subagent execution
- **Hooks**: PreToolUse, PostToolUse, SessionStart, etc.; shell/HTTP/prompt/agent runners
- **Escape hatch**: `dangerouslyDisableSandbox` on failure (disablable)

### 16.2 Cline Architecture

- **MIT licensed**, open source
- GUI approval for every action (not sandboxing)
- Checkpoint snapshots for diff comparison and state restoration
- MCP server creation for custom tools
- Headless browser automation

### 16.3 Local Provider Comparison

| Provider | Tool Calling | API | Throughput | Best For |
|---|---|---|---|---|
| Ollama | Via prompting | OpenAI-compatible | Good | General local |
| LM Studio | Native + MCP | OpenAI + Anthropic | Excellent | GPU workloads |
| vLLM | Native xgrammar | OpenAI + Anthropic | SOTA | Production |
| OpenInterpreter | Function-calling | LiteLLM | Moderate | Code execution |

### 16.4 MCP Architecture

JSON-RPC 2.0-based; STDIO for local (optimal), HTTP/SSE for remote. Primitives: Tools (executable functions), Resources (data sources), Prompts (templates). JSON Schema-based input validation aligns with REQUIREMENTS.md tool schema.

### 16.5 Agent Training Datasets

- **SWE-agent**: YAML-configurable ACIs; SWE-bench benchmark
- **Magicoder**: 75K from open-source references; mitigates LLM-synthesized bias
- **OpenCodeInterpreter**: 68K with execution feedback; 9-point improvement

### 16.6 Unsloth Fine-tuning

500+ models; 2x faster, 70% less VRAM; RL (GRPO), full fine-tuning, 4-bit/16-bit/FP8; Data Recipes from PDF/CSV/DOCX. Qwen3 + Gemma4 recommended.

### 16.7 Sandbox Security

| Approach | Pros | Cons | Use Case |
|---|---|---|---|
| gVisor (`runsc`) | Strong, rootless | Complex | Production |
| Docker | Familiar | Slow cold starts | Containerized |
| bubblewrap | Fast, Linux-native | Linux only | Local |
| Landlock | Kernel-level | New, limited tooling | Syscall restrictions |

### 16.8 JSON Schema Enforcement

- vLLM: xgrammar native enforcement
- Ollama/LM Studio: JSON repair + retry
- Best: System prompt + instruction prompting + json_repair fallback

### 16.9 Multi-Step Agent Reasoning

- ReAct pattern (Thought/Action/Observation)
- Checkpoint snapshots for comparison
- Execution feedback for iterative improvement
- Max steps enforcement

### 16.10 Provider Summary

| Provider | Local | Tool Calling | Speed | Best For |
|---|---|---|---|---|
| Ollama | Yes | Via prompting | Good | Default local |
| LM Studio | Yes | Native + MCP | Excellent | GPU workloads |
| vLLM | Yes | Native xgrammar | SOTA | Production |
| OpenRouter | Cloud | Best-in-class | Good | Broad access |
| NVIDIA NIM | Cloud | Good | Good | Cloud + local feel |
| OpenAI | Cloud | Best | Excellent | GPT-4o tool calling |

### 16.11 Key Research Sources

| Topic | Source |
|---|---|
| Claude Code | code.claude.com/docs |
| Cline | github.com/cline/cline |
| MCP | modelcontextprotocol.io |
| vLLM | github.com/vllm-project/vllm |
| Ollama | github.com/ollama/ollama |
| LM Studio | lmstudio.ai/docs |
| Qwen3 | qwenlm.github.io/blog/qwen3 |
| SWE-agent | github.com/princeton-nlp/SWE-agent |
| Unsloth | github.com/unslothai/unsloth |
| OpenInterpreter | github.com/OpenInterpreter/open-interpreter |
| Magicoder | github.com/ise-uiuc/Magicoder |
| Continue | github.com/continuedev/continue |

---

## 17. Conclusion

beastcli provides a strong foundation — 40+ native tools, proven agent loop, multi-provider support, and human-in-the-loop hooks. The critical missing pieces are:

1. **Schema normalization** — enforcing `tool_call`/`tool_result`/`final_answer` JSON schema regardless of provider
2. **Dataset pipeline** — converting sessions to Unsloth-ready JSONL with validation and auto-fix
3. **Strict JSON enforcement** — ensuring the model always outputs valid JSON

The implementation is feasible as 4 phases totaling ~12 new/modified files. The existing architecture requires modification but not replacement.

**Research validates the approach**: Execution feedback (OpenCodeInterpreter), real-world data (Magicoder), dual-mode thinking (Qwen3), and consumer-grade fine-tuning (Unsloth) all confirm that the specified system is buildable with current technology. The primary risk is local model JSON output quality — mitigated by vLLM's xgrammar, JSON repair libraries, and retry logic.

---

*Senior Research Paper — BeastCLI ENG1 — Engineer 1 — 2026-04-19*
*Research: Claude Code docs, Cline, SWE-agent, OpenCodeInterpreter, Magicoder, Unsloth, vLLM, Ollama, LM Studio, Qwen3, Gemma4, MCP, bubblewrap, SWE-bench*
