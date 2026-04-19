# BeastCLI ENG1 — Local-First Agentic CLI Dataset Pipeline

> Engineer 1 POC: Dataset generation for training agentic CLI agents with strict JSON tool-calling

## What is this?

This repo contains **eng1.py** — a Python dataset generation pipeline for training CLI agents that:
- Call native tools (bash, file ops, git, web)
- Use strict JSON schema (`{"type":"tool_call",...}`, `{"type":"final_answer",...}`)
- Follow the agent loop from REQUIREMENTS.md
- Output Unsloth-compatible JSONL for fine-tuning

## Quick Start

```bash
# Generate 1000 training samples
python eng1.py generate --samples 1000 --output dataset.jsonl

# Validate and auto-fix
python eng1.py validate --input dataset.jsonl --fix

# Show dataset statistics
python eng1.py stats --input dataset.jsonl

# Replay random samples
python eng1.py replay --input dataset.jsonl --sample 5

# Convert to Unsloth format
python eng1.py convert --input dataset.jsonl --output unsloth_dataset.jsonl
```

## Schema

Each JSONL line follows REQUIREMENTS.md sections 4.1–4.4:

```json
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

## Tools Supported

| Category | Tools |
|----------|-------|
| File System | `read_file`, `write_file`, `update_file`, `delete_file`, `list_files`, `search_files` |
| Bash | `bash` |
| Code Execution | `python_exec`, `node_exec`, `java_exec` |
| Git | `git_pull`, `git_push`, `git_commit`, `git_stash` |
| Web | `web_search`, `fetch_url` |

## Architecture

```
eng1.py generate
    → DatasetGenerator ( TASK_TEMPLATES + simulate_tool_output )
    → TrainingSample { messages: [...] }
    → JSONL file

eng1.py validate
    → ValidationEngine
    ├── JSON structure check
    ├── Message sequence validation
    ├── Tool call validation (TOOL_NAMES registry)
    ├── Hallucination detection
    └── Final answer validation
    → [ --fix → AutoFix → _fixed.jsonl ]

eng1.py convert
    → Unsloth JSONL format
```

## Validation Rules

1. **JSON validity** — Every line parses as valid JSON
2. **Sequence** — Must be: system → user → (tool_call → tool_result)\* → final_answer
3. **Tool names** — Must be in TOOL_NAMES registry
4. **Arguments** — Required args per tool must be present
5. **Final answer** — Must be non-empty `{"type":"final_answer",...}`
6. **Hallucination** — Detects contradictory success/error language

## Dataset Stats (after 1000 samples)

```
Tool distribution:
  bash              ~170 (17%)
  list_files        ~150 (15%)
  read_file         ~120 (12%)
  search_files      ~100 (10%)
  python_exec        ~80 (8%)
  web_search         ~70 (7%)
  write_file         ~60 (6%)
  ... (other tools remaining)
```

## Training with Unsloth

```python
from unsloth import UnslothTrainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./unsloth_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    max_seq_length=4096,
    learning_rate=2e-5,
)
# Target: Qwen 3.6, Gemma 4, or distilled reasoning models
# Constraint: < 12GB VRAM
```

## Files

| File | Purpose |
|------|---------|
| `eng1.py` | Main CLI — generate/validate/stats/replay/convert |
| `POC.md` | Full engineering analysis (gap analysis, architecture, implementation plan) |
| `requirements.txt` | Python dependencies (none — stdlib only) |

## Related Work

- **beastcli** — Production CLI agent (npm, @simpletoolsindia/beastcli v1.1.1)
- **code-cli-plan** — Next-gen CLI with TypeScript+Bun+Ink (Phase 1 complete)
- **agentic-coding-dataset** — Cline-style XML dataset generation

## License

MIT
