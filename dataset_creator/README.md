# Dataset Creator — Agentic CLI Training Data Generator

A Python module for generating high-quality JSONL training data for agentic CLI agents with tool-calling capabilities.

```
dataset_creator/
├── schemas.py      # Data types: ToolCall, ToolResult, FinalAnswer, TrainingSample
├── tools.py        # Tool registry + realistic output simulators
├── generator.py    # Template-based sample generation
├── validator.py    # Schema validation + quality checks
├── fixer.py        # Auto-repair for common failures
├── quality.py      # Quality scoring (diversity, complexity, hallucination)
└── pipeline.py     # End-to-end pipeline orchestrator
```

## Quick Start

```python
from dataset_creator import Pipeline

# One-liner: generate and export
pipeline = Pipeline(seed=42)
result = pipeline.run(n=1000, output="training_data.jsonl")
print(result.summary())
# {generated: 1000, valid: 847, exported: 847, rate: 84.7%}
```

## Architecture

### Data Flow

```
TaskTemplate → DatasetCreator → TrainingSample
                                    ↓
                            Validator (7 rules)
                                    ↓
                            Fixer (auto-repair)
                                    ↓
                         QualityScorer (4 dimensions)
                                    ↓
                            JSONL export
```

### Schema (REQUIREMENTS.md compliant)

Training samples use a provider-normalized schema compatible with Anthropic, OpenAI, and custom tool-calling formats:

```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert CLI assistant..."},
    {"role": "user",       "content": "Show me the contents of src/main.py"},
    {"role": "assistant",  "content": "{\"type\":\"tool_call\",\"id\":\"call_abc123\",\"tool_name\":\"read_file\",\"arguments\":{\"path\":\"src/main.py\"}}"},
    {"role": "tool",       "content": "{\"type\":\"tool_result\",\"tool_call_id\":\"call_abc123\",\"output\":\"# main.py\\n\\ndef main():\\n    print('Hello!')\\n\",\"exit_code\":0}"},
    {"role": "assistant",  "content": "{\"type\":\"final_answer\",\"content\":\"The file contains...\"}"}
  ]
}
```

**Key design decisions:**
- `tool_call.tool_name` (not `.name`) — provider-normalized
- `tool_call.arguments` (not `.input`) — works across providers
- Assistant content is always a JSON **string**, not a parsed object
- This trains the model to output structured JSON as text

## Usage Examples

### Basic: Generate 100 samples

```python
from dataset_creator import DatasetCreator, Validator, Pipeline

creator = DatasetCreator(seed=42)
samples = creator.generate(n=100)

# Validate
validator = Validator()
issues = validator.validate(samples)
valid = validator.filter(samples, issues)
print(f"Valid: {len(valid)}/{len(samples)}")

# Export
with open("training_data.jsonl", "w") as f:
    for s in valid:
        f.write(s.to_jsonl() + "\n")
```

### Intermediate: Domain-specific generation

```python
# Only git and filesystem tools
creator = DatasetCreator(
    seed=42,
    domains=["git", "filesystem"],
    min_tool_calls=2,
    max_tool_calls=4,
)
samples = creator.generate(n=200)
```

### Advanced: Full pipeline with quality filtering

```python
pipeline = Pipeline(
    seed=42,
    quality_threshold=50.0,
    strict_validation=True,
)

result = pipeline.run(
    n=5000,
    output="training_data.jsonl",
    verbose=True,
)

print(result.summary())
# {generated: 5000, valid: 4231, exported: 3987, rate: 79.7%}
```

### Debugging: Preview samples

```python
pipeline = Pipeline(seed=42)
samples = pipeline.generate_and_preview(n=3, show_jsonl=True)
```

### Validate an existing file

```python
result = pipeline.validate_file("training_data.jsonl")
print(result["stats"])
# {total: 100, errors: 0, warnings: 2, by_rule: {...}}
```

## Tool Registry

16 tools across 5 categories:

| Category | Tools |
|----------|-------|
| **Filesystem** | `read_file`, `write_file`, `update_file`, `delete_file`, `list_files`, `search_files` |
| **Execution** | `bash`, `python_exec`, `node_exec` |
| **Git** | `git_status`, `git_log`, `git_commit`, `git_pull`, `git_push`, `git_stash` |
| **Web** | `web_search`, `fetch_url` |
| **Utility** | `get_timestamp`, `env_get` |

Each tool has an Anthropic-format `input_schema`:

```python
from dataset_creator import TOOL_REGISTRY

for tool in TOOL_REGISTRY:
    schema = tool.to_anthropic_schema()
    print(f"{tool.name}: {schema['input_schema']}")
```

## Validation Rules

The validator checks 8 rules:

| Rule | Severity | Fixable | Description |
|------|----------|---------|-------------|
| `json_validity` | error | yes | All message content must be valid JSON |
| `message_sequence` | error | partial | Valid: system? → user → (assistant→tool)* → final_answer |
| `tool_name_registry` | error | yes | tool_name must exist in TOOL_REGISTRY |
| `required_arguments` | error | yes | All required tool arguments must be present |
| `final_answer_empty` | error | yes | final_answer content must be non-empty |
| `missing_tool_call_id` | error | yes | tool_call must have non-empty ID |
| `hallucination_generic` | warning | yes | final_answer appears too generic |
| `invalid_role` | error | yes | Message role must be system/user/assistant/tool |

## Quality Scoring

Four dimensions (0-100):

| Dimension | Weight | Measures |
|-----------|--------|----------|
| **Diversity** | 30% | Unique tools, category spread |
| **Complexity** | 30% | Tool chain length, result content, reasoning depth |
| **Hallucination Risk** | 40% | Answer grounded in tool results, specificity |
| **Overall** | 100% | Weighted combination |

```python
from dataset_creator import DatasetCreator, QualityScorer

scorer = QualityScorer()
creator = DatasetCreator(seed=42)
samples = creator.generate(n=100)
scores = scorer.score_batch(samples)

for i, score in enumerate(scores):
    print(f"Sample {i}: {score}")
    # Sample 0: QualityScore(d=40.0, c=65.0, h=85.0, overall=64.5)

# Filter by quality
good = scorer.filter_by_score(samples, scores, min_overall=50.0)
```

## Design Principles

Based on research from:

1. **OpenCodeInterpreter** (arXiv:2402.09128): Execution feedback → 9-point HumanEval improvement. Our `simulate_tool()` provides analogous feedback without real execution.

2. **Magicoder / OSS-Instruct** (ICLR 2024): 75K diverse synthetic samples outperform curated datasets. Our template library covers 16 tools across 5 domains.

3. **Anthropic Claude Cookbooks**: JSON Schema `input_schema` enforcement reduces tool call errors. Our schema is provider-normalized.

4. **SWE-agent**: Error recovery in training data improves real-world robustness. Our templates include file-not-found and permission denied scenarios.

## API Reference

### Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `ToolCall`, `ToolResult`, `FinalAnswer`, `Message`, `TrainingSample` | schemas.py | Data types |
| `Tool`, `ToolArg` | schemas.py | Tool definitions |
| `DatasetCreator` | generator.py | Sample generation |
| `Validator`, `ValidationIssue` | validator.py | Schema validation |
| `Fixer`, `FixResult` | fixer.py | Auto-repair |
| `QualityScorer`, `QualityScore` | quality.py | Quality scoring |
| `Pipeline`, `PipelineResult` | pipeline.py | Orchestration |

### TrainingSample Accessors

```python
sample.tool_calls()      # List[ToolCall] — all tool calls
sample.tool_results()    # List[ToolResult] — all tool results
sample.final_answer()     # FinalAnswer or None
sample.user_request()     # str — first user message
sample.system_prompt()    # str — system message
sample.step_count()      # int — number of tool calls
sample.is_valid_schema() # bool — basic validity
sample.token_estimate()   # int — approximate tokens
```

### TrainingSample Builders

```python
sample.add_system("You are an expert CLI assistant.")
sample.add_user("Show me src/main.py")
sample.add_tool_call(ToolCall(tool_name="read_file", arguments={"path": "src/main.py"}))
sample.add_tool_result(ToolResult(tool_call_id="call_abc", output="...", exit_code=0))
sample.add_final_answer(FinalAnswer(content="The file contains..."))
```

## Export Format

JSONL (JSON Lines) — one JSON object per line:

```
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},...]}
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},...]}
```

Compatible with:
- **Unsloth**: `from datasets import load_dataset; ds = load_dataset("json", data_files="training_data.jsonl")`
- **Axolotl**: `datasets: training_data.jsonl`
- **LLama-Factory**: `dataset: training_data`
- **Anthropic fine-tuning**: Convert to their format with the `convert` tool

## Extending

### Add a new tool

1. Add to `TOOL_REGISTRY` in `tools.py`:

```python
Tool(
    name="my_tool",
    description="Does something useful",
    category="custom",
    args=[
        ToolArg(name="param1", description="...", required=True),
        ToolArg(name="param2", description="...", required=False, default="x"),
    ],
    tags=["custom"],
)
```

2. Add simulator branch in `simulate_tool()`:

```python
elif tool_name == "my_tool":
    param = arguments.get("param1", "")
    return ToolResult(tool_call_id=call_id, output=f"Result: {param}\n", exit_code=0)
```

### Add a new task template

Add to the `TEMPLATES` list in `generator.py`:

```python
TaskTemplate(
    description="My new task",
    system_prompt="You are an expert...",
    user_request="Do the thing",
    steps=[
        {"tool_name": "my_tool", "arguments": {"param1": "value"}},
    ],
    domain="custom",
    complexity=2,
)
```

## Research Basis

See `/home/sridhar/beastcli-eng1/POC.md` for comprehensive research covering:
- Claude Code architecture and tool calling
- Cline checkpoint snapshots
- SWE-agent and OpenCodeInterpreter
- Magicoder OSS-Instruct
- Anthropic Claude Cookbooks
- Provider comparison (Ollama, LM Studio, vLLM)
- Qwen3 and Gemma4 analysis
- MCP JSON-RPC 2.0 architecture
- Unsloth training optimization
- Sandbox security (bubblewrap, Landlock, Docker)
- JSON schema enforcement (xgrammar, json_repair)
- 4-phase implementation roadmap

## License

MIT
