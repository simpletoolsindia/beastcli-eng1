# BeastCLI ENG1 — Ultimate Agent Training Dataset Generator

> Production-grade dataset generator for training agentic CLI agents with localization, curriculum learning, and research-backed quality metrics.

## Version 4.0 Features

- **27 tools** across 8 categories
- **12 languages** with localization (English, Hindi, Spanish, French, German, Japanese, Chinese, Korean, etc.)
- **4 tones** (Professional, Casual, Technical, Friendly)
- **4 difficulty levels** (Easy, Medium, Hard, Expert)
- **15% error rate** for robust error recovery training
- **10-15% humanization** for natural responses
- **100% validation pass rate**
- **Industry standard format** (Unsloth compatible)

---

## Quick Start (5 Minutes)

### Step 1: Clone the Repository

```bash
git clone https://github.com/simpletoolsindia/beastcli-eng1.git
cd beastcli-eng1
```

### Step 2: Run the Generator

```bash
# Generate default dataset (2400 examples)
python3 merged_dataset_generator.py
```

**Output:**
```
======================================================================
ULTIMATE AGENT TRAINING DATASET GENERATOR v4.0
======================================================================
Generating dataset with all localizations...
Validation: 2400/2400 valid
Saved 2160 examples to output/merged_train.jsonl
Saved 240 examples to output/merged_test.jsonl
```

### Step 3: Check the Generated Files

```bash
ls -la output/
# merged_train.jsonl (2160 examples)
# merged_test.jsonl (240 examples)
```

**Done!** Your dataset is ready for training.

---

## Complete Usage Guide

### Option A: Quick Generation (Single Command)

```bash
# Generate default dataset (2400 examples, all localizations)
python3 merged_dataset_generator.py

# Check output
ls -la output/
head -1 output/merged_train.jsonl | python3 -m json.tool
```

### Option B: Custom Generation (Python API)

```bash
python3 << 'EOF'
from merged_dataset_generator import (
    ComprehensiveDatasetPipeline,
    Localization,
    DifficultyLevel,
    Language,
    Tone,
    FormalityLevel
)

# Initialize pipeline
pipeline = ComprehensiveDatasetPipeline(seed=42)

# Generate single example
example = pipeline.generate_single(
    localization=Localization(
        language="en",
        tone="professional",
        formality="neutral",
        humanize=True,
        humanize_level="medium"
    ),
    difficulty=DifficultyLevel.MEDIUM,
    include_error=False
)

print(f"Generated: {example.metadata['tool_name']}")
print(f"Difficulty: {example.metadata['difficulty']}")
print(f"Success: {example.metadata['success']}")
EOF
```

### Option C: Batch Generation (Multiple Languages)

```bash
python3 << 'EOF'
from merged_dataset_generator import (
    ComprehensiveDatasetPipeline,
    Localization,
    Language,
    Tone
)
import random

# Initialize pipeline
pipeline = ComprehensiveDatasetPipeline(seed=42)

# Generate batch with multiple localizations
examples = pipeline.generate_localized_batch(
    count_per_locale=25,
    languages=["en", "hi", "es", "fr", "de"],
    tones=["professional", "casual"],
    formalities=["formal", "neutral"]
)

# Validate
valid, stats = pipeline.validator.validate_batch(examples)
print(f"Generated: {len(examples)} examples")
print(f"Valid: {stats['valid']}/{stats['total']}")
EOF
```

### Option D: Using the Original CLI (eng1.py)

```bash
# Generate 1000 samples
python3 eng1.py generate --samples 1000 --output my_dataset.jsonl

# Validate dataset
python3 eng1.py validate --input my_dataset.jsonl

# Show statistics
python3 eng1.py stats --input my_dataset.jsonl

# Replay random samples
python3 eng1.py replay --input my_dataset.jsonl --sample 5

# Convert to Unsloth format
python3 eng1.py convert --input my_dataset.jsonl --output unsloth_dataset.jsonl
```

---

## Dataset Structure

### Generated Files

| File | Examples | Purpose |
|------|----------|---------|
| `output/merged_train.jsonl` | 2,160 (90%) | Training data |
| `output/merged_test.jsonl` | 240 (10%) | Test data |

### JSONL Format

```json
{
  "messages": [
    {"role": "system", "content": "You are an agent..."},
    {"role": "user", "content": "Show me what's in main.py"},
    {"role": "assistant", "content": "{\"type\":\"tool_call\",\"tool_name\":\"File_Read\",\"arguments\":{\"file_path\":\"main.py\"}}"},
    {"role": "tool", "content": "{\"type\":\"tool_result\",\"tool_call_id\":\"call_abc123\",\"output\":\"Hello World!\",\"exit_code\":0}"},
    {"role": "assistant", "content": "{\"type\":\"final_answer\",\"content\":\"The file contains: Hello World!\"}"}
  ],
  "localization": {
    "language": "en",
    "tone": "professional",
    "formality": "neutral",
    "humanize": true,
    "humanize_level": "medium"
  },
  "tools": [...],
  "metadata": {
    "difficulty": "easy",
    "tool_category": "file_operations",
    "tool_name": "File_Read",
    "success": true,
    "generated_at": "2026-04-19T00:00:00Z"
  }
}
```

### Supported Languages

| Language | Code | Example |
|----------|------|---------|
| English | en | "Done." |
| Hindi | hi | "Ho gaya." |
| Spanish | es | "Hecho." |
| French | fr | "Fait." |
| German | de | "Erledigt." |
| Japanese | ja | "Yarimashita." |
| Chinese | zh | "Wancheng le." |
| Korean | ko | "Dohaetseumnida." |
| Arabic | ar | "Ja'al." |
| Russian | ru | "Gotovo." |
| Portuguese | pt | "Feito." |
| Italian | it | "Fatto." |

### Supported Tools (27 Total)

| Category | Tools | Count |
|----------|-------|-------|
| **File Operations** | File_Read, File_Write, File_Search, File_List, File_Delete, File_Copy | 6 |
| **Bash** | Bash_Execute, Bash_ShellStatus | 2 |
| **Code Execution** | Python_Run, Python_Test, Node_Run, JavaScript_Test | 4 |
| **Git** | Git_Status, Git_Log, Git_Commit, Git_Branch, Git_Diff, Git_Pull, Git_Push | 7 |
| **Web** | Web_Search, Web_Fetch, Web_Screenshot | 3 |
| **Search** | Search_Code, Search_Replace | 2 |
| **System** | System_Info, Process_List | 2 |
| **Database** | Database_Query, Database_List | 2 |

---

## Training with Unsloth

### Step 1: Install Dependencies

```bash
pip install unsloth torch transformers datasets
```

### Step 2: Prepare Data

```python
from datasets import load_dataset

# Load your generated dataset
train_dataset = load_dataset("json", data_files="output/merged_train.jsonl", split="train")
test_dataset = load_dataset("json", data_files="output/merged_test.jsonl", split="train")

print(f"Train: {len(train_dataset)} examples")
print(f"Test: {len(test_dataset)} examples")
```

### Step 3: Fine-tune with Unsloth

```python
from unsloth import UnslothTrainer, TrainingArguments
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

training_args = TrainingArguments(
    output_dir="./unsloth_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    max_seq_length=4096,
    fp16=True,
)

trainer = UnslothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

---

## Research Foundation

### Key Papers

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| [TOUCAN](https://arxiv.org/abs/2510.01179) | arXiv 2025 | 1.5M trajectories, 495 MCP servers |
| [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html) | ICML 2025 | 40% agentic, 30% multi-turn |
| [GAIA](https://arxiv.org/abs/2311.12983) | Meta AI | 466+ questions, human baseline 92% |
| [SFT vs RL](https://icml.cc/virtual/2025/poster/44633) | ICML 2025 | SFT stabilizes format, RL generalizes |
| [MCP](https://modelcontextprotocol.io) | Anthropic 2024 | JSON-RPC 2.0 tool protocol |

### Quality Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Validation Pass Rate | >95% | 100% |
| Tool Call Accuracy | >95% | Target |
| Multi-step Success | >80% | Target |
| Error Recovery | >60% | Target |

---

## File Structure

```
beastcli-eng1/
├── README.md                          # This file
├── COMPREHENSIVE_ANALYSIS.md          # Full research analysis
├── merged_dataset_generator.py         # Production generator (v4.0)
├── eng1.py                            # Original CLI tool
├── requirements.txt                   # Python dependencies (none)
│
├── dataset_creator/                    # Module-based generator
│   ├── __init__.py
│   ├── schemas.py                     # Data types
│   ├── tools.py                      # Tool registry + simulators
│   ├── generator.py                  # Sample generation
│   ├── validator.py                  # Validation engine
│   ├── fixer.py                      # Auto-repair
│   ├── quality.py                    # Quality scoring
│   └── pipeline.py                   # Orchestrator
│
└── output/                           # Generated datasets
    ├── merged_train.jsonl            # 2,160 training examples
    └── merged_test.jsonl             # 240 test examples
```

---

## Troubleshooting

### Python Not Found

```bash
# Use python3 instead of python
python3 merged_dataset_generator.py

# Or check your Python installation
which python3
python3 --version
```

### Output Directory Error

```bash
# Create output directory manually
mkdir -p output

# Run again
python3 merged_dataset_generator.py
```

### Permission Denied

```bash
# Make script executable
chmod +x merged_dataset_generator.py

# Run again
./merged_dataset_generator.py
```

### Need More Examples

```python
# Edit merged_dataset_generator.py line ~1980
# Change count_per_locale from 25 to 100+

examples = pipeline.generate_localized_batch(
    count_per_locale=100,  # Increase this
    languages=["en", "hi", "es", "fr", "de", "ja", "zh", "ko"],
    tones=["professional", "casual", "technical", "friendly"],
    formalities=["formal", "neutral", "informal"],
)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## License

MIT

---

## Citation

```bibtex
@software{beastcli-eng1,
  title = {BeastCLI ENG1: Ultimate Agent Training Dataset Generator},
  author = {simpletoolsindia},
  year = {2026},
  version = {4.0},
  url = {https://github.com/simpletoolsindia/beastcli-eng1}
}
```

---

**Last Updated:** 2026-04-19
**Version:** 4.0
**Validation:** 100% pass rate (10 iterations)
