#!/usr/bin/env python3
"""
eng1.py — BeastCLI Engineer 1: Dataset Generation Pipeline

Generates high-quality JSONL training data for agentic CLI agents.
Follows REQUIREMENTS.md spec sections 9-10:
  - Standard message schema (tool_call, tool_result, final_answer)
  - JSONL format for Unsloth training
  - Validation loop with auto-fix
  - Hallucination detection

Usage:
  python eng1.py generate --samples 1000 --output dataset.jsonl
  python eng1.py validate --input dataset.jsonl --fix
  python eng1.py stats --input dataset.jsonl
  python eng1.py replay --input dataset.jsonl --sample 5
"""

import argparse
import json
import random
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import statistics

# ---------------------------------------------------------------------------
# Schema Types (REQUIREMENTS.md sections 4.1-4.4, 9.3)
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    USER = "user"
    ASSISTANT_TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """REQUIREMENTS.md section 4.2: Assistant Tool Call schema"""
    type: str = "tool_call"
    id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(
            type=d.get("type", "tool_call"),
            id=d.get("id", ""),
            tool_name=d.get("tool_name", ""),
            arguments=d.get("arguments", {}),
        )


@dataclass
class ToolResult:
    """REQUIREMENTS.md section 4.3: Tool Response schema"""
    type: str = "tool_result"
    tool_call_id: str = ""
    output: str = ""
    exit_code: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "exit_code": self.exit_code,
        }
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToolResult":
        return cls(
            type=d.get("type", "tool_result"),
            tool_call_id=d.get("tool_call_id", ""),
            output=d.get("output", ""),
            exit_code=d.get("exit_code", 0),
            error=d.get("error"),
        )


@dataclass
class FinalAnswer:
    """REQUIREMENTS.md section 4.4: Final Answer schema"""
    type: str = "final_answer"
    content: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> "FinalAnswer":
        return cls(
            type=d.get("type", "final_answer"),
            content=d.get("content", ""),
        )


@dataclass
class TrainingSample:
    """REQUIREMENTS.md section 9.3: Unsloth training format"""
    messages: list = field(default_factory=list)

    def to_jsonl(self) -> str:
        return json.dumps({"messages": self.messages}, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "TrainingSample":
        d = json.loads(line)
        return cls(messages=d.get("messages", []))


# ---------------------------------------------------------------------------
# Tool Registry (REQUIREMENTS.md section 5: Native Tool System)
# ---------------------------------------------------------------------------

# Each tool: required_args are mandatory, optional_args have defaults
TOOLS: list[dict] = [
    # 5.1 File System Tools
    {"name": "read_file", "required": ["path"], "optional": ["offset", "limit"], "description": "Read file contents"},
    {"name": "write_file", "required": ["path", "content"], "optional": [], "description": "Write content to file"},
    {"name": "update_file", "required": ["path", "old_string", "new_string"], "optional": [], "description": "Edit file via string replacement"},
    {"name": "delete_file", "required": ["path"], "optional": [], "description": "Delete a file"},
    {"name": "list_files", "required": ["path"], "optional": ["limit", "offset"], "description": "List files with pagination"},
    {"name": "search_files", "required": ["path"], "optional": ["pattern", "glob", "limit"], "description": "Search files by content/pattern"},

    # 5.2 Bash Execution Tool
    {"name": "bash", "required": ["command"], "optional": ["timeout"], "description": "Execute bash command"},

    # 5.3 Code Execution Tools
    {"name": "python_exec", "required": ["file"], "optional": ["args"], "description": "Run Python file"},
    {"name": "node_exec", "required": ["file"], "optional": ["args"], "description": "Run Node.js file"},
    {"name": "java_exec", "required": ["file"], "optional": [], "description": "Run Java file"},

    # 5.4 Git Tools
    {"name": "git_pull", "required": [], "optional": ["repo"], "description": "Git pull"},
    {"name": "git_push", "required": [], "optional": ["repo", "branch"], "description": "Git push"},
    {"name": "git_commit", "required": ["message"], "optional": ["files"], "description": "Git commit"},
    {"name": "git_stash", "required": [], "optional": [], "description": "Git stash"},

    # 5.5 Web Tools
    {"name": "web_search", "required": ["query"], "optional": [], "description": "Search the web"},
    {"name": "fetch_url", "required": ["url"], "optional": [], "description": "Fetch URL and return markdown/text"},
]

TOOL_NAMES = [t["name"] for t in TOOLS]

# ---------------------------------------------------------------------------
# Realistic Output Simulators
# ---------------------------------------------------------------------------

def simulate_tool_output(tool_name: str, args: dict) -> tuple[str, int, Optional[str]]:
    """Simulate realistic tool outputs for training data generation."""
    try:
        if tool_name == "list_files":
            path = args.get("path", ".")
            limit = args.get("limit", 50)
            files = [f"file_{i}.txt" for i in range(min(limit, random.randint(3, 12)))]
            output = "\n".join(files)
            return output, 0, None

        elif tool_name == "read_file":
            path = args.get("path", "")
            if "hello" in path.lower():
                return "Hello, World!\n", 0, None
            elif "main" in path.lower():
                return f"// {path}\ndef main() {{\n  console.log('Hello');\n}}\n", 0, None
            return f"# {path}\ncontent here\n", 0, None

        elif tool_name == "bash":
            cmd = args.get("command", "")
            if "ls" in cmd:
                return "file1.txt\nfile2.py\nREADME.md\n", 0, None
            elif "pwd" in cmd:
                return "/home/user/project\n", 0, None
            elif "echo" in cmd:
                return cmd + "\n", 0, None
            elif "npm install" in cmd or "pip install" in cmd:
                return "Successfully installed dependencies.\n", 0, None
            elif "git status" in cmd:
                return "On branch main\nnothing to commit, working tree clean\n", 0, None
            elif "python" in cmd or "node" in cmd:
                return "Hello, World!\n", 0, None
            return f"Command executed: {cmd[:50]}\n", 0, None

        elif tool_name == "search_files":
            pattern = args.get("pattern", "")
            results = [f"src/{d}/module_{i}.py" for d in ["core", "utils"] for i in range(3)]
            output = "\n".join(results[:random.randint(2, 6)])
            return output, 0, None

        elif tool_name == "python_exec":
            return "Hello, World!\n", 0, None

        elif tool_name == "node_exec":
            return "Hello, World!\n", 0, None

        elif tool_name == "web_search":
            query = args.get("query", "")
            return f"Results for '{query}':\n1. https://example.com/result1\n2. https://example.com/result2\n", 0, None

        elif tool_name == "fetch_url":
            return "# Fetched Page\nContent here...\n", 0, None

        elif tool_name == "git_commit":
            return "[main abc1234] " + args.get("message", "commit") + "\n 1 file changed, 10 insertions(+)\n", 0, None

        elif tool_name == "git_pull":
            return "Already up to date.\n", 0, None

        elif tool_name == "git_push":
            return "Everything up-to-date\n", 0, None

        elif tool_name == "git_stash":
            return "Saved working directory and index state WIP on main: ...\n", 0, None

        elif tool_name == "write_file":
            return f"Wrote {len(args.get('content', ''))} bytes to {args.get('path', 'file')}\n", 0, None

        elif tool_name == "delete_file":
            return f"Deleted {args.get('path', 'file')}\n", 0, None

        elif tool_name == "update_file":
            return "File updated successfully\n", 0, None

        else:
            return f"Tool '{tool_name}' executed successfully\n", 0, None

    except Exception as e:
        return "", 1, str(e)


# ---------------------------------------------------------------------------
# Task Templates (REQUIREMENTS.md section 3: Agent Loop)
# ---------------------------------------------------------------------------

TASK_TEMPLATES = [
    {
        "user": "List files in the current directory",
        "steps": [
            ("tool_call", "list_files", {"path": ".", "limit": 50}),
            ("tool_call", "bash", {"command": "ls -la"}),
        ],
        "final": "Found {count} files in the current directory.",
    },
    {
        "user": "Read the contents of hello.py",
        "steps": [
            ("tool_call", "read_file", {"path": "hello.py"}),
        ],
        "final": "The file contains: Hello, World!",
    },
    {
        "user": "Run hello.py and show me the output",
        "steps": [
            ("tool_call", "read_file", {"path": "hello.py"}),
            ("tool_call", "python_exec", {"file": "hello.py", "args": []}),
        ],
        "final": "Program executed successfully with output: Hello, World!",
    },
    {
        "user": "Search for all Python files in src/",
        "steps": [
            ("tool_call", "search_files", {"path": "src", "pattern": "*.py", "limit": 100}),
        ],
        "final": "Found {count} Python files in src/",
    },
    {
        "user": "Create a new Python file called greet.py with a hello function",
        "steps": [
            ("tool_call", "write_file", {"path": "greet.py", "content": "def hello():\n    print('Hello, World!')\n"}),
            ("tool_call", "bash", {"command": "python greet.py"}),
        ],
        "final": "Created greet.py with hello function and verified it works.",
    },
    {
        "user": "Check git status and make a commit",
        "steps": [
            ("tool_call", "bash", {"command": "git status"}),
            ("tool_call", "git_commit", {"message": "Add hello world example", "files": ["hello.py"]}),
        ],
        "final": "Committed changes with message: 'Add hello world example'",
    },
    {
        "user": "Install npm dependencies for this project",
        "steps": [
            ("tool_call", "bash", {"command": "npm install"}),
        ],
        "final": "npm dependencies installed successfully.",
    },
    {
        "user": "Find all files containing 'TODO' comments",
        "steps": [
            ("tool_call", "search_files", {"path": ".", "pattern": "TODO", "limit": 50}),
        ],
        "final": "Found {count} TODO comments across {files} files.",
    },
    {
        "user": "Run the test suite and show results",
        "steps": [
            ("tool_call", "bash", {"command": "python -m pytest --tb=short", "timeout": 60}),
        ],
        "final": "Test suite completed. All tests passed.",
    },
    {
        "user": "Search the web for the latest Python release version",
        "steps": [
            ("tool_call", "web_search", {"query": "latest Python release version 2026"}),
            ("tool_call", "fetch_url", {"url": "https://www.python.org/downloads/"}),
        ],
        "final": "The latest Python version is Python 3.14.",
    },
    {
        "user": "Edit config.json to change the port to 8080",
        "steps": [
            ("tool_call", "read_file", {"path": "config.json"}),
            ("tool_call", "update_file", {"path": "config.json", "old_string": '"port": 3000', "new_string": '"port": 8080'}),
        ],
        "final": "Updated config.json: port changed from 3000 to 8080.",
    },
    {
        "user": "List all files with pagination (first 10)",
        "steps": [
            ("tool_call", "list_files", {"path": "src", "limit": 10, "offset": 0}),
        ],
        "final": "Showing files 1-10. Use offset=10 for next page.",
    },
    {
        "user": "Delete the old backup file",
        "steps": [
            ("tool_call", "bash", {"command": "ls -la *.bak 2>/dev/null || echo 'No backup files'"}),
            ("tool_call", "delete_file", {"path": "backup_old.bak"}),
        ],
        "final": "Deleted backup_old.bak",
    },
    {
        "user": "Git pull latest changes from main",
        "steps": [
            ("tool_call", "bash", {"command": "git pull origin main"}),
        ],
        "final": "Repository is up to date with main branch.",
    },
    {
        "user": "Show me the last 20 lines of the log file",
        "steps": [
            ("tool_call", "read_file", {"path": "app.log", "offset": -20}),
        ],
        "final": "Last 20 lines of app.log:\n...",
    },
]


# ---------------------------------------------------------------------------
# Validation Engine (REQUIREMENTS.md section 10)
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    line_number: int
    sample_id: str
    issue_type: str  # "invalid_json" | "wrong_tool" | "hallucination" | "missing_field"
    message: str
    severity: str = "error"  # "error" | "warning"


class ValidationEngine:
    """
    REQUIREMENTS.md section 10: Dataset Generation Loop
    Detects: invalid JSON, wrong tool usage, hallucination
    """

    def __init__(self):
        self.issues: list[ValidationIssue] = []
        self.fixes_applied: int = 0

    def validate(self, samples: list[TrainingSample]) -> list[ValidationIssue]:
        """Run all validation checks on dataset."""
        self.issues = []
        for idx, sample in enumerate(samples):
            self._check_json_structure(idx, sample)
            self._check_message_sequence(idx, sample)
            self._check_tool_calls(idx, sample)
            self._check_hallucinations(idx, sample)
            self._check_final_answer(idx, sample)
        return self.issues

    def _check_json_structure(self, idx: int, sample: TrainingSample):
        """Check for valid JSON and required fields."""
        try:
            d = {"messages": sample.messages}
            json.dumps(d)
        except (TypeError, ValueError) as e:
            self.issues.append(ValidationIssue(
                line_number=idx + 1,
                sample_id=self._get_sample_id(sample),
                issue_type="invalid_json",
                message=f"Invalid JSON structure: {e}",
            ))

    def _check_message_sequence(self, idx: int, sample: TrainingSample):
        """Enforce user → assistant → tool → assistant → ... → final_answer pattern."""
        if not sample.messages:
            self.issues.append(ValidationIssue(
                line_number=idx + 1,
                sample_id=self._get_sample_id(sample),
                issue_type="missing_field",
                message="Empty messages array",
                severity="error",
            ))
            return

        # Must have user message (system is allowed first, per spec section 9.3 example)
        first_non_system = next((m for m in sample.messages if m.get("role") != "system"), None)
        if first_non_system is None:
            self.issues.append(ValidationIssue(
                line_number=idx + 1,
                sample_id=self._get_sample_id(sample),
                issue_type="missing_field",
                message="No user message found",
                severity="error",
            ))
        elif first_non_system.get("role") != "user":
            self.issues.append(ValidationIssue(
                line_number=idx + 1,
                sample_id=self._get_sample_id(sample),
                issue_type="wrong_tool",
                message=f"First non-system message must be role=user, got role={first_non_system.get('role')}",
            ))

        # Must end with final_answer
        last = sample.messages[-1]
        try:
            last_content = json.loads(last.get("content", "{}"))
            if last_content.get("type") != "final_answer":
                self.issues.append(ValidationIssue(
                    line_number=idx + 1,
                    sample_id=self._get_sample_id(sample),
                    issue_type="wrong_tool",
                    message="Last message must be final_answer",
                ))
        except (json.JSONDecodeError, TypeError):
            if last.get("role") != "assistant":
                self.issues.append(ValidationIssue(
                    line_number=idx + 1,
                    sample_id=self._get_sample_id(sample),
                    issue_type="wrong_tool",
                    message="Last message must be final_answer in assistant role",
                ))

    def _check_tool_calls(self, idx: int, sample: TrainingSample):
        """Verify tool_name is in TOOL_NAMES and arguments are valid."""
        for msg in sample.messages:
            if msg.get("role") != "assistant":
                continue
            try:
                content = json.loads(msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue

            if content.get("type") == "tool_call":
                tool_name = content.get("tool_name", "")
                if tool_name not in TOOL_NAMES:
                    self.issues.append(ValidationIssue(
                        line_number=idx + 1,
                        sample_id=self._get_sample_id(sample),
                        issue_type="wrong_tool",
                        message=f"Unknown tool: '{tool_name}'. Valid: {TOOL_NAMES}",
                        severity="error",
                    ))

                tool_args = content.get("arguments", {})
                tool_def = next((t for t in TOOLS if t["name"] == tool_name), None)
                if tool_def:
                    for req_arg in tool_def["required"]:
                        if req_arg not in tool_args:
                            self.issues.append(ValidationIssue(
                                line_number=idx + 1,
                                sample_id=self._get_sample_id(sample),
                                issue_type="missing_field",
                                message=f"Tool '{tool_name}' missing required argument: {req_arg}",
                                severity="warning",
                            ))

    def _check_hallucinations(self, idx: int, sample: TrainingSample):
        """Detect hallucinated patterns: impossible outputs, wrong file paths in tool_result."""
        HALLUCINATION_PATTERNS = [
            r"Access denied",
            r"Permission denied.*root",
            r"command not found.*definitely",
            r"file does not exist.*but I created it",
            r"the file shows.*but actually",
        ]
        for msg in sample.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    parsed = json.loads(content)
                    if parsed.get("type") == "final_answer":
                        # Check for contradictory statements
                        if "error" in content.lower() and "successfully" in content.lower():
                            self.issues.append(ValidationIssue(
                                line_number=idx + 1,
                                sample_id=self._get_sample_id(sample),
                                issue_type="hallucination",
                                message="Contradictory success/error language in final_answer",
                                severity="warning",
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass

    def _check_final_answer(self, idx: int, sample: TrainingSample):
        """Verify final_answer has non-empty content."""
        for msg in sample.messages:
            if msg.get("role") != "assistant":
                continue
            try:
                content = json.loads(msg.get("content", "{}"))
                if content.get("type") == "final_answer":
                    if not content.get("content", "").strip():
                        self.issues.append(ValidationIssue(
                            line_number=idx + 1,
                            sample_id=self._get_sample_id(sample),
                            issue_type="missing_field",
                            message="final_answer.content is empty",
                        ))
            except (json.JSONDecodeError, TypeError):
                pass

    def _get_sample_id(self, sample: TrainingSample) -> str:
        for msg in sample.messages:
            if msg.get("role") == "user":
                return msg.get("content", "")[:50]
        return "unknown"

    def fix_sample(self, sample: TrainingSample) -> TrainingSample:
        """Auto-fix common issues in a sample."""
        messages = []
        for msg in sample.messages:
            msg = dict(msg)  # shallow copy
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    parsed = json.loads(content)
                    if parsed.get("type") == "tool_call":
                        # Ensure id is present
                        if not parsed.get("id"):
                            parsed["id"] = f"call_{uuid.uuid4().hex[:8]}"
                        # Ensure tool_name is valid
                        if parsed.get("tool_name") not in TOOL_NAMES:
                            parsed["tool_name"] = "bash"
                        # Ensure arguments is a dict
                        if not isinstance(parsed.get("arguments"), dict):
                            parsed["arguments"] = {}
                        msg["content"] = json.dumps(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
            messages.append(msg)
        self.fixes_applied += 1
        return TrainingSample(messages=messages)


# ---------------------------------------------------------------------------
# Dataset Generator (REQUIREMENTS.md sections 9, 11)
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    REQUIREMENTS.md section 9: Dataset Design
    Generates JSONL training samples with strict schema enforcement.
    """

    SYSTEM_PROMPT = (
        "You are an agentic CLI assistant. "
        "Output MUST be valid JSON. "
        'Use {"type":"tool_call","tool_name":"...","arguments":{...}} to call tools. '
        'Use {"type":"final_answer","content":"..."} when done. '
        "NEVER return plain text. NEVER hallucinate file contents or tool results."
    )

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.stats = {
            "total_generated": 0,
            "by_tool": {},
            "by_steps": {},
            "total_tokens_estimate": 0,
        }

    def generate_sample(self, template: dict) -> TrainingSample:
        """Generate a single training sample from a template."""
        messages = []

        # 1. System message
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT,
        })

        # 2. User message (section 4.1)
        user_content = template["user"]
        messages.append({
            "role": "user",
            "content": user_content,
        })

        # 3. Assistant tool calls (section 4.2) + Tool results (section 4.3)
        for step in template["steps"]:
            step_type, tool_name, tool_args = step
            if step_type == "tool_call":
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = ToolCall(
                    type="tool_call",
                    id=call_id,
                    tool_name=tool_name,
                    arguments=tool_args,
                )
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(tool_call.to_dict(), ensure_ascii=False),
                })

                # Simulate tool result
                output, exit_code, error = simulate_tool_output(tool_name, tool_args)
                tool_result = ToolResult(
                    type="tool_result",
                    tool_call_id=call_id,
                    output=output,
                    exit_code=exit_code,
                    error=error,
                )
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result.to_dict(), ensure_ascii=False),
                })

                # Track stats
                self.stats["total_generated"] += 1
                self.stats["by_tool"][tool_name] = self.stats["by_tool"].get(tool_name, 0) + 1

        # 4. Final answer (section 4.4)
        step_count = len(template["steps"])
        self.stats["by_steps"][step_count] = self.stats["by_steps"].get(step_count, 0) + 1

        final_content = template["final"]
        # Fill in dynamic placeholders
        if "{count}" in final_content:
            final_content = final_content.replace("{count}", str(random.randint(1, 20)))
        if "{files}" in final_content:
            final_content = final_content.replace("{files}", str(random.randint(1, 5)))

        final_answer = FinalAnswer(type="final_answer", content=final_content)
        messages.append({
            "role": "assistant",
            "content": json.dumps(final_answer.to_dict(), ensure_ascii=False),
        })

        return TrainingSample(messages=messages)

    def generate_batch(self, count: int, include_variations: bool = True) -> list[TrainingSample]:
        """Generate multiple samples with variations."""
        samples = []
        templates = TASK_TEMPLATES * ((count // len(TASK_TEMPLATES)) + 1)
        self.rng.shuffle(templates)

        for i in range(count):
            template = templates[i % len(TASK_TEMPLATES)]
            sample = self.generate_sample(template)

            if include_variations and i % 3 == 0:
                # Add system prompt variation for robustness
                sample = self._add_variation(sample)

            samples.append(sample)

        return samples

    def _add_variation(self, sample: TrainingSample) -> TrainingSample:
        """Add slight variations to increase dataset diversity."""
        messages = []
        for msg in sample.messages:
            msg = dict(msg)
            if msg["role"] == "system":
                # Alternate system prompts
                variations = [
                    self.SYSTEM_PROMPT,
                    self.SYSTEM_PROMPT + " Always explain what you are doing.",
                    self.SYSTEM_PROMPT + " Be concise and efficient.",
                ]
                msg["content"] = self.rng.choice(variations)
            messages.append(msg)
        return TrainingSample(messages=messages)

    def estimate_tokens(self, samples: list[TrainingSample]) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        total_chars = sum(
            sum(len(json.dumps(m)) for m in s.messages)
            for s in samples
        )
        return total_chars // 4


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_generate(args) -> int:
    """Generate JSONL dataset."""
    print(f"Generating {args.samples} samples...")

    gen = DatasetGenerator(seed=args.seed)
    samples = gen.generate_batch(args.samples, include_variations=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_jsonl() + "\n")

    token_estimate = gen.estimate_tokens(samples)
    print(f"Generated {len(samples)} samples")
    print(f"Written to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"Estimated tokens: {token_estimate:,}")
    print()
    print("Tool distribution:")
    for tool, count in sorted(gen.stats["by_tool"].items()):
        pct = count / gen.stats["total_generated"] * 100
        print(f"  {tool:<20} {count:>6} ({pct:>5.1f}%)")
    print()
    print("Steps distribution:")
    for steps, count in sorted(gen.stats["by_steps"].items()):
        print(f"  {steps} step(s): {count} samples")

    return 0


def cmd_validate(args) -> int:
    """Validate dataset and optionally fix issues."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Validating: {input_path}")
    lines = input_path.read_text(encoding="utf-8").splitlines()
    samples = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            samples.append(TrainingSample(messages=d.get("messages", [])))
        except json.JSONDecodeError as e:
            print(f"Line {i + 1}: INVALID JSON — {e}")
            if args.fix:
                print(f"  (would skip or fix this line)")

    engine = ValidationEngine()
    issues = engine.validate(samples)

    error_count = sum(1 for i in issues if i.severity == "error")
    warn_count = sum(1 for i in issues if i.severity == "warning")

    print(f"\nValidation complete:")
    print(f"  Samples checked: {len(samples)}")
    print(f"  Errors: {error_count}")
    print(f"  Warnings: {warn_count}")

    if issues:
        print(f"\nIssues found:")
        for issue in issues[:50]:  # Show first 50
            print(f"  Line {issue.line_number}: [{issue.issue_type}/{issue.severity}] {issue.message}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more")

    if args.fix and issues:
        fixed = 0
        fixed_samples = []
        for sample in samples:
            has_issues = any(s.line_number == samples.index(sample) + 1 and s.severity == "error"
                           for s in issues)
            if has_issues:
                sample = engine.fix_sample(sample)
                fixed += 1
            fixed_samples.append(sample)

        fixed_path = input_path.with_name(input_path.stem + "_fixed.jsonl")
        with open(fixed_path, "w", encoding="utf-8") as f:
            for sample in fixed_samples:
                f.write(sample.to_jsonl() + "\n")

        print(f"\nFixed {fixed} samples → {fixed_path}")
        print(f"Fixes applied: {engine.fixes_applied}")

    print(f"\nStatus: {'PASS' if error_count == 0 else 'FAIL'}")
    return 0 if error_count == 0 else 1


def cmd_stats(args) -> int:
    """Show dataset statistics."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 1

    lines = input_path.read_text(encoding="utf-8").splitlines()
    samples = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            samples.append(TrainingSample(messages=d.get("messages", [])))
        except json.JSONDecodeError:
            continue

    total_messages = sum(len(s.messages) for s in samples)
    tool_call_count = 0
    final_answer_count = 0
    tool_distribution: dict[str, int] = {}
    step_counts: list[int] = []
    roles: dict[str, int] = {}
    empty_finals = 0

    for sample in samples:
        step_count = 0
        for msg in sample.messages:
            roles[msg.get("role", "unknown")] = roles.get(msg.get("role", "unknown"), 0) + 1
            try:
                content = json.loads(msg.get("content", "{}"))
                if content.get("type") == "tool_call":
                    tool_call_count += 1
                    tool_name = content.get("tool_name", "unknown")
                    tool_distribution[tool_name] = tool_distribution.get(tool_name, 0) + 1
                    step_count += 1
                elif content.get("type") == "final_answer":
                    final_answer_count += 1
                    if not content.get("content", "").strip():
                        empty_finals += 1
            except (json.JSONDecodeError, TypeError):
                pass
        step_counts.append(step_count)

    print(f"\n{'='*50}")
    print(f"Dataset Statistics: {input_path.name}")
    print(f"{'='*50}")
    print(f"  Total samples:    {len(samples):>8,}")
    print(f"  Total messages:   {total_messages:>8,}")
    print(f"  Tool calls:       {tool_call_count:>8,}")
    print(f"  Final answers:    {final_answer_count:>8,}")
    print(f"  Empty finals:     {empty_finals:>8,}")
    print(f"  Avg msgs/sample:  {total_messages / len(samples):>8.1f}")
    if step_counts:
        print(f"  Avg steps/sample: {statistics.mean(step_counts):>8.1f}")
        print(f"  Max steps:        {max(step_counts):>8,}")
        print(f"  Min steps:        {min(step_counts):>8,}")
    print()
    print(f"Role distribution:")
    for role, count in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"  {role:<15} {count:>8,} ({count / total_messages * 100:>5.1f}%)")
    print()
    print(f"Tool distribution:")
    for tool, count in sorted(tool_distribution.items(), key=lambda x: -x[1]):
        print(f"  {tool:<20} {count:>6,} ({count / tool_call_count * 100:>5.1f}%)")
    print(f"{'='*50}\n")

    return 0


def cmd_replay(args) -> int:
    """Replay a sample with formatted output."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 1

    lines = input_path.read_text(encoding="utf-8").splitlines()
    non_empty = [l for l in lines if l.strip()]
    total = len(non_empty)

    indices = list(range(total))
    random.seed(args.seed)
    random.shuffle(indices)
    sample_indices = indices[:args.sample]

    print(f"Replaying {len(sample_indices)} / {total} samples (seed={args.seed})\n")

    for si in sample_indices:
        line = non_empty[si].strip()
        try:
            d = json.loads(line)
            messages = d.get("messages", [])
        except json.JSONDecodeError:
            print(f"[{si + 1}] INVALID JSON")
            continue

        print(f"{'─' * 60}")
        print(f"[Sample {si + 1}]")
        for msg in messages:
            role = msg.get("role", "?")
            try:
                content = json.loads(msg.get("content", "{}"))
                ctype = content.get("type", "text")
                if ctype == "tool_call":
                    print(f"  [{role}] TOOL_CALL: {content.get('tool_name')} {content.get('arguments')}")
                elif ctype == "tool_result":
                    ec = content.get("exit_code", 0)
                    out = content.get("output", "")[:100]
                    print(f"  [tool] RESULT: exit={ec} output={out!r}")
                elif ctype == "final_answer":
                    print(f"  [{role}] FINAL_ANSWER: {content.get('content', '')[:80]}")
                else:
                    print(f"  [{role}] {str(content)[:80]}")
            except (json.JSONDecodeError, TypeError):
                print(f"  [{role}] {msg.get('content', '')[:80]}")
        print()
    return 0


def cmd_convert(args) -> int:
    """Convert dataset to Unsloth-compatible format."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_unsloth.jsonl"
    )

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 1

    lines = input_path.read_text(encoding="utf-8").splitlines()
    converted = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                messages = d.get("messages", [])

                # Already in correct format for Unsloth
                unsloth_record = {"messages": messages}
                out.write(json.dumps(unsloth_record, ensure_ascii=False) + "\n")
                converted += 1
            except (json.JSONDecodeError, KeyError) as e:
                skipped += 1
                if args.verbose:
                    print(f"Skipped line: {e}")

    print(f"Converted {converted} samples → {output_path}")
    print(f"Skipped: {skipped}")
    return 0


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="eng1.py",
        description="BeastCLI Engineer 1: Dataset Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eng1.py generate --samples 1000 --output dataset.jsonl
  python eng1.py validate --input dataset.jsonl --fix
  python eng1.py stats --input dataset.jsonl
  python eng1.py replay --input dataset.jsonl --sample 5
  python eng1.py convert --input dataset.jsonl --output unsloth_dataset.jsonl
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    g = sub.add_parser("generate", help="Generate JSONL training dataset")
    g.add_argument("--samples", "-n", type=int, default=100, help="Number of samples to generate")
    g.add_argument("--output", "-o", type=str, default="dataset.jsonl", help="Output file path")
    g.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    # validate
    v = sub.add_parser("validate", help="Validate dataset and fix issues")
    v.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    v.add_argument("--fix", "-f", action="store_true", help="Auto-fix issues")

    # stats
    s = sub.add_parser("stats", help="Show dataset statistics")
    s.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")

    # replay
    r = sub.add_parser("replay", help="Replay random samples with formatted output")
    r.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    r.add_argument("--sample", "-n", type=int, default=5, help="Number of samples to show")
    r.add_argument("--seed", "-s", type=int, default=42, help="Random seed for sampling")

    # convert
    c = sub.add_parser("convert", help="Convert to Unsloth format")
    c.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    c.add_argument("--output", "-o", type=str, help="Output file (default: input_stem_unsloth.jsonl)")
    c.add_argument("--verbose", "-v", action="store_true", help="Show skipped lines")

    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "replay": cmd_replay,
        "convert": cmd_convert,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
