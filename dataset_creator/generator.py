"""
generator.py — Sample Generation Engine
=======================================

Generates diverse, realistic training samples using template-based
synthetic generation with controlled randomness.

Generation Strategy
-------------------
Based on OpenCodeInterpreter + Magicoder research findings:
1. Template-driven: Each domain has curated task templates
2. Tool chaining: 2-5 tools per sample for reasoning depth
3. Realistic outputs: Simulated tool results mirror real tool behavior
4. Error recovery: Some samples include error handling paths
5. Diverse domains: filesystem, git, code execution, web, utility

Research basis: OpenCodeInterpreter paper (arXiv:2402.09128) showed
that execution feedback → 9-point improvement on HumanEval. Our
simulator provides analogous feedback without real execution.

Magicoder (ICLR 2024) showed that OSS-Instruct (75K synthetic samples)
significantly outperforms curated datasets. Our template-based approach
follows similar principles with domain-expert-curated templates.
"""

from __future__ import annotations

import random
import hashlib
import json
import re
from typing import Any, Optional

from dataset_creator.schemas import (
    ToolCall,
    ToolResult,
    FinalAnswer,
    TrainingSample,
    Message,
)
from dataset_creator.tools import (
    TOOL_REGISTRY_BY_NAME,
    TOOL_NAMES,
    simulate_tool,
    SimulatedFilesystem,
)


# --------------------------------------------------------------------------
# Task Templates
# --------------------------------------------------------------------------
# Each template defines a complete sample with:
#   system_prompt: What the agent is told
#   user_request: The user's actual request
#   steps: Ordered list of tool calls + expected final answer
#   description: Human-readable template name for debugging

class TaskTemplate:
    """A single task template for sample generation."""

    def __init__(
        self,
        description: str,
        system_prompt: str,
        user_request: str,
        steps: list[dict],
        domain: str = "general",
        complexity: int = 1,
    ):
        self.description = description
        self.system_prompt = system_prompt
        self.user_request = user_request
        self.steps = steps  # list of {"tool_name": ..., "arguments": {...}}
        self.domain = domain
        self.complexity = complexity  # 1-5 (number of tool calls approx)


# --------------------------------------------------------------------------
# Template Library
# --------------------------------------------------------------------------
# Organized by domain. Each template produces 1 sample when instantiated.
# Templates use SimulatedFilesystem paths for realism.

TEMPLATES: list[TaskTemplate] = [

    # =====================================================================
    # File System — Single Tool (complexity 1)
    # =====================================================================
    TaskTemplate(
        description="Read a file",
        system_prompt=(
            "You are an expert CLI assistant. Use the appropriate tools to complete "
            "each task efficiently. Always read files before modifying them."
        ),
        user_request="Show me the contents of src/main.py",
        steps=[
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/main.py"},
            },
        ],
        domain="filesystem",
        complexity=1,
    ),
    TaskTemplate(
        description="List directory contents",
        system_prompt=(
            "You are an expert CLI assistant. Use filesystem tools to explore "
            "and navigate project directories."
        ),
        user_request="What files are in the current directory?",
        steps=[
            {
                "tool_name": "list_files",
                "arguments": {"path": "."},
            },
        ],
        domain="filesystem",
        complexity=1,
    ),
    TaskTemplate(
        description="Search for files by glob",
        system_prompt=(
            "You are an expert CLI assistant. Use search_files to find files "
            "matching patterns across project directories."
        ),
        user_request="Find all Python test files in this project",
        steps=[
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.py"},
            },
        ],
        domain="filesystem",
        complexity=1,
    ),

    # =====================================================================
    # File System — Two Tools (complexity 2)
    # =====================================================================
    TaskTemplate(
        description="Read then update a file",
        system_prompt=(
            "You are an expert CLI assistant. Always read a file before editing it. "
            "Use update_file with exact old_string matching for safe edits."
        ),
        user_request="Update the version number in src/__init__.py from 0.1.0 to 0.2.0",
        steps=[
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/__init__.py"},
            },
            {
                "tool_name": "update_file",
                "arguments": {
                    "path": "src/__init__.py",
                    "old_string": '__version__ = "0.1.0"',
                    "new_string": '__version__ = "0.2.0"',
                },
            },
        ],
        domain="filesystem",
        complexity=2,
    ),
    TaskTemplate(
        description="Explore then read",
        system_prompt=(
            "You are an expert CLI assistant. Explore the directory structure "
            "before diving into specific files."
        ),
        user_request="List the docs directory and show me what's in the README",
        steps=[
            {
                "tool_name": "list_files",
                "arguments": {"path": "docs"},
            },
            {
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
            },
        ],
        domain="filesystem",
        complexity=2,
    ),
    TaskTemplate(
        description="Search then read matching file",
        system_prompt=(
            "You are an expert CLI assistant. Find files first, then read the "
            "most relevant one to answer the user's question."
        ),
        user_request="Find all JSON config files and read the settings file",
        steps=[
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.json"},
            },
            {
                "tool_name": "read_file",
                "arguments": {"path": "config/settings.json"},
            },
        ],
        domain="filesystem",
        complexity=2,
    ),

    # =====================================================================
    # File System — Three+ Tools (complexity 3)
    # =====================================================================
    TaskTemplate(
        description="Explore, search, and read",
        system_prompt=(
            "You are an expert CLI assistant. Build a mental model of the project "
            "structure before making changes. List, search, read in sequence."
        ),
        user_request="Find all Python files, list the src directory, and read the main entry point",
        steps=[
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.py"},
            },
            {
                "tool_name": "list_files",
                "arguments": {"path": "src"},
            },
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/main.py"},
            },
        ],
        domain="filesystem",
        complexity=3,
    ),
    TaskTemplate(
        description="Create a new file with parent directories",
        system_prompt=(
            "You are an expert CLI assistant. Use write_file to create new files. "
            "It automatically creates parent directories."
        ),
        user_request="Create a new test file at tests/test_api.py with a basic test class",
        steps=[
            {
                "tool_name": "write_file",
                "arguments": {
                    "path": "tests/test_api.py",
                    "content": '"""API tests."""\n\nimport unittest\n\n\nclass TestAPI(unittest.TestCase):\n    def test_health(self):\n        self.assertTrue(True)\n',
                },
            },
        ],
        domain="filesystem",
        complexity=1,
    ),

    # =====================================================================
    # Git Operations
    # =====================================================================
    TaskTemplate(
        description="Check git status",
        system_prompt=(
            "You are an expert CLI assistant. Use git_status to check the current "
            "state of the repository before any git operations."
        ),
        user_request="What's the current status of my git repository?",
        steps=[
            {
                "tool_name": "git_status",
                "arguments": {},
            },
        ],
        domain="git",
        complexity=1,
    ),
    TaskTemplate(
        description="Check git log",
        system_prompt=(
            "You are an expert CLI assistant. Use git_log to review recent commits "
            "before making changes or writing commit messages."
        ),
        user_request="Show me the last 5 commits on this branch",
        steps=[
            {
                "tool_name": "git_log",
                "arguments": {"limit": 5},
            },
        ],
        domain="git",
        complexity=1,
    ),
    TaskTemplate(
        description="Status then commit",
        system_prompt=(
            "You are an expert CLI assistant. Always check git_status before "
            "committing to understand what will be included."
        ),
        user_request="Check the status and commit the changes with a message about fixing the bug",
        steps=[
            {
                "tool_name": "git_status",
                "arguments": {},
            },
            {
                "tool_name": "git_commit",
                "arguments": {
                    "message": "Fix bug in user authentication",
                },
            },
        ],
        domain="git",
        complexity=2,
    ),
    TaskTemplate(
        description="Check status, stash, and pull",
        system_prompt=(
            "You are an expert CLI assistant. Stash local changes before pulling "
            "to avoid merge conflicts."
        ),
        user_request="Stash my current changes and pull the latest from origin",
        steps=[
            {
                "tool_name": "git_status",
                "arguments": {},
            },
            {
                "tool_name": "git_stash",
                "arguments": {"message": "WIP: local changes"},
            },
            {
                "tool_name": "git_pull",
                "arguments": {"repo": "origin"},
            },
        ],
        domain="git",
        complexity=3,
    ),
    TaskTemplate(
        description="Log, status, and push",
        system_prompt=(
            "You are an expert CLI assistant. Verify your work with git_log "
            "before pushing to ensure you're sending the right commits."
        ),
        user_request="Review recent commits and push to the main branch",
        steps=[
            {
                "tool_name": "git_log",
                "arguments": {"limit": 3},
            },
            {
                "tool_name": "git_status",
                "arguments": {},
            },
            {
                "tool_name": "git_push",
                "arguments": {"repo": "origin", "branch": "main"},
            },
        ],
        domain="git",
        complexity=3,
    ),

    # =====================================================================
    # Bash Commands
    # =====================================================================
    TaskTemplate(
        description="Run pytest via bash",
        system_prompt=(
            "You are an expert CLI assistant. Use bash to run commands like "
            "pytest, npm, pip, and other shell utilities."
        ),
        user_request="Run the test suite with pytest",
        steps=[
            {
                "tool_name": "bash",
                "arguments": {"command": "pytest -v", "timeout": 60},
            },
        ],
        domain="execution",
        complexity=1,
    ),
    TaskTemplate(
        description="Install dependencies",
        system_prompt=(
            "You are an expert CLI assistant. Use bash to install project "
            "dependencies before running code."
        ),
        user_request="Install the Python dependencies from requirements.txt",
        steps=[
            {
                "tool_name": "bash",
                "arguments": {"command": "pip install -r requirements.txt", "timeout": 120},
            },
        ],
        domain="execution",
        complexity=1,
    ),
    TaskTemplate(
        description="Run bash then pytest",
        system_prompt=(
            "You are an expert CLI assistant. Navigate to the project "
            "directory and run tests."
        ),
        user_request="Navigate to the project directory and run pytest",
        steps=[
            {
                "tool_name": "bash",
                "arguments": {"command": "cd /home/user/projects/myproject && pytest", "timeout": 60},
            },
        ],
        domain="execution",
        complexity=1,
    ),

    # =====================================================================
    # Code Execution
    # =====================================================================
    TaskTemplate(
        description="Execute a Python script",
        system_prompt=(
            "You are an expert CLI assistant. Use python_exec to run Python files. "
            "It captures stdout and stderr for feedback."
        ),
        user_request="Run the main.py script to see the output",
        steps=[
            {
                "tool_name": "python_exec",
                "arguments": {"file": "src/main.py"},
            },
        ],
        domain="execution",
        complexity=1,
    ),
    TaskTemplate(
        description="Read file, execute, then run tests",
        system_prompt=(
            "You are an expert CLI assistant. Read source files to understand "
            "them, then execute and test."
        ),
        user_request="Show me the main.py file, run it, and then run the test suite",
        steps=[
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/main.py"},
            },
            {
                "tool_name": "python_exec",
                "arguments": {"file": "src/main.py"},
            },
            {
                "tool_name": "bash",
                "arguments": {"command": "pytest", "timeout": 60},
            },
        ],
        domain="execution",
        complexity=3,
    ),

    # =====================================================================
    # Web Search & Fetch
    # =====================================================================
    TaskTemplate(
        description="Search the web",
        system_prompt=(
            "You are an expert CLI assistant. Use web_search to find information "
            "before fetching specific pages."
        ),
        user_request="Search for the latest Python best practices for 2024",
        steps=[
            {
                "tool_name": "web_search",
                "arguments": {"query": "Python best practices 2024"},
            },
        ],
        domain="web",
        complexity=1,
    ),
    TaskTemplate(
        description="Search then fetch URL",
        system_prompt=(
            "You are an expert CLI assistant. Search for relevant URLs first, "
            "then fetch the most useful one for detailed information."
        ),
        user_request="Search for TypeScript documentation and then fetch the official docs",
        steps=[
            {
                "tool_name": "web_search",
                "arguments": {"query": "TypeScript handbook official documentation"},
            },
            {
                "tool_name": "fetch_url",
                "arguments": {"url": "https://www.typescriptlang.org/docs/"},
            },
        ],
        domain="web",
        complexity=2,
    ),

    # =====================================================================
    # Multi-Tool Complex Tasks
    # =====================================================================
    TaskTemplate(
        description="Full project exploration: list, search, read, update",
        system_prompt=(
            "You are an expert CLI assistant. Thoroughly explore a project before "
            "making changes. Understand the structure, find relevant files, "
            "read them, then make targeted edits."
        ),
        user_request=(
            "Explore this project: list the top-level structure, find all Python files, "
            "read the main.py file, and update the version string"
        ),
        steps=[
            {
                "tool_name": "list_files",
                "arguments": {"path": "."},
            },
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.py"},
            },
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/main.py"},
            },
            {
                "tool_name": "update_file",
                "arguments": {
                    "path": "src/__init__.py",
                    "old_string": '__version__ = "0.1.0"',
                    "new_string": '__version__ = "1.0.0"',
                },
            },
        ],
        domain="filesystem",
        complexity=4,
    ),
    TaskTemplate(
        description="Git workflow: status, log, commit, push",
        system_prompt=(
            "You are an expert CLI assistant. Follow proper git workflow: "
            "check status, review log, commit, then push."
        ),
        user_request="Check what changed, review the history, commit with a meaningful message, and push to origin",
        steps=[
            {
                "tool_name": "git_status",
                "arguments": {},
            },
            {
                "tool_name": "git_log",
                "arguments": {"limit": 3},
            },
            {
                "tool_name": "git_commit",
                "arguments": {"message": "Add new feature: user profile management"},
            },
            {
                "tool_name": "git_push",
                "arguments": {"repo": "origin"},
            },
        ],
        domain="git",
        complexity=4,
    ),

    # =====================================================================
    # Error Recovery Scenarios
    # =====================================================================
    TaskTemplate(
        description="Handle file not found gracefully",
        system_prompt=(
            "You are an expert CLI assistant. When a file doesn't exist, "
            "explore the directory to find the correct path."
        ),
        user_request="Read the app.py file from the root directory",
        steps=[
            {
                "tool_name": "read_file",
                "arguments": {"path": "app.py"},
            },
            {
                "tool_name": "list_files",
                "arguments": {"path": "."},
            },
            {
                "tool_name": "read_file",
                "arguments": {"path": "src/app.py"},
            },
        ],
        domain="filesystem",
        complexity=3,
    ),
    TaskTemplate(
        description="Search multiple patterns",
        system_prompt=(
            "You are an expert CLI assistant. Search with different patterns "
            "to find the right files."
        ),
        user_request="Find all Python and JavaScript files in the project",
        steps=[
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.py"},
            },
            {
                "tool_name": "search_files",
                "arguments": {"path": ".", "glob": "*.js"},
            },
        ],
        domain="filesystem",
        complexity=2,
    ),

    # =====================================================================
    # Utility & Environment
    # =====================================================================
    TaskTemplate(
        description="Get environment information",
        system_prompt=(
            "You are an expert CLI assistant. Use utility tools to gather "
            "system information when needed."
        ),
        user_request="What is the current timestamp and what's in the PATH environment variable?",
        steps=[
            {
                "tool_name": "get_timestamp",
                "arguments": {},
            },
            {
                "tool_name": "env_get",
                "arguments": {"name": "PATH"},
            },
        ],
        domain="utility",
        complexity=2,
    ),
    TaskTemplate(
        description="Check environment setup",
        system_prompt=(
            "You are an expert CLI assistant. Verify the environment is set up "
            "correctly before proceeding with tasks."
        ),
        user_request="Check the HOME directory and the current user from environment variables",
        steps=[
            {
                "tool_name": "env_get",
                "arguments": {"name": "HOME"},
            },
            {
                "tool_name": "env_get",
                "arguments": {"name": "USER"},
            },
            {
                "tool_name": "get_timestamp",
                "arguments": {},
            },
        ],
        domain="utility",
        complexity=3,
    ),
]


# --------------------------------------------------------------------------
# Dataset Creator
# --------------------------------------------------------------------------

class DatasetCreator:
    """
    Generates diverse training samples following REQUIREMENTS.md schema.

    Usage:
        creator = DatasetCreator(seed=42)
        samples = creator.generate(n=100)

        for sample in samples:
            print(sample.to_jsonl())

    Research basis:
    - Template diversity: Magicoder OSS-Instruct showed diverse instructions
      are key to generalization
    - Tool chaining: Anthropic's tool_use research shows multi-step reasoning
      significantly improves downstream performance
    - Error recovery: SWE-agent showed that realistic error handling in
      training data improves real-world agent robustness
    """

    def __init__(
        self,
        seed: int = 42,
        min_tool_calls: int = 1,
        max_tool_calls: int = 5,
        domains: Optional[list[str]] = None,
    ):
        """
        Initialize the dataset creator.

        Args:
            seed: Random seed for reproducible generation
            min_tool_calls: Minimum tool calls per sample (filter templates)
            max_tool_calls: Maximum tool calls per sample (filter templates)
            domains: Restrict to specific domains (e.g. ["filesystem", "git"])
                    None = all domains
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.min_tool_calls = min_tool_calls
        self.max_tool_calls = max_tool_calls

        # Filter templates by tool call count and domain
        self.templates = [
            t for t in TEMPLATES
            if min_tool_calls <= len(t.steps) <= max_tool_calls
            and (domains is None or t.domain in domains)
        ]
        if not self.templates:
            # Fallback: use all templates
            self.templates = TEMPLATES

    def _seed_from_index(self, index: int) -> int:
        """Derive a reproducible seed from index."""
        base = self.seed * 31337 + index * 7
        return base % (2**31)

    def generate_one(self, index: int = 0) -> TrainingSample:
        """
        Generate a single training sample.

        Args:
            index: Sample index (used for reproducible seeding)

        Returns:
            TrainingSample with full conversation
        """
        seed = self._seed_from_index(index)
        rng = random.Random(seed)
        fs_seed = seed + 12345
        fs = SimulatedFilesystem(seed=fs_seed)

        # Pick a random template
        template = rng.choice(self.templates)

        # Build sample
        sample = TrainingSample()
        sample.add_system(template.system_prompt)
        sample.add_user(template.user_request)

        # Execute each step and build conversation
        prev_result_content = ""

        for i, step in enumerate(template.steps):
            tool_name = step["tool_name"]
            arguments = step["arguments"]

            # Vary arguments slightly for diversity (substitute paths/names)
            args = self._vary_arguments(arguments, rng, fs)

            # Create tool call
            call_id = f"call_{hashlib.md5(f'{seed}{i}{tool_name}'.encode()).hexdigest()[:8]}"
            tool_call = ToolCall(
                id=call_id,
                tool_name=tool_name,
                arguments=args,
            )
            sample.add_tool_call(tool_call)

            # Simulate tool result
            step_seed = seed + i * 111
            tool_result = simulate_tool(tool_name, args, seed=step_seed, filesystem=fs)
            tool_result = ToolResult(
                tool_call_id=call_id,
                output=tool_result.output,
                exit_code=tool_result.exit_code,
                error=tool_result.error,
            )
            sample.add_tool_result(tool_result)
            prev_result_content = tool_result.output

        # Generate final answer based on tool results
        final_answer = self._generate_final_answer(template, prev_result_content, rng)
        sample.add_final_answer(FinalAnswer(content=final_answer))

        return sample

    def _vary_arguments(
        self,
        arguments: dict,
        rng: random.Random,
        fs: SimulatedFilesystem,
    ) -> dict:
        """
        Vary template arguments slightly for diversity.

        For example, substitute file paths with simulated filesystem paths.
        """
        varied = dict(arguments)

        # Substitute path values with realistic simulated paths
        path_fields = ["path", "file", "repo"]
        for field in path_fields:
            if field in varied and isinstance(varied[field], str):
                val = varied[field]
                # Substitute generic paths with simulated ones
                if val == "src/main.py":
                    varied[field] = rng.choice([
                        "src/main.py",
                        "src/entry.py",
                        "src/app.py",
                    ])
                elif val == "src/__init__.py":
                    varied[field] = rng.choice([
                        "src/__init__.py",
                        "package/__init__.py",
                    ])
                elif val == "README.md":
                    varied[field] = rng.choice([
                        "README.md",
                        "docs/README.md",
                    ])
                elif val == "config/settings.json":
                    varied[field] = rng.choice([
                        "config/settings.json",
                        "config/config.json",
                    ])
                elif val == ".":
                    varied[field] = rng.choice([".", "./", "src"])

        return varied

    def _generate_final_answer(
        self,
        template: TaskTemplate,
        last_output: str,
        rng: random.Random,
    ) -> str:
        """
        Generate a realistic final answer based on the task and tool results.

        Based on Anthropic's final_answer research: the final answer should
        summarize what was accomplished using information from tool results.
        """
        step_count = len(template.steps)
        last_output_preview = last_output[:100].replace("\n", " ") if last_output else ""

        templates = {
            1: [
                f"Here is the file content:\n\n{last_output_preview}...",
                f"Done. The output is:\n{last_output_preview}",
                f"The file contains:\n{last_output_preview}",
            ],
            2: [
                f"I found the following information:\n\n{last_output_preview}",
                f"Based on the files I checked:\n{last_output_preview}",
                f"The results:\n{last_output_preview}",
            ],
            3: [
                f"Here's what I found after exploring the project:\n\n{last_output_preview}",
                f"I've gathered the information you requested:\n{last_output_preview}",
                f"After examining the codebase:\n{last_output_preview}",
            ],
        }

        pool = templates.get(min(step_count, 3), templates[3])
        return rng.choice(pool)

    def generate(self, n: int = 100) -> list[TrainingSample]:
        """
        Generate n training samples.

        Args:
            n: Number of samples to generate

        Returns:
            List of TrainingSample objects
        """
        samples = []
        for i in range(n):
            sample = self.generate_one(index=i)
            samples.append(sample)
        return samples

    def generate_by_domain(self, n: int = 20) -> dict[str, list[TrainingSample]]:
        """
        Generate n samples per domain.

        Returns:
            Dict mapping domain name to list of samples
        """
        domains = list({t.domain for t in TEMPLATES})
        result = {}
        for domain in domains:
            creator = DatasetCreator(
                seed=self.seed,
                min_tool_calls=self.min_tool_calls,
                max_tool_calls=self.max_tool_calls,
                domains=[domain],
            )
            result[domain] = creator.generate(n=n)
        return result

    def template_stats(self) -> dict:
        """Return statistics about available templates."""
        by_domain = {}
        by_complexity = {}
        for t in TEMPLATES:
            by_domain[t.domain] = by_domain.get(t.domain, 0) + 1
            by_complexity[len(t.steps)] = by_complexity.get(len(t.steps), 0) + 1

        return {
            "total_templates": len(TEMPLATES),
            "by_domain": by_domain,
            "by_tool_call_count": by_complexity,
            "domains": list(by_domain.keys()),
        }
