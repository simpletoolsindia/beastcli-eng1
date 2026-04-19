"""
tools.py — Tool Registry and Output Simulators
==============================================

Tool Definitions
---------------
Each tool is defined with:
- Anthropic-style input_schema (JSON Schema format)
- Realistic output simulator
- Category and tags for filtering

Output Simulation Strategy
-------------------------
Simulators return realistic outputs based on the tool and arguments.
This is NOT real execution — it generates training data that trains
the model to call tools with correct arguments.

Key insight from Anthropic Cookbooks:
- Tool descriptions should be specific and include what the tool returns
- Argument descriptions guide the model on when/how to use each arg
- Output should match what a real tool would return

Safety Notes
------------
- Simulators never access the filesystem
- Network tools return mock data, never make real requests
- Dangerous operations (rm, chmod) return error outputs for safety
"""

from __future__ import annotations

import os
import random
import subprocess
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Optional

from dataset_creator.schemas import Tool, ToolArg, ToolCall, ToolResult


# --------------------------------------------------------------------------
# Tool Registry
# --------------------------------------------------------------------------

TOOL_REGISTRY: list[Tool] = [
    # ---- File System Tools (REQUIREMENTS.md section 5.1) ----
    Tool(
        name="read_file",
        description=(
            "Read the contents of a file. Returns the file's text content. "
            "Use offset and limit for large files to paginate."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Absolute or relative path to the file", required=True),
            ToolArg(name="offset", description="Starting line number (0-indexed)", required=False, default=0),
            ToolArg(name="limit", description="Maximum number of lines to read", required=False, default=100),
        ],
        tags=["read-only"],
    ),
    Tool(
        name="write_file",
        description=(
            "Create a new file or overwrite an existing file with the given content. "
            "Creates parent directories if they don't exist."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Path to the file to write", required=True),
            ToolArg(name="content", description="Text content to write to the file", required=True),
        ],
        tags=["write"],
    ),
    Tool(
        name="update_file",
        description=(
            "Edit a file by replacing a specific substring (old_string) with new content. "
            "The old_string must match exactly. For line-based edits, include the full line."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Path to the file to edit", required=True),
            ToolArg(
                name="old_string",
                description="The exact text to find and replace. Must match content in the file.",
                required=True,
            ),
            ToolArg(name="new_string", description="The replacement text", required=True),
        ],
        tags=["write"],
    ),
    Tool(
        name="delete_file",
        description=(
            "Delete a file from the filesystem. Cannot delete directories. "
            "Returns an error if the file doesn't exist."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Path to the file to delete", required=True),
        ],
        tags=["write", "dangerous"],
    ),
    Tool(
        name="list_files",
        description=(
            "List files and directories in a given path. Returns entries with names. "
            "Use pagination (limit/offset) for large directories."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Directory path to list", required=True),
            ToolArg(name="limit", description="Maximum number of entries to return", required=False, default=50),
            ToolArg(name="offset", description="Starting offset for pagination", required=False, default=0),
        ],
        tags=["read-only"],
    ),
    Tool(
        name="search_files",
        description=(
            "Search for files matching a content pattern (regex) or glob pattern. "
            "Returns matching file paths. Use limit to control results."
        ),
        category="filesystem",
        args=[
            ToolArg(name="path", description="Root directory to search in", required=True),
            ToolArg(name="pattern", description="Regex pattern to search for in file contents", required=False),
            ToolArg(name="glob", description="Glob pattern for file names (e.g., '*.py')", required=False),
            ToolArg(name="limit", description="Maximum number of results", required=False, default=100),
        ],
        tags=["read-only"],
    ),

    # ---- Bash Execution Tool (REQUIREMENTS.md section 5.2) ----
    Tool(
        name="bash",
        description=(
            "Execute a bash shell command. Returns stdout, stderr, and exit code. "
            "Timeout kills the process after the specified seconds. "
            "Never run interactive commands (vim, less) — use specific flags instead."
        ),
        category="execution",
        args=[
            ToolArg(name="command", description="The bash command to execute", required=True),
            ToolArg(name="timeout", description="Timeout in seconds (default: 30)", required=False, default=30),
            ToolArg(name="cwd", description="Working directory for the command", required=False),
        ],
        tags=["execution", "network"],
    ),

    # ---- Code Execution Tools (REQUIREMENTS.md section 5.3) ----
    Tool(
        name="python_exec",
        description=(
            "Execute a Python file using the system Python interpreter. "
            "Returns stdout (print output) and stderr (errors). "
            "The file must exist and be readable."
        ),
        category="execution",
        args=[
            ToolArg(name="file", description="Path to the Python file to execute", required=True),
            ToolArg(name="args", description="Command-line arguments to pass to the script", required=False),
        ],
        tags=["execution"],
    ),
    Tool(
        name="node_exec",
        description=(
            "Execute a Node.js JavaScript file. "
            "Returns stdout and stderr. "
            "Supports CommonJS modules."
        ),
        category="execution",
        args=[
            ToolArg(name="file", description="Path to the JavaScript file to execute", required=True),
            ToolArg(name="args", description="Command-line arguments", required=False),
        ],
        tags=["execution"],
    ),

    # ---- Git Tools (REQUIREMENTS.md section 5.4) ----
    Tool(
        name="git_pull",
        description=(
            "Pull changes from a remote repository. "
            "Defaults to origin remote and current branch. "
            "Returns a summary of changes pulled."
        ),
        category="git",
        args=[
            ToolArg(name="repo", description="Remote name (e.g., 'origin')", required=False, default="origin"),
            ToolArg(name="branch", description="Branch name to pull", required=False),
        ],
        tags=["read-only"],
    ),
    Tool(
        name="git_push",
        description=(
            "Push commits to a remote repository. "
            "Defaults to origin remote and current branch. "
            "May require authentication."
        ),
        category="git",
        args=[
            ToolArg(name="repo", description="Remote name (e.g., 'origin')", required=False, default="origin"),
            ToolArg(name="branch", description="Branch name to push", required=False),
        ],
        tags=["network"],
    ),
    Tool(
        name="git_commit",
        description=(
            "Commit staged changes with a message. "
            "Automatically stages all tracked files with changes. "
            "Returns the commit hash and summary."
        ),
        category="git",
        args=[
            ToolArg(name="message", description="Commit message", required=True),
            ToolArg(name="files", description="Specific files to commit", required=False),
        ],
        tags=["write"],
    ),
    Tool(
        name="git_stash",
        description=(
            "Stash uncommitted changes temporarily. "
            "Use git stash pop to restore. "
            "Returns stash reference and count."
        ),
        category="git",
        args=[
            ToolArg(name="message", description="Optional stash message", required=False),
        ],
        tags=["write"],
    ),
    Tool(
        name="git_status",
        description=(
            "Show the working tree status: modified files, staged changes, untracked files. "
            "Returns a summary of current repository state."
        ),
        category="git",
        args=[],
        tags=["read-only"],
    ),
    Tool(
        name="git_log",
        description=(
            "Show recent commit history. "
            "Returns commit hashes, authors, dates, and messages."
        ),
        category="git",
        args=[
            ToolArg(name="limit", description="Number of commits to show", required=False, default=10),
        ],
        tags=["read-only"],
    ),

    # ---- Web Tools (REQUIREMENTS.md section 5.5) ----
    Tool(
        name="web_search",
        description=(
            "Search the web for information. "
            "Returns search results with titles, URLs, and snippets. "
            "Use specific queries for best results."
        ),
        category="web",
        args=[
            ToolArg(name="query", description="The search query", required=True),
        ],
        tags=["network"],
    ),
    Tool(
        name="fetch_url",
        description=(
            "Fetch the content of a URL. Returns the page as markdown/text. "
            "Follows redirects. Limited to publicly accessible URLs. "
            "Returns error for paywalled or authenticated content."
        ),
        category="web",
        args=[
            ToolArg(name="url", description="The URL to fetch", required=True),
        ],
        tags=["network"],
    ),

    # ---- Utility Tools ----
    Tool(
        name="get_timestamp",
        description="Get the current Unix timestamp and ISO-formatted datetime.",
        category="utility",
        args=[],
        tags=["read-only"],
    ),
    Tool(
        name="env_get",
        description="Get the value of an environment variable.",
        category="utility",
        args=[
            ToolArg(name="name", description="Environment variable name", required=True),
        ],
        tags=["read-only"],
    ),
]


TOOL_REGISTRY_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOL_REGISTRY}
TOOL_NAMES: list[str] = list(TOOL_REGISTRY_BY_NAME.keys())


def get_tool(name: str) -> Optional[Tool]:
    """Look up a tool by name."""
    return TOOL_REGISTRY_BY_NAME.get(name)


# --------------------------------------------------------------------------
# Realistic Output Simulators
# --------------------------------------------------------------------------

class SimulatedFilesystem:
    """
    Simulates a realistic project filesystem for training data generation.

    Generates plausible file trees, file contents, and directory listings
    based on common project types (Python, Node.js, Go, Rust, generic).
    """

    PROJECT_TYPES = ["python", "nodejs", "go", "rust", "generic"]

    PYTHON_STRUCTURE = {
        "src/": ["__init__.py", "main.py", "utils.py", "models.py", "api.py"],
        "tests/": ["test_main.py", "test_utils.py"],
        "docs/": ["README.md", "API.md"],
        "config/": ["settings.json", "pyproject.toml"],
        "scripts/": ["setup.sh", "deploy.py"],
    }

    NODEJS_STRUCTURE = {
        "src/": ["index.js", "server.js", "routes.js", "utils.js"],
        "tests/": ["index.test.js", "server.test.js"],
        "public/": ["index.html", "style.css"],
        "docs/": ["README.md"],
        "scripts/": ["build.sh"],
    }

    GENERIC_STRUCTURE = {
        "src/": ["main.c", "utils.c", "types.h"],
        "tests/": ["test_main.c"],
        "docs/": ["README.md", "SPEC.md"],
    }

    STRUCTURES = {
        "python": PYTHON_STRUCTURE,
        "nodejs": NODEJS_STRUCTURE,
        "generic": GENERIC_STRUCTURE,
    }

    def __init__(self, project_type: str = "python", seed: int = 42):
        self.project_type = project_type
        self.rng = random.Random(seed)
        self.structure = self.STRUCTURES.get(project_type, self.GENERIC_STRUCTURE)
        self.project_name = self.rng.choice([
            "myproject", "api-server", "cli-tool", "webapp", "libutils"
        ])

    def list_files(self, path: str, limit: int = 50, offset: int = 0) -> dict:
        """Simulate list_files output."""
        all_entries = []
        # Add files from structure
        for dir_path, files in self.structure.items():
            if path == "." or path == "./":
                all_entries.append({"name": dir_path.rstrip("/"), "type": "dir"})
            for f in files:
                if path == "." or path == "./":
                    all_entries.append({"name": f, "type": "file"})
                elif dir_path.rstrip("/") == path.lstrip("./"):
                    all_entries.append({"name": f, "type": "file"})

        # Add some random extra files
        extra = self.rng.randint(0, 5)
        for i in range(extra):
            ext = self.rng.choice([".txt", ".log", ".tmp", ".md"])
            all_entries.append({"name": f"file_{i}{ext}", "type": "file"})

        total = len(all_entries)
        page = all_entries[offset:offset + limit]
        return {
            "entries": page,
            "total": total,
            "next_offset": offset + limit if offset + limit < total else None,
        }

    def read_file(self, path: str, offset: int = 0, limit: int = 100) -> dict:
        """Simulate read_file output."""
        content_map = {
            "src/main.py": f"# {self.project_name}\n\ndef main():\n    print('Hello from {self.project_name}!')\n\nif __name__ == '__main__':\n    main()\n",
            "src/__init__.py": f'"""{self.project_name} package."""\n__version__ = "0.1.0"\n',
            "README.md": f"# {self.project_name}\n\nA sample project.\n\n## Installation\n\npip install {self.project_name}\n",
            "src/utils.py": "def format_date(dt):\n    return dt.strftime('%Y-%m-%d')\n\ndef validate_email(email):\n    return '@' in email\n",
            "src/api.py": "from flask import Flask\napp = Flask(__name__)\n",
            "src/models.py": "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    name: str\n    email: str\n",
        }
        content = content_map.get(path, f"# {path}\n# File content placeholder\n")
        lines = content.split("\n")
        total_lines = len(lines)
        page_lines = lines[offset:offset + limit]
        return {
            "content": "\n".join(page_lines),
            "total_lines": total_lines,
            "offset": offset,
            "limit": limit,
            "truncated": total_lines > limit,
        }

    def search_files(self, path: str, pattern: Optional[str] = None,
                     glob: Optional[str] = None, limit: int = 100) -> dict:
        """Simulate search_files output."""
        results = []
        if glob:
            # Match by glob pattern
            for dir_path, files in self.structure.items():
                for f in files:
                    if self._glob_match(f, glob):
                        if path == "." or dir_path.lstrip("./") == path.lstrip("./"):
                            results.append(f"{dir_path}{f}")

        matches = len(results)
        results = results[:limit]
        return {
            "matches": results,
            "total_matches": matches,
            "pattern": pattern,
            "glob": glob,
            "truncated": matches > limit,
        }

    def _glob_match(self, filename: str, pattern: str) -> bool:
        """Simple glob matching."""
        pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        try:
            return bool(re.match(f"^{pattern}$", filename))
        except re.error:
            return pattern in filename


# --------------------------------------------------------------------------
# Tool Output Simulator
# --------------------------------------------------------------------------

def simulate_tool(
    tool_name: str,
    arguments: dict,
    seed: int = 42,
    filesystem: Optional[SimulatedFilesystem] = None,
    error_rate: float = 0.0,
) -> ToolResult:
    """
    Simulate the output of a tool call for training data generation.

    This is the core output simulation engine. It generates realistic,
    plausible tool outputs that train the model to recognize correct
    tool behavior — without accessing the real filesystem.

    Design principles:
    1. Output should be plausible for the given arguments
    2. Errors should be realistic (file not found, permission denied, etc.)
    3. Output should vary based on arguments (pagination, filtering, etc.)
    4. Never return harmful or misleading content

    Args:
        tool_name: Name of the tool being simulated
        arguments: Tool arguments from the ToolCall
        seed: Random seed for reproducible generation
        filesystem: Optional SimulatedFilesystem instance for file operations
        error_rate: Probability of returning an error result (0.0-1.0).
                   0.15 (15%) is optimal per AgentErrorBench research.

    Returns:
        ToolResult with realistic output and appropriate exit code

    Note on research: Based on OpenCodeInterpreter's execution feedback
    approach — the model learns from execution results. Our simulator
    provides analogous feedback without real execution.
    """
    rng = random.Random(seed + hash(tool_name) % 10000)
    fs = filesystem or SimulatedFilesystem(seed=seed)

    # AgentErrorBench: 15% error rate optimal for robust training
    # Only apply error simulation to read operations (files must exist to succeed)
    if error_rate > 0 and rng.random() < error_rate:
        if tool_name in ("read_file", "list_files", "search_files", "bash", "python_exec", "node_exec"):
            return _simulate_error(tool_name, arguments, rng)

    # Auto-generate call_id for result matching
    call_id = f"call_{hashlib.md5(f'{tool_name}{arguments}'.encode()).hexdigest()[:8]}"

    try:
        # --- File System Tools ---
        if tool_name == "list_files":
            path = arguments.get("path", ".")
            limit = arguments.get("limit", 50)
            offset = arguments.get("offset", 0)
            result = fs.list_files(path, limit=limit, offset=offset)
            output_lines = [f"{e['name']}" for e in result["entries"]]
            if result["next_offset"]:
                output_lines.append(f"... (showing {limit} of {result['total']} entries, offset={result['next_offset']})")
            elif result["total"] > limit:
                output_lines.append(f"(showing {result['total']} entries total)")
            return ToolResult(tool_call_id=call_id, output="\n".join(output_lines), exit_code=0)

        elif tool_name == "read_file":
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 100)
            if not path:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="path is required")
            result = fs.read_file(path, offset=offset, limit=limit)
            if result["truncated"]:
                output = f"{result['content']}\n[Showing lines {offset}-{offset+limit} of {result['total_lines']}]"
            else:
                output = result["content"]
            return ToolResult(tool_call_id=call_id, output=output, exit_code=0)

        elif tool_name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            if not path:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="path is required")
            lines = len(content.split("\n"))
            size = len(content)
            return ToolResult(
                tool_call_id=call_id,
                output=f"Wrote {lines} lines ({size} bytes) to {path}\n",
                exit_code=0,
            )

        elif tool_name == "update_file":
            path = arguments.get("path", "")
            old = arguments.get("old_string", "")
            new = arguments.get("new_string", "")
            if not path:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="path is required")
            if not old:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="old_string is required")
            return ToolResult(
                tool_call_id=call_id,
                output=f"Updated {path}: replaced {len(old)} chars with {len(new)} chars\n",
                exit_code=0,
            )

        elif tool_name == "delete_file":
            path = arguments.get("path", "")
            if not path:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="path is required")
            # Safety: simulate common "not found" for obviously fake paths
            if rng.random() < 0.1:
                return ToolResult(
                    tool_call_id=call_id,
                    output="",
                    exit_code=1,
                    error=f"rm: cannot remove '{path}': No such file or directory",
                )
            return ToolResult(tool_call_id=call_id, output=f"Deleted {path}\n", exit_code=0)

        elif tool_name == "search_files":
            path = arguments.get("path", ".")
            pattern = arguments.get("pattern")
            glob = arguments.get("glob")
            limit = arguments.get("limit", 100)
            result = fs.search_files(path, pattern=pattern, glob=glob, limit=limit)
            if result["matches"]:
                output = "\n".join(result["matches"][:limit])
                if result["truncated"]:
                    output += f"\n[Showing {limit} of {result['total_matches']} matches]"
                return ToolResult(tool_call_id=call_id, output=output, exit_code=0)
            return ToolResult(tool_call_id=call_id, output="No matches found.\n", exit_code=0)

        # --- Bash Execution Tool ---
        elif tool_name == "bash":
            cmd = arguments.get("command", "")
            timeout = arguments.get("timeout", 30)
            if not cmd:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="command is required")

            # Simulate common bash commands
            cmd_stripped = cmd.strip()

            # ls variants
            if cmd_stripped.startswith("ls"):
                entries = ["file1.txt", "file2.py", "README.md", "src/", "tests/"]
                if "-la" in cmd or "-l" in cmd:
                    lines = ["total 24", "drwxr-xr-x  4 user staff  128 Apr 19 10:00 src"]
                    for e in entries:
                        if e.endswith("/"):
                            lines.append(f"drwxr-xr-x  2 user staff   64 Apr 19 10:00 {e}")
                        else:
                            lines.append(f"-rw-r--r--  1 user staff  1024 Apr 19 10:00 {e}")
                    return ToolResult(tool_call_id=call_id, output="\n".join(lines), exit_code=0)
                return ToolResult(tool_call_id=call_id, output="\n".join(entries), exit_code=0)

            # pwd
            if cmd_stripped == "pwd":
                return ToolResult(tool_call_id=call_id, output=f"/home/user/projects/{fs.project_name}\n", exit_code=0)

            # echo
            if cmd_stripped.startswith("echo "):
                text = cmd_stripped[5:].strip('"').strip("'")
                return ToolResult(tool_call_id=call_id, output=f"{text}\n", exit_code=0)

            # git status
            if "git status" in cmd_stripped:
                return ToolResult(
                    tool_call_id=call_id,
                    output="On branch main\n"
                           "Changes not staged for commit:\n"
                           "  modified:   src/main.py\n"
                           "  modified:   README.md\n"
                           "Untracked files:\n"
                           "  new_feature.py\n",
                    exit_code=0,
                )

            # git commit
            if "git commit" in cmd_stripped:
                hash_str = hashlib.md5(cmd_stripped.encode()).hexdigest()[:7]
                return ToolResult(
                    tool_call_id=call_id,
                    output=f"[main {hash_str}] {arguments.get('message', 'update')}\n"
                           f" 2 files changed, 15 insertions(+)\n",
                    exit_code=0,
                )

            # git pull
            if "git pull" in cmd_stripped:
                return ToolResult(tool_call_id=call_id, output="Already up to date.\n", exit_code=0)

            # git push
            if "git push" in cmd_stripped:
                return ToolResult(tool_call_id=call_id, output="Everything up-to-date\n", exit_code=0)

            # git stash
            if "git stash" in cmd_stripped:
                return ToolResult(
                    tool_call_id=call_id,
                    output="Saved working directory and index state WIP on main: abc123 update README\n",
                    exit_code=0,
                )

            # git log
            if "git log" in cmd_stripped:
                commits = [
                    f"commit abc{rng.randint(1000,9999)}def",
                    f"Author: Developer <dev@example.com>",
                    f"Date:   {rng.choice(['Mon Apr 19', 'Sun Apr 18', 'Sat Apr 17'])} 10:30:00",
                    f"",
                    f"    {rng.choice(['Update main.py', 'Add new feature', 'Fix bug in utils', 'Refactor API'])}",
                    "",
                ]
                return ToolResult(tool_call_id=call_id, output="\n".join(commits[:5]), exit_code=0)

            # npm install / pip install
            if "npm install" in cmd_stripped:
                return ToolResult(
                    tool_call_id=call_id,
                    output=f"added {rng.randint(50, 500)} packages in {rng.randint(5, 30)}s\n",
                    exit_code=0,
                )
            if "pip install" in cmd_stripped:
                return ToolResult(
                    tool_call_id=call_id,
                    output=f"Successfully installed {rng.randint(1, 10)} packages\n",
                    exit_code=0,
                )

            # pytest
            if "pytest" in cmd_stripped:
                passed = rng.randint(5, 30)
                return ToolResult(
                    tool_call_id=call_id,
                    output=f"===== {passed} passed in {rng.randint(1, 10)}s =====\n",
                    exit_code=0,
                )

            # python / node execution
            if "python" in cmd_stripped or "node" in cmd_stripped:
                return ToolResult(tool_call_id=call_id, output="Hello, World!\n", exit_code=0)

            # cat
            if cmd_stripped.startswith("cat "):
                fname = cmd_stripped[4:].strip()
                content_map = {
                    "package.json": '{"name": "myproject", "version": "1.0.0", "scripts": {"start": "node src/index.js"}}',
                    "requirements.txt": "flask==2.0.0\nrequests==2.28.0\n",
                    ".env": "DATABASE_URL=postgres://localhost/mydb\nSECRET_KEY=dev-secret-key\n",
                }
                for k, v in content_map.items():
                    if k in fname:
                        return ToolResult(tool_call_id=call_id, output=v, exit_code=0)
                return ToolResult(tool_call_id=call_id, output=f"# {fname}\ncontent here\n", exit_code=0)

            # find
            if cmd_stripped.startswith("find "):
                found = rng.sample([f"src/module_{i}.py" for i in range(10)], k=rng.randint(2, 6))
                return ToolResult(tool_call_id=call_id, output="\n".join(found) + "\n", exit_code=0)

            # grep
            if "grep" in cmd_stripped:
                matches = rng.randint(0, 5)
                if matches:
                    lines = [f"src/file_{i}.py:{rng.randint(1,100)}: matching line content" for i in range(matches)]
                    return ToolResult(tool_call_id=call_id, output="\n".join(lines) + "\n", exit_code=0)
                return ToolResult(tool_call_id=call_id, output="", exit_code=1)

            # curl
            if "curl" in cmd_stripped:
                if rng.random() < 0.8:
                    return ToolResult(
                        tool_call_id=call_id,
                        output='{"status": "ok", "message": "success"}\n',
                        exit_code=0,
                    )
                return ToolResult(
                    tool_call_id=call_id,
                    output="",
                    exit_code=22,
                    error="curl: (22) The requested URL returned error: 404",
                )

            # mkdir
            if cmd_stripped.startswith("mkdir "):
                return ToolResult(tool_call_id=call_id, output="", exit_code=0)

            # chmod / chown (simulate as potentially restricted)
            if "chmod 777" in cmd_stripped or "chmod -R 777" in cmd_stripped:
                return ToolResult(
                    tool_call_id=call_id,
                    output="",
                    exit_code=1,
                    error="chmod: Permission denied (are you root?)",
                )

            # Unknown command — simulate plausible output
            return ToolResult(
                tool_call_id=call_id,
                output=f"{cmd_stripped[:50]}: command executed\n",
                exit_code=0,
            )

        # --- Code Execution ---
        elif tool_name == "python_exec":
            file = arguments.get("file", "")
            if not file:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="file is required")
            if not file.endswith(".py"):
                return ToolResult(
                    tool_call_id=call_id,
                    output="",
                    exit_code=1,
                    error=f"python: can't open file '{file}': No such file or directory",
                )
            return ToolResult(tool_call_id=call_id, output="Hello, World!\n", exit_code=0)

        elif tool_name == "node_exec":
            file = arguments.get("file", "")
            if not file:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="file is required")
            if not (file.endswith(".js") or file.endswith(".mjs")):
                return ToolResult(
                    tool_call_id=call_id,
                    output="",
                    exit_code=1,
                    error=f"node: Cannot open file '{file}': No such file or directory",
                )
            return ToolResult(tool_call_id=call_id, output="Hello, World!\n", exit_code=0)

        # --- Git Tools ---
        elif tool_name == "git_status":
            return ToolResult(
                tool_call_id=call_id,
                output="On branch main\n"
                       "Changes not staged for commit:\n"
                       "  modified:   src/main.py\n"
                       "Untracked files:\n"
                       "  (use \"git add <file>\" to include in what will be committed)\n"
                       "\tnew_feature.py\n",
                exit_code=0,
            )

        elif tool_name == "git_log":
            limit = arguments.get("limit", 10)
            lines = []
            for i in range(min(limit, 5)):
                lines.append(f"commit abc{1000+i}def")
                lines.append(f"Author: Dev <dev@example.com>")
                lines.append(f"Date:   {rng.choice(['Mon', 'Tue', 'Wed'])} Apr {rng.randint(10,19)} 10:30:00")
                lines.append(f"")
                lines.append(f"    {rng.choice(['Update', 'Fix', 'Add', 'Refactor'])} {rng.choice(['main', 'utils', 'api', 'tests'])}")
                lines.append("")
            return ToolResult(tool_call_id=call_id, output="\n".join(lines), exit_code=0)

        elif tool_name == "git_pull":
            return ToolResult(
                tool_call_id=call_id,
                output=f"Already up to date.\n",
                exit_code=0,
            )

        elif tool_name == "git_push":
            return ToolResult(
                tool_call_id=call_id,
                output="Everything up-to-date\n",
                exit_code=0,
            )

        elif tool_name == "git_commit":
            msg = arguments.get("message", "update")
            hash_str = hashlib.md5(msg.encode()).hexdigest()[:7]
            return ToolResult(
                tool_call_id=call_id,
                output=f"[main {hash_str}] {msg}\n 1 file changed, 5 insertions(+)\n",
                exit_code=0,
            )

        elif tool_name == "git_stash":
            msg = arguments.get("message", "")
            prefix = f"stash@{0}: WIP on main: " if not msg else f"stash@{0}: On main: {msg}"
            return ToolResult(
                tool_call_id=call_id,
                output=f"{prefix}\n 2 files changed, 10 insertions(+)\n",
                exit_code=0,
            )

        # --- Web Tools ---
        elif tool_name == "web_search":
            query = arguments.get("query", "")
            if not query:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="query is required")
            results = rng.sample(
                [
                    ("https://example.com/article-1", "Example Article 1", "A relevant article about " + query),
                    ("https://example.com/article-2", "Example Article 2", "Another result for " + query),
                    ("https://example.com/docs", "Documentation", "Official documentation for " + query),
                ],
                k=min(3, rng.randint(1, 3)),
            )
            output_lines = [f"Results for: {query}", ""]
            for i, (url, title, snippet) in enumerate(results, 1):
                output_lines.append(f"{i}. {title}")
                output_lines.append(f"   {url}")
                output_lines.append(f"   {snippet[:100]}...")
                output_lines.append("")
            return ToolResult(tool_call_id=call_id, output="\n".join(output_lines), exit_code=0)

        elif tool_name == "fetch_url":
            url = arguments.get("url", "")
            if not url:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="url is required")
            if "github.com" in url:
                return ToolResult(
                    tool_call_id=call_id,
                    output="# Repository\n\nRepository content for {url}\n\n## Files\n- README.md\n- src/\n- tests/\n",
                    exit_code=0,
                )
            return ToolResult(
                tool_call_id=call_id,
                output=f"# Fetched: {url}\n\nPage content would appear here.\n",
                exit_code=0,
            )

        # --- Utility Tools ---
        elif tool_name == "get_timestamp":
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            return ToolResult(
                tool_call_id=call_id,
                output=f"timestamp: {int(now.timestamp())}\ndatetime: {now.isoformat()}\n",
                exit_code=0,
            )

        elif tool_name == "env_get":
            name = arguments.get("name", "")
            if not name:
                return ToolResult(tool_call_id=call_id, output="", exit_code=1, error="name is required")
            env_map = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/home/user",
                "USER": "user",
                "SHELL": "/bin/bash",
                "LANG": "en_US.UTF-8",
            }
            value = env_map.get(name, "")
            if value:
                return ToolResult(tool_call_id=call_id, output=f"{name}={value}\n", exit_code=0)
            return ToolResult(
                tool_call_id=call_id,
                output="",
                exit_code=1,
                error=f"env_get: '{name}' is not set",
            )

        # --- Unknown tool ---
        return ToolResult(
            tool_call_id=call_id,
            output=f"Tool '{tool_name}' executed successfully\n",
            exit_code=0,
        )

    except Exception as e:
        return ToolResult(
            tool_call_id=call_id,
            output="",
            exit_code=1,
            error=f"{type(e).__name__}: {str(e)}",
        )
