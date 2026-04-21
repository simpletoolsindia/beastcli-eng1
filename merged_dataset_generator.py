"""
BeastCLI Dataset Generator — 5 Core Tools
Based on research: Terminus 2 (Terminal-Bench winner) uses 1 bash tool.
We use 5: Bash, File_Read, File_Write, Web_Search, Web_Fetch.
Everything else is bash (git, python3, node, npm, cargo, etc.)
"""

from __future__ import annotations

import json
import random
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    EN = "en"; HI = "hi"; ES = "es"; FR = "fr"; DE = "de"
    JA = "ja"; ZH = "zh"; PT = "pt"; IT = "it"; KO = "ko"
    AR = "ar"; RU = "ru"

class Tone(str, Enum):
    PROFESSIONAL = "professional"; CASUAL = "casual"
    TECHNICAL = "technical"; FRIENDLY = "friendly"

class DifficultyLevel(str, Enum):
    EASY = "easy"     # 30%
    MEDIUM = "medium" # 40%
    HARD = "hard"     # 20%
    EXPERT = "expert" # 10%

class FormalityLevel(str, Enum):
    FORMAL = "formal"; NEUTRAL = "neutral"; INFORMAL = "informal"


# ═══════════════════════════════════════════════════════════════════════════════
# LOCALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Localization:
    language: str = "en"
    tone: str = "professional"
    formality: str = "neutral"
    humanize: bool = True
    humanize_level: str = "medium"

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "tone": self.tone,
            "formality": self.formality,
            "humanize": self.humanize,
            "humanize_level": self.humanize_level
        }


class LocalizationContent:
    """Language-specific content for responses."""

    SUCCESS = {
        DifficultyLevel.EASY: {
            Language.EN: ["Done.", "Completed.", "Finished."],
            Language.HI: ["Ho gaya.", "Poora ho gaya."],
            Language.ES: ["Hecho.", "Completado."],
            Language.FR: ["Fait.", "Termine."],
            Language.DE: ["Erledigt.", "Abgeschlossen."],
            Language.JA: ["Yarimashita.", "Kanryou shimashita."],
            Language.ZH: ["Wancheng le.", "Haole."],
            Language.KO: ["Dohaetseumnida."],
            Language.AR: ["Ja'al.", "Tamma."],
            Language.RU: ["Gotovo.", "Sdelano."],
            Language.PT: ["Feito.", "Concluido."],
            Language.IT: ["Fatto.", "Completato."],
        },
        DifficultyLevel.MEDIUM: {
            Language.EN: ["Task completed successfully.", "All done."],
            Language.HI: ["Kaam safalta se pura hua."],
            Language.ES: ["Tarea completada exitosamente."],
            Language.FR: ["Tache terminee avec succes."],
            Language.DE: ["Aufgabe erfolgreich abgeschlossen."],
            Language.JA: ["Tsukkomi shimashita."],
            Language.ZH: ["Renwu wancheng."],
            Language.KO: ["Taekeu haesin dae haetseumnida."],
            Language.AR: ["Al'amr tamma bi alnajaha."],
            Language.RU: ["Zadacha vypolnena."],
            Language.PT: ["Tarefa concluida com sucesso."],
            Language.IT: ["Attivita completata con successo."],
        },
        DifficultyLevel.HARD: {
            Language.EN: ["Multi-step operation completed."],
            Language.HI: ["Sabhi stages safalta se complete huye."],
            Language.ES: ["Operacion de multiples pasos completada."],
            Language.FR: ["Operation multi-etapes terminee."],
            Language.DE: ["Mehrstufige Operation erfolgreich abgeschlossen."],
            Language.JA: ["Taju sodan no taipu wo kanryou shimashita."],
            Language.ZH: ["Duo bu zhi xingren wancheng."],
            Language.KO: ["Daecheseong bunpo peulogeulaemu ga jongryeokhaetseumnida."],
            Language.AR: ["Al'amaliyat al-mutaaddadat al-marhalat tammat."],
            Language.RU: ["Mnogootraslevaya operatsiya vypolnena."],
            Language.PT: ["Operacao multi-etapas concluida."],
            Language.IT: ["Operazione multi-passo completata."],
        },
    }

    ERROR = {
        Language.EN: ["I encountered an error: {e}", "Operation failed: {e}"],
        Language.HI: ["Ek error aaya: {e}", "Kaam nahi ho saka: {e}"],
        Language.ES: ["Encontre un error: {e}", "La operacion fallo: {e}"],
        Language.FR: ["J'ai rencontre une erreur: {e}"],
        Language.DE: ["Ein Fehler ist aufgetreten: {e}"],
        Language.JA: ["Eraa ni natte imasu: {e}"],
        Language.ZH: ["Cuowu fasheng: {e}"],
        Language.KO: ["Sileul mannasseumnida: {e}"],
        Language.AR: ["Waqaeat khata': {e}"],
        Language.RU: ["Poyavilas oshibka: {e}"],
        Language.PT: ["Encontrei um erro: {e}"],
        Language.IT: ["Ho incontrato un errore: {e}"],
    }

    @classmethod
    def get_success(cls, difficulty: DifficultyLevel, lang: str) -> str:
        l = Language(lang) if lang in [x.value for x in Language] else Language.EN
        msgs = cls.SUCCESS.get(difficulty, cls.SUCCESS[DifficultyLevel.EASY])
        lang_msgs = msgs.get(l, msgs[Language.EN])
        return random.choice(lang_msgs)

    @classmethod
    def get_error(cls, error: str, lang: str) -> str:
        l = Language(lang) if lang in [x.value for x in Language] else Language.EN
        errs = cls.ERROR.get(l, cls.ERROR[Language.EN])
        return random.choice(errs).format(e=error)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS (5 Tools Only)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolArgument:
    name: str; type: str; description: str
    required: bool = True; default: Any = None
    enum_values: list[str] | None = None

@dataclass
class ToolSchema:
    name: str; category: str; description: str
    arguments: list[ToolArgument]; returns: str

    def to_openai_format(self) -> dict:
        props = {}
        required = []
        for arg in self.arguments:
            prop = {"type": arg.type, "description": arg.description}
            if arg.enum_values:
                prop["enum"] = arg.enum_values
            props[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": required}
            }
        }


class ToolRegistry:
    """5 core tools. Everything else is bash."""

    TOOLS = [
        ToolSchema(
            name="Bash",
            category="bash",
            description="Execute a bash command. Covers: python3, node, npm, git, cargo, pip, docker, find, grep, ls, cat, echo, tee, chmod, and any CLI tool.",
            arguments=[
                ToolArgument("command", "string", "The bash command to execute (e.g., python3 script.py, git status, npm install)"),
                ToolArgument("timeout", "integer", "Max execution time in seconds", required=False, default=30),
                ToolArgument("working_directory", "string", "Directory to run command in", required=False),
            ],
            returns="stdout, stderr, exit_code",
        ),
        ToolSchema(
            name="File_Read",
            category="file",
            description="Read contents of a file.",
            arguments=[
                ToolArgument("file_path", "string", "Absolute or relative path to the file"),
                ToolArgument("offset", "integer", "Line number to start reading from", required=False, default=0),
                ToolArgument("limit", "integer", "Max number of lines to read", required=False),
            ],
            returns="File contents with metadata",
        ),
        ToolSchema(
            name="File_Write",
            category="file",
            description="Write or append content to a file. Creates file if it doesn't exist.",
            arguments=[
                ToolArgument("file_path", "string", "Path to the file"),
                ToolArgument("content", "string", "Content to write"),
                ToolArgument("append", "boolean", "Append instead of overwrite", required=False, default=False),
            ],
            returns="Success confirmation with path and bytes written",
        ),
        ToolSchema(
            name="File_Edit",
            category="file",
            description="Make a precise string replacement in a file. You MUST read the file first before editing.",
            arguments=[
                ToolArgument("file_path", "string", "Path to the file to edit"),
                ToolArgument("old_string", "string", "Exact text to find and replace. Must match the file content exactly."),
                ToolArgument("new_string", "string", "Replacement text"),
            ],
            returns="Number of replacements made",
        ),
        ToolSchema(
            name="Glob",
            category="file",
            description="Find files matching a glob pattern. Supports ** for recursive matching.",
            arguments=[
                ToolArgument("pattern", "string", "Glob pattern (e.g., **/*.py, src/*.js, **/config*)"),
                ToolArgument("path", "string", "Root directory to search in", required=False, default="."),
            ],
            returns="List of matching file paths",
        ),
        ToolSchema(
            name="Grep",
            category="search",
            description="Search file contents using regex. Returns matching lines with file path and line numbers.",
            arguments=[
                ToolArgument("pattern", "string", "Regex pattern to search for"),
                ToolArgument("path", "string", "Directory or file to search in", required=False, default="."),
                ToolArgument("file_types", "string", "File extensions to search (e.g., .py, .js)", required=False),
                ToolArgument("case_sensitive", "boolean", "Case sensitive search", required=False, default=True),
            ],
            returns="List of matches with file path, line number, and matching text",
        ),
        ToolSchema(
            name="TodoWrite",
            category="task",
            description="Create or update a todo item in the task tracking list.",
            arguments=[
                ToolArgument("todo", "string", "The todo item text to add or update"),
                ToolArgument("status", "string", "Status: pending, in_progress, completed", required=False, default="pending"),
            ],
            returns="Todo item created or updated",
        ),
        ToolSchema(
            name="TodoRead",
            category="task",
            description="Read the current todo list to track progress during complex tasks.",
            arguments=[],
            returns="List of all todo items with their status",
        ),
        ToolSchema(
            name="Web_Search",
            category="web",
            description="Search the web. Use specific keywords for best results.",
            arguments=[
                ToolArgument("query", "string", "Search query with specific keywords"),
                ToolArgument("num_results", "integer", "Number of results (1-20)", required=False, default=5),
            ],
            returns="List of search results with title, URL, and snippet",
        ),
        ToolSchema(
            name="Web_Fetch",
            category="web",
            description="Fetch content from a URL. Returns HTML/text content.",
            arguments=[
                ToolArgument("url", "string", "HTTP/HTTPS URL to fetch"),
                ToolArgument("prompt", "string", "Instructions for extracting specific info", required=False),
            ],
            returns="URL content, status code, content type, and length",
        ),
    ]

    # Bash covers these sub-categories via commands
    BASH_SUBCOMMANDS = {
        "git": ["git status", "git log --oneline -5", "git diff", "git add .", "git commit -m", "git push", "git pull", "git branch", "git checkout"],
        "npm": ["npm install", "npm run build", "npm test", "npm start"],
        "pip": ["pip install", "pip list", "pip freeze"],
        "cargo": ["cargo build", "cargo run", "cargo test"],
        "docker": ["docker ps", "docker images", "docker-compose up"],
        "system": ["ls -la", "find . -name", "ps aux", "df -h", "free -m", "pwd", "whoami", "uname -a"],
        "python": ["python3 -c", "python3 script.py", "pip3 install"],
        "node": ["node script.js", "npm install"],
        "code": ["grep -rn", "rg", "sed -i", "awk"],
    }

    @classmethod
    def get_all_tools(cls) -> list[ToolSchema]:
        return cls.TOOLS

    @classmethod
    def get_tool_names(cls) -> set[str]:
        return {t.name for t in cls.TOOLS}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None

@dataclass
class DatasetExample:
    messages: list[Message]
    localization: Localization
    tools: list[dict]
    metadata: dict

    def to_dict(self) -> dict:
        msgs = []
        for m in self.messages:
            if isinstance(m, Message):
                d = {"role": m.role, "content": m.content}
                if m.tool_call_id:
                    d["tool_call_id"] = m.tool_call_id
                if m.name:
                    d["name"] = m.name
                msgs.append(d)
            else:
                msgs.append(m)
        return {
            "messages": msgs,
            "localization": self.localization.to_dict(),
            "tools": self.tools,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

class SystemPromptGenerator:
    TOOL_DESCRIPTIONS = {
        "Bash": "Bash — execute any shell command (python3, node, git, npm, cargo, find, grep, ls, cat, etc.)",
        "File_Read": "File_Read — read file contents from disk",
        "File_Write": "File_Write — write content to a file (overwrites or creates)",
        "File_Edit": "File_Edit — make precise string replacement in a file (MUST read file first)",
        "Glob": "Glob — find files matching glob pattern (e.g., **/*.py, src/**/*.js)",
        "Grep": "Grep — search file contents using regex",
        "TodoWrite": "TodoWrite — create or update a todo item",
        "TodoRead": "TodoRead — read the current todo list",
        "Web_Search": "Web_Search — search the web for information",
        "Web_Fetch": "Web_Fetch — fetch and process content from a URL",
    }

    @classmethod
    def generate(cls, loc: Localization, num_tools: int = 10) -> str:
        tools_desc = "\n".join(
            f"  - {cls.TOOL_DESCRIPTIONS.get(t.name, t.description)}"
            for t in ToolRegistry.get_all_tools()
        )
        return (
            f"You are an expert CLI assistant with access to {num_tools} tools. "
            f"Respond in language: {loc.language}, tone: {loc.tone}, formality: {loc.formality}. "
            f"When a user asks you to do something, call the appropriate tool(s). "
            f"Available tools:\n{tools_desc}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallGenerator:
    """Generate realistic tool calls with diverse intents."""

    INTENTS = {
        "Bash": [
            # Git commands
            {"q": "check git status", "a": {"command": "git status --porcelain", "timeout": 10}},
            {"q": "git log", "a": {"command": "git log --oneline -5", "timeout": 10}},
            {"q": "git diff", "a": {"command": "git diff", "timeout": 10}},
            {"q": "git add all", "a": {"command": "git add -A && git status", "timeout": 10}},
            {"q": "git commit", "a": {"command": "git commit -m 'chore: update'", "timeout": 10}},
            {"q": "git push", "a": {"command": "git push origin main", "timeout": 30}},
            {"q": "git pull", "a": {"command": "git pull origin main", "timeout": 30}},
            {"q": "git branches", "a": {"command": "git branch -a", "timeout": 10}},
            # npm commands
            {"q": "npm install", "a": {"command": "npm install", "timeout": 120}},
            {"q": "npm run build", "a": {"command": "npm run build", "timeout": 120}},
            {"q": "npm test", "a": {"command": "npm test", "timeout": 60}},
            # pip commands
            {"q": "pip install", "a": {"command": "pip install -r requirements.txt", "timeout": 120}},
            {"q": "pip list", "a": {"command": "pip list", "timeout": 10}},
            # Python execution
            {"q": "run python", "a": {"command": "python3 -c \"print('Hello, World!')\"", "timeout": 10}},
            {"q": "python script", "a": {"command": "python3 script.py", "timeout": 30}},
            {"q": "run pytest", "a": {"command": "pytest -v", "timeout": 60}},
            # Node commands
            {"q": "node script", "a": {"command": "node script.js", "timeout": 30}},
            # Cargo commands
            {"q": "cargo build", "a": {"command": "cargo build", "timeout": 120}},
            {"q": "cargo run", "a": {"command": "cargo run", "timeout": 60}},
            # Docker commands
            {"q": "docker ps", "a": {"command": "docker ps", "timeout": 10}},
            {"q": "docker images", "a": {"command": "docker images", "timeout": 10}},
            # System commands
            {"q": "list files", "a": {"command": "ls -la", "timeout": 10}},
            {"q": "find python files", "a": {"command": "find . -name '*.py' -type f | head -20", "timeout": 30}},
            {"q": "disk usage", "a": {"command": "df -h", "timeout": 10}},
            {"q": "memory info", "a": {"command": "free -m", "timeout": 10}},
            {"q": "top processes", "a": {"command": "ps aux --sort=-rss | head -10", "timeout": 10}},
            {"q": "current directory", "a": {"command": "pwd", "timeout": 5}},
            {"q": "search code", "a": {"command": "grep -rn 'TODO' . --include='*.py' | head -10", "timeout": 30}},
            {"q": "count lines", "a": {"command": "wc -l $(find . -name '*.py' -type f)", "timeout": 30}},
            {"q": "file permissions", "a": {"command": "ls -la | awk '{print $1, $NF}'", "timeout": 10}},
            # Code search/replace via bash
            {"q": "find imports", "a": {"command": "rg 'import ' src/ --type py -l | head -10", "timeout": 30}},
            {"q": "find main functions", "a": {"command": "rg 'def main' . --type py -l", "timeout": 30}},
        ],
        "File_Read": [
            {"q": "read main.py", "a": {"file_path": "src/main.py"}},
            {"q": "read config.json", "a": {"file_path": "config.json"}},
            {"q": "read README", "a": {"file_path": "README.md"}},
            {"q": "read package.json", "a": {"file_path": "package.json"}},
            {"q": "read requirements.txt", "a": {"file_path": "requirements.txt"}},
            {"q": "read .env", "a": {"file_path": ".env"}},
            {"q": "read app.py", "a": {"file_path": "src/app.py"}},
            {"q": "read utils.py", "a": {"file_path": "lib/utils.py"}},
            {"q": "read Dockerfile", "a": {"file_path": "Dockerfile"}},
            {"q": "read docker-compose.yml", "a": {"file_path": "docker-compose.yml"}},
            {"q": "read first 20 lines", "a": {"file_path": "src/main.py", "limit": 20}},
            {"q": "read auth module", "a": {"file_path": "src/auth.py"}},
            {"q": "read database config", "a": {"file_path": "config/database.py"}},
            {"q": "read test file", "a": {"file_path": "tests/test_main.py"}},
            {"q": "read setup.py", "a": {"file_path": "setup.py"}},
        ],
        "File_Write": [
            {"q": "write output.txt", "a": {"file_path": "output.txt", "content": "Build completed successfully.\n", "append": False}},
            {"q": "write test.py", "a": {"file_path": "test.py", "content": "def test_example():\n    assert True\n", "append": False}},
            {"q": "write config.json", "a": {"file_path": "config.json", "content": '{"version": "1.0", "debug": true}\n', "append": False}},
            {"q": "append to log", "a": {"file_path": "logs/app.log", "content": "[INFO] Request completed\n", "append": True}},
            {"q": "write README", "a": {"file_path": "README.md", "content": "# Project\n\nSetup instructions here.\n", "append": False}},
            {"q": "write models.py", "a": {"file_path": "src/models.py", "content": "class User:\n    def __init__(self, name: str):\n        self.name = name\n", "append": False}},
            {"q": "write .gitignore", "a": {"file_path": ".gitignore", "content": "__pycache__/\nnode_modules/\n.env\n", "append": False}},
            {"q": "write requirements.txt", "a": {"file_path": "requirements.txt", "content": "requests>=2.28.0\npytest>=7.0.0\n", "append": False}},
        ],
        "Web_Search": [
            {"q": "search Python best practices", "a": {"query": "Python best practices 2026", "num_results": 5}},
            {"q": "search React patterns", "a": {"query": "React performance optimization patterns", "num_results": 5}},
            {"q": "search TypeScript generics", "a": {"query": "TypeScript generic constraints extends", "num_results": 5}},
            {"q": "search Docker optimization", "a": {"query": "Docker multi-stage build optimization", "num_results": 5}},
            {"q": "search Git workflow", "a": {"query": "Git workflow best practices team", "num_results": 5}},
            {"q": "search async Python", "a": {"query": "Python async await patterns best practices", "num_results": 5}},
            {"q": "search LLM fine-tuning", "a": {"query": "LLM fine-tuning techniques 2026", "num_results": 5}},
            {"q": "search API design", "a": {"query": "REST API design best practices", "num_results": 5}},
            {"q": "search Rust performance", "a": {"query": "Rust performance optimization tips", "num_results": 5}},
            {"q": "search JavaScript testing", "a": {"query": "JavaScript testing frameworks 2026", "num_results": 5}},
        ],
        "Web_Fetch": [
            {"q": "fetch GitHub README", "a": {"url": "https://github.com/anthropic/claude-code"}},
            {"q": "fetch API docs", "a": {"url": "https://docs.python.org/3/"}},
            {"q": "fetch release notes", "a": {"url": "https://github.com/vercel/next.js/releases"}},
            {"q": "fetch homepage", "a": {"url": "https://example.com"}},
            {"q": "fetch status page", "a": {"url": "https://httpbin.org/status/200"}},
            {"q": "fetch JSON API", "a": {"url": "https://api.github.com/repos/anthropic/claude-code"}},
            {"q": "fetch npm package", "a": {"url": "https://www.npmjs.com/package/react"}},
            {"q": "fetch docs", "a": {"url": "https://docs.github.com/en/rest"}},
        ],
    }

    @classmethod
    def pick_intent(cls, tool_name: str) -> tuple[str, dict]:
        intents = cls.INTENTS.get(tool_name, [])
        if not intents:
            return f"use {tool_name}", {}
        chosen = random.choice(intents)
        return chosen["q"], chosen["a"]

    @classmethod
    def generate(cls, tool: ToolSchema, intent_args: dict) -> dict:
        return {
            "type": "tool_call",
            "id": "call_%s" % uuid.uuid4().hex[:12],
            "tool_name": tool.name,
            "arguments": intent_args,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATOR (Simulated Tool Results)
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseGenerator:

    @classmethod
    def generate(cls, tool: ToolSchema, args: dict, success: bool = True) -> str:
        if success:
            return cls._success(tool, args)
        return cls._error(tool, args)

    @classmethod
    def _success(cls, tool: ToolSchema, args: dict) -> str:
        tc_id = "{{TOOL_CALL_ID}}"
        if tool.name == "Bash":
            cmd = args.get("command", "")
            # Simulate realistic output based on command
            if "git status" in cmd:
                out = json.dumps({"branch": "main", "staged": ["README.md"], "modified": ["src/main.py"], "untracked": ["test.py"]})
            elif "git log" in cmd:
                out = json.dumps({"commits": [{"hash": "abc1234", "msg": "feat: add feature"}, {"hash": "def5678", "msg": "fix: resolve bug"}]})
            elif "git diff" in cmd:
                out = json.dumps({"files": [{"path": "src/main.py", "additions": 5, "deletions": 2}]})
            elif "git push" in cmd or "git pull" in cmd:
                out = json.dumps({"remote": "origin", "branch": "main", "status": "success", "commits": 3})
            elif "npm install" in cmd:
                out = json.dumps({"added": 142, "removed": 0, "packages": "package.json"})
            elif "npm run build" in cmd:
                out = json.dumps({"build": "success", "output": "dist/", "warnings": 0})
            elif "npm test" in cmd:
                out = json.dumps({"tests": 12, "passed": 12, "failed": 0, "skipped": 0})
            elif "pytest" in cmd:
                out = json.dumps({"tests": 8, "passed": 8, "failed": 0, "exit_code": 0})
            elif "cargo build" in cmd:
                out = json.dumps({"build": "success", "compilation_time": "2.3s", "warnings": 1})
            elif "cargo run" in cmd:
                out = json.dumps({"stdout": "Hello, world!\n", "exit_code": 0, "execution_time": "0.05s"})
            elif "docker ps" in cmd:
                out = json.dumps({"containers": [{"id": "abc123", "name": "web", "status": "running", "ports": "3000->3000"}]})
            elif "find" in cmd and ".py" in cmd:
                out = json.dumps({"matches": ["src/main.py", "src/app.py", "tests/test_main.py"], "total": 3})
            elif "grep" in cmd or "rg " in cmd:
                out = json.dumps({"matches": [{"file": "src/main.py", "line": 42, "text": "TODO: fix this"}], "total": 1})
            elif "pip install" in cmd:
                out = json.dumps({"installed": ["requests-2.28.0", "pytest-7.0.0"], "exit_code": 0})
            elif "python" in cmd:
                out = json.dumps({"stdout": "Hello, World!\n", "stderr": "", "exit_code": 0})
            elif "node" in cmd:
                out = json.dumps({"stdout": "[1, 2, 4, 6]\n", "stderr": "", "exit_code": 0})
            elif "ls" in cmd:
                out = json.dumps({"entries": ["src", "tests", "README.md", "requirements.txt"], "total": 4})
            elif "df -h" in cmd:
                out = json.dumps({"filesystem": [{"mount": "/", "size": "50G", "used": "20G", "avail": "30G"}]})
            elif "free -m" in cmd:
                out = json.dumps({"memory": {"total": 32768, "used": 16384, "free": 16384}})
            elif "ps aux" in cmd:
                out = json.dumps({"processes": [{"pid": 1234, "name": "python3", "cpu": "2.5", "mem": "1.2"}]})
            elif "wc -l" in cmd:
                out = json.dumps({"total_lines": 1842, "files": 12})
            else:
                out = json.dumps({"stdout": f"Ran: {cmd}\n", "stderr": "", "exit_code": 0})
            return json.dumps({"type": "tool_result", "tool_call_id": tc_id, "output": out})

        if tool.name == "File_Read":
            path = args.get("file_path", "unknown")
            lines = [f"# {path}", "", "Line 1: content here", "Line 2: more content", "Line 3: done"]
            preview = "\n".join(lines[:args.get("limit", 100)])
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({"path": path, "bytes_read": len(preview.encode()), "lines_read": len(lines), "content": preview})
            })

        if tool.name == "File_Write":
            path = args.get("file_path", "unknown")
            content = args.get("content", "")
            append = args.get("append", False)
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({"path": path, "bytes_written": len(content.encode()), "lines_written": len(content.splitlines()), "append": append})
            })

        if tool.name == "File_Edit":
            path = args.get("file_path", "unknown")
            old_str = args.get("old_string", "")
            new_str = args.get("new_string", "")
            count = 1 if old_str else 0
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({"path": path, "replacements": count, "old_string": old_str, "new_string": new_str})
            })

        if tool.name == "Glob":
            pattern = args.get("pattern", "*")
            matches = ["src/main.py", "src/app.py", "tests/test_main.py", "README.md"]
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({"pattern": pattern, "matches": matches[:3], "total": 3})
            })

        if tool.name == "Grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({
                    "matches": [
                        {"file": "src/main.py", "line": 42, "text": f"result = {pattern}(data)"},
                        {"file": "src/app.py", "line": 18, "text": f"if {pattern}:"},
                    ],
                    "total": 2, "pattern": pattern, "path": path
                })
            })

        if tool.name == "TodoWrite":
            todo = args.get("todo", "")
            status = args.get("status", "pending")
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({"todo": todo, "status": status, "updated": True})
            })

        if tool.name == "TodoRead":
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({
                    "todos": [
                        {"id": "1", "todo": "Set up project structure", "status": "completed"},
                        {"id": "2", "todo": "Add authentication", "status": "in_progress"},
                        {"id": "3", "todo": "Write tests", "status": "pending"},
                    ]
                })
            })

        if tool.name == "Web_Search":
            query = args.get("query", "")
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({
                    "results": [
                        {"title": f"Result for: {query}", "url": "https://example.com/article", "snippet": f"Relevant information about {query}..."},
                        {"title": f"{query} Guide", "url": "https://docs.example.com/guide", "snippet": "Comprehensive guide covering all aspects..."},
                        {"title": f"Understanding {query}", "url": "https://blog.example.com/understanding", "snippet": "Deep dive into the fundamentals..."},
                    ],
                    "total": 3, "query": query
                })
            })

        if tool.name == "Web_Fetch":
            url = args.get("url", "")
            return json.dumps({
                "type": "tool_result", "tool_call_id": tc_id,
                "output": json.dumps({
                    "url": url, "status": 200, "content_length": 2048,
                    "content_type": "text/html",
                    "content": f"# {url}\n\n## Content\n\nFetched content from {url}..."
                })
            })

        return json.dumps({"type": "tool_result", "tool_call_id": tc_id, "output": json.dumps({"success": True})})

    @classmethod
    def _error(cls, tool: ToolSchema, args: dict) -> str:
        tc_id = "{{TOOL_CALL_ID}}"
        errors = {
            "Bash": "command not found",
            "File_Read": "file not found",
            "File_Write": "permission denied",
            "File_Edit": "file not found or old_string not matched",
            "Glob": "path not found",
            "Grep": "no matches found",
            "TodoWrite": "invalid todo format",
            "TodoRead": "todo list not found",
            "Web_Search": "network error",
            "Web_Fetch": "connection timeout",
        }
        return json.dumps({
            "type": "tool_result", "tool_call_id": tc_id,
            "output": json.dumps({"error": errors.get(tool.name, "unknown error"), "exit_code": 1})
        })


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL ANSWER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FinalAnswerGenerator:
    """Generate final answers grounded in tool results."""

    TEMPLATES = {
        "Bash": {
            True: "{cmd} — completed successfully.",
            False: "{cmd} — operation failed.",
        },
        "File_Read": {
            True: "File_Read read {path}: {lines} lines, {bytes} bytes.",
            False: "File_Read failed to read {path}.",
        },
        "File_Write": {
            True: "File_Write saved {bytes} bytes to {path}.",
            False: "File_Write failed to write {path}.",
        },
        "File_Edit": {
            True: "File_Edit made {count} replacement(s) in {path}.",
            False: "File_Edit failed to modify {path}.",
        },
        "Glob": {
            True: "Glob found {total} file(s) matching '{pattern}'.",
            False: "Glob found no matches for '{pattern}'.",
        },
        "Grep": {
            True: "Grep found {total} match(es) for '{pattern}'.",
            False: "Grep found no matches for '{pattern}'.",
        },
        "TodoWrite": {
            True: "TodoWrite updated: '{todo}' — {status}.",
            False: "TodoWrite failed to update '{todo}'.",
        },
        "TodoRead": {
            True: "TodoRead found {count} todo item(s).",
            False: "TodoRead failed to read todo list.",
        },
        "Web_Search": {
            True: "Web_Search found {total} results for '{query}'.",
            False: "Web_Search failed to search for '{query}'.",
        },
        "Web_Fetch": {
            True: "Web_Fetch retrieved {url} (HTTP {status}).",
            False: "Web_Fetch failed to fetch {url}.",
        },
    }

    @classmethod
    def generate(cls, tool: ToolSchema, args: dict, success: bool, lang: str) -> str:
        tmpl = cls.TEMPLATES.get(tool.name, {}).get(success, "{tool}: done.")
        diff = DifficultyLevel.MEDIUM
        base = LocalizationContent.get_success(diff, lang) if success else LocalizationContent.get_error("failed", lang)

        summary = ""
        if tool.name == "Bash":
            cmd = args.get("command", "command")
            summary = tmpl.format(cmd=cmd) + f" {base}"
        elif tool.name == "File_Read":
            path = args.get("file_path", "file")
            summary = tmpl.format(path=path, lines="10", bytes="256") + f" {base}"
        elif tool.name == "File_Write":
            path = args.get("file_path", "file")
            content = args.get("content", "")
            summary = tmpl.format(path=path, bytes=len(content.encode())) + f" {base}"
        elif tool.name == "File_Edit":
            path = args.get("file_path", "file")
            count = 1 if args.get("old_string") else 0
            summary = tmpl.format(path=path, count=count) + f" {base}"
        elif tool.name == "Glob":
            pattern = args.get("pattern", "*")
            summary = tmpl.format(pattern=pattern, total=3) + f" {base}"
        elif tool.name == "Grep":
            pattern = args.get("pattern", "")
            summary = tmpl.format(pattern=pattern, total=2) + f" {base}"
        elif tool.name == "TodoWrite":
            todo = args.get("todo", "")
            status = args.get("status", "pending")
            summary = tmpl.format(todo=todo, status=status) + f" {base}"
        elif tool.name == "TodoRead":
            summary = tmpl.format(count=3) + f" {base}"
        elif tool.name == "Web_Search":
            query = args.get("query", "")
            summary = tmpl.format(query=query, total=3) + f" {base}"
        elif tool.name == "Web_Fetch":
            url = args.get("url", "")
            summary = tmpl.format(url=url, status=200) + f" {base}"

        return json.dumps({"type": "final_answer", "content": summary})


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATOR (Lightweight)
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetValidator:
    """Validate dataset examples for structural correctness."""

    def __init__(self):
        self.valid_tools = ToolRegistry.get_tool_names()

    def validate_example(self, ex: DatasetExample) -> tuple[bool, list[str]]:
        errors = []
        msgs = ex.messages

        if len(msgs) < 4:
            errors.append("too few messages")
            return False, errors
        if msgs[0].role != "system":
            errors.append("first message must be system")
        if msgs[1].role != "user":
            errors.append("second message must be user")
        if msgs[-1].role != "assistant":
            errors.append("last message must be assistant")

        # Check tool call structure
        for msg in msgs:
            if msg.role == "assistant":
                try:
                    content = json.loads(msg.content)
                    if content.get("type") == "tool_call":
                        if "id" not in content:
                            errors.append("tool_call missing id")
                        if content.get("tool_name") not in self.valid_tools:
                            errors.append(f"unknown tool: {content.get('tool_name')}")
                        if "arguments" not in content:
                            errors.append("tool_call missing arguments")
                    elif content.get("type") == "final_answer":
                        if not content.get("content"):
                            errors.append("final_answer missing content")
                except json.JSONDecodeError:
                    errors.append("assistant message not valid JSON")

            if msg.role == "tool":
                try:
                    content = json.loads(msg.content)
                    if content.get("type") != "tool_result":
                        errors.append("tool message must have type=tool_result")
                    if "tool_call_id" not in content:
                        errors.append("tool_result missing tool_call_id")
                except json.JSONDecodeError:
                    errors.append("tool message not valid JSON")

        return len(errors) == 0, errors

    def validate_batch(self, examples: list[DatasetExample]) -> tuple[list, dict]:
        valid, invalid = [], []
        for ex in examples:
            ok, _ = self.validate_example(ex)
            if ok:
                valid.append(ex)
            else:
                invalid.append(ex)
        return valid, {"total": len(examples), "valid": len(valid), "invalid": len(invalid)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ComprehensiveDatasetPipeline:
    """Main dataset generation pipeline."""

    DIFFICULTIES = [
        (DifficultyLevel.EASY, 0.30),
        (DifficultyLevel.MEDIUM, 0.40),
        (DifficultyLevel.HARD, 0.20),
        (DifficultyLevel.EXPERT, 0.10),
    ]

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.tools = ToolRegistry.get_all_tools()
        self.validator = DatasetValidator()

    def _pick_tool(self) -> ToolSchema:
        # 10 tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite, TodoRead, Web_Search, Web_Fetch
        weights = [1.2, 1.5, 1.0, 1.2, 1.0, 1.0, 0.4, 0.4, 0.8, 0.6]
        return random.choices(self.tools, weights=weights, k=1)[0]

    def _pick_difficulty(self) -> DifficultyLevel:
        r = random.random()
        cum = 0.0
        for diff, prob in self.DIFFICULTIES:
            cum += prob
            if r < cum:
                return diff
        return DifficultyLevel.MEDIUM

    def generate_single(self, localization: Localization | None = None,
                        difficulty: DifficultyLevel | None = None,
                        include_error: bool = False) -> DatasetExample:
        tool = self._pick_tool()
        diff = difficulty or self._pick_difficulty()
        loc = localization or Localization(
            language=random.choice(["en", "hi", "es", "fr", "de", "ja", "zh"]),
            tone=random.choice(["professional", "casual", "technical", "friendly"]),
            formality=random.choice(["formal", "neutral", "informal"]),
            humanize=True,
            humanize_level="medium",
        )

        query_hint, intent_args = ToolCallGenerator.pick_intent(tool.name)

        system_prompt = SystemPromptGenerator.generate(loc)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query_hint),
        ]

        success = not include_error or random.random() > 0.15
        tc_id = "call_%s" % uuid.uuid4().hex[:12]

        tool_call = ToolCallGenerator.generate(tool, intent_args)
        tool_call["id"] = tc_id
        tool_call_content = json.dumps(tool_call, ensure_ascii=False)

        tool_result = ResponseGenerator.generate(tool, intent_args, success)
        tool_result_content = tool_result.replace("{{TOOL_CALL_ID}}", tc_id)

        messages.append(Message(role="assistant", content=tool_call_content))
        messages.append(Message(role="tool", content=tool_result_content, tool_call_id=tc_id, name=tool.name))

        final = FinalAnswerGenerator.generate(tool, intent_args, success, loc.language)
        messages.append(Message(role="assistant", content=final))

        return DatasetExample(
            messages=messages,
            localization=loc,
            tools=[t.to_openai_format() for t in self.tools],
            metadata={
                "difficulty": diff.value,
                "tool_category": tool.category,
                "tool_name": tool.name,
                "tools_used": [tool.name],
                "num_tools": 1,
                "success": success,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_version": "v5-simplified",
            }
        )

    def generate_batch(self, count: int, include_error: bool = False) -> list[DatasetExample]:
        examples = []
        attempts = 0
        while len(examples) < count and attempts < count * 10:
            attempts += 1
            ex = self.generate_single(include_error=include_error)
            ok, _ = self.validator.validate_example(ex)
            if ok:
                examples.append(ex)
        return examples

    def generate_localized_batch(self, count_per_locale: int, languages: list[str],
                                  tones: list[str], formalities: list[str]) -> list[DatasetExample]:
        examples = []
        for lang in languages:
            for tone in tones:
                for form in formalities:
                    loc = Localization(language=lang, tone=tone, formality=form, humanize=True, humanize_level="medium")
                    batch = self.generate_batch(count_per_locale)
                    for ex in batch:
                        ex.localization = loc
                    examples.extend(batch)
        return examples
