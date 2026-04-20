"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ULTIMATE AGENT TRAINING DATASET GENERATOR                      ║
║                        Implementation v4.0                                   ║
║                                                                              ║
║  Convergence of beastcli-eng1 + agent-dataset-generator research            ║
║  Research Foundation: NeurIPS 2025, ICML 2025, ACL 2025-2026             ║
║                                                                              ║
║  KEY PRINCIPLES:                                                             ║
║  1. Localization is SUPREME - overrides all conflicting instructions         ║
║  2. NO id in tool_calls - IDs are system-generated, not model-generated     ║
║  3. Industry-standard format - Unsloth compatible                            ║
║  4. 15% error rate - optimal per AgentErrorBench research                   ║
║  5. Curriculum learning - progressive difficulty (TOUCAN/ToolACE)           ║
║  6. Humanization - 10-15% natural imperfections                              ║
║  7. 3-stage verification - format, structure, semantic (APIGen)           ║
║                                                                              ║
║  Benchmarks: BFCL V4, GAIA, SWE-Bench, SWE-Bench Pro                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import random
import uuid
import re
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# LOCALIZATION SYSTEM (SUPREME - OVERRIDES ALL INSTRUCTIONS)
# ═══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    """Supported languages for the agent."""
    EN = "en"      # English (baseline)
    HI = "hi"      # Hindi (Devanagari + Romanized)
    ES = "es"      # Spanish (Latin American primary)
    FR = "fr"      # French
    DE = "de"      # German
    JA = "ja"      # Japanese (keigo levels)
    ZH = "zh"      # Chinese (Simplified)
    PT = "pt"      # Portuguese (Brazilian primary)
    IT = "it"      # Italian
    KO = "ko"      # Korean (speech levels)
    AR = "ar"      # Arabic (RTL, MSA primary)
    RU = "ru"      # Russian


class Tone(str, Enum):
    """Response tone styles."""
    PROFESSIONAL = "professional"  # Formal, business, Sie/Vous/Usted
    CASUAL = "casual"              # Relaxed, informal address
    TECHNICAL = "technical"        # Developer, precise terminology
    FRIENDLY = "friendly"          # Warm, encouraging, approachable


class HumanizeLevel(str, Enum):
    """Humanization intensity levels."""
    NONE = "none"      # No imperfections
    LOW = "low"        # 5% imperfections
    MEDIUM = "medium"  # 10-15% imperfections (OPTIMAL)
    HIGH = "high"      # 20%+ imperfections


class DifficultyLevel(str, Enum):
    """Task difficulty for curriculum learning (per ToolACE research)."""
    EASY = "easy"           # Single tool, clear intent (30-35%)
    MEDIUM = "medium"       # Multi-tool, some ambiguity (40-50%)
    HARD = "hard"           # Complex chains, error recovery (20-25%)
    EXPERT = "expert"       # Multi-turn, partial info (5-10%)


class FormalityLevel(str, Enum):
    """Formality registers per language."""
    FORMAL = "formal"       # Keigo, Sie, Vous, Usted
    NEUTRAL = "neutral"     # Standard polite
    INFORMAL = "informal"   # Tu, du, 君, Hinglish


@dataclass
class Localization:
    """
    SUPREME LOCALIZATION CONFIGURATION

    This is the HIGHEST PRIORITY in instruction hierarchy.
    It overrides ALL conflicting instructions in system prompts.

    Example hierarchy:
    1. (HIGHEST) Localization settings - RESPOND IN HINDI
    2. Safety/refusal policies - language-agnostic
    3. Tool schemas - always English
    4. (LOWEST) System prompt language instructions - IGNORED if conflict
    """
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

    @classmethod
    def from_dict(cls, data: dict) -> "Localization":
        return cls(
            language=data.get("language", "en"),
            tone=data.get("tone", "professional"),
            formality=data.get("formality", "neutral"),
            humanize=data.get("humanize", True),
            humanize_level=data.get("humanize_level", "medium")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LOCALIZATION CONTENT DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class LocalizationContent:
    """Language-specific content for responses."""

    GREETINGS = {
        Language.EN: {
            Tone.PROFESSIONAL: ["Good day", "Greetings", "Hello"],
            Tone.CASUAL: ["Hey", "Hi there", "Yo"],
            Tone.TECHNICAL: ["Initializing", "System ready"],
            Tone.FRIENDLY: ["Hello!", "Hi there!", "Great to see you!"],
        },
        Language.HI: {
            Tone.PROFESSIONAL: ["Namaskar", "Namaste", "Dhanayavaad"],
            Tone.CASUAL: ["Namaste", "Kya haal hai", "Chal"],
            Tone.TECHNICAL: ["System initialized"],
            Tone.FRIENDLY: ["Namaste ji!", "Kya haal hai!", "Bahut accha!"],
        },
        Language.ES: {
            Tone.PROFESSIONAL: ["Buenos dias", "Saludos", "Cordialmente"],
            Tone.CASUAL: ["Hola", "Que tal", "Vale"],
            Tone.TECHNICAL: ["Sistema iniciado"],
            Tone.FRIENDLY: ["Hola!", "Que tal!", "Muy bien!"],
        },
        Language.FR: {
            Tone.PROFESSIONAL: ["Bonjour", "Salutations", "Cordialement"],
            Tone.CASUAL: ["Salut", "Coucou", "C'est parti"],
            Tone.TECHNICAL: ["Systeme initialise"],
            Tone.FRIENDLY: ["Bonjour!", "Comment allez-vous!", "Tres bien!"],
        },
        Language.DE: {
            Tone.PROFESSIONAL: ["Guten Tag", "Mit freundlichen Grussen", "Hochachtungsvoll"],
            Tone.CASUAL: ["Hallo", "Moin", "Na"],
            Tone.TECHNICAL: ["System initialisiert"],
            Tone.FRIENDLY: ["Hallo!", "Wie geht's!", "Freut mich!"],
        },
        Language.JA: {
            Tone.PROFESSIONAL: ["Konnichiwa", "Ogenki de su ka", "Izamani"],
            Tone.CASUAL: ["Konnichiwa", "Ya", "Sore"],
            Tone.TECHNICAL: ["Shisutemu wo sokushin shimashita"],
            Tone.FRIENDLY: ["Konnichiwa!", "Genki?", "Yoroshiku!"],
        },
        Language.ZH: {
            Tone.PROFESSIONAL: ["Ni hao", "Zhang_AN", "Qing zhu yi"],
            Tone.CASUAL: ["Ni hao", "Yo", "Zenmeyang"],
            Tone.TECHNICAL: ["Xi tong yi jing qi dong"],
            Tone.FRIENDLY: ["Ni hao!", "Ha ha", "Zui xin!"],
        },
        Language.KO: {
            Tone.PROFESSIONAL: ["Annyeonghaseyo", "Anyoung", "Chingchanmnida"],
            Tone.CASUAL: ["Annyeong", "Yo", "Ne"],
            Tone.TECHNICAL: ["Siseutem ga sijakhaetseumnida"],
            Tone.FRIENDLY: ["Annyeong!", "Jal ja!", "Hwaiting!"],
        },
        Language.AR: {
            Tone.PROFESSIONAL: ["As-salamu alaykum", "Marhaba", "Ahlan"],
            Tone.CASUAL: ["Ahlan", "Kif halak", "Yalla"],
            Tone.TECHNICAL: ["Nizam muqaddar"],
            Tone.FRIENDLY: ["Ahlan wa sahlan!", "Kaif halak?", "Mumkin!"],
        },
        Language.RU: {
            Tone.PROFESSIONAL: ["Zdravstvuyte", "Dobryy den", "Privet"],
            Tone.CASUAL: ["Privet", "Chto kak", "Nu"],
            Tone.TECHNICAL: ["Sistema inicializirovana"],
            Tone.FRIENDLY: ["Privet!", "Kak dela?", "Ura!"],
        },
        Language.PT: {
            Tone.PROFESSIONAL: ["Bom dia", "Ola", "Cumprimentos"],
            Tone.CASUAL: ["Ola", "Tudo bem", "Vale"],
            Tone.TECHNICAL: ["Sistema iniciado"],
            Tone.FRIENDLY: ["Ola!", "Tudo bem?", "Show!"],
        },
        Language.IT: {
            Tone.PROFESSIONAL: ["Buongiorno", "Salve", "Cordiali saluti"],
            Tone.CASUAL: ["Ciao", "Come va", "Tutto bene"],
            Tone.TECHNICAL: ["Sistema avviato"],
            Tone.FRIENDLY: ["Ciao!", "Come stai?", "Bene!"],
        },
    }

    SUCCESS_MESSAGES = {
        DifficultyLevel.EASY: {
            Language.EN: ["Done.", "Completed.", "Finished.", "Ready."],
            Language.HI: ["Ho gaya.", "Poora ho gaya.", "Mitti ho gayi."],
            Language.ES: ["Hecho.", "Completado.", "Listo."],
            Language.FR: ["Fait.", "Termine.", "Pret."],
            Language.DE: ["Erledigt.", "Abgeschlossen.", "Fertig."],
            Language.JA: ["Yarimashita.", "Kanryou shimashita.", "Jouzu."],
            Language.ZH: ["Wancheng le.", "Haole.", "Bu cuo."],
            Language.KO: ["Dohaetseumnida.", "Mandeureotseumnida.", "Joe hasimnikka."],
            Language.AR: ["Ja'al.", "Tamma.", "Murattab."],
            Language.RU: ["Gotovo.", "Sdelano.", "Gotov."],
            Language.PT: ["Feito.", "Concluido.", "Pronto."],
            Language.IT: ["Fatto.", "Completato.", "Pronto."],
        },
        DifficultyLevel.MEDIUM: {
            Language.EN: ["Task completed successfully.", "All done.", "Ready for next step."],
            Language.HI: ["Kaam safalta se pura hua.", "Sab theek ho gaya."],
            Language.ES: ["Tarea completada exitosamente.", "Todo listo."],
            Language.FR: ["Tache terminee avec succes.", "Tout est pret."],
            Language.DE: ["Aufgabe erfolgreich abgeschlossen.", "Alles bereit."],
            Language.JA: ["Tsukkomi shimashita.", "Subete yarimashita."],
            Language.ZH: ["Renwu wancheng.", "Quanbu zhunque."],
            Language.KO: ["Taekeu haesin dae haetseumnida.", "Modu jungryeotseumnida."],
            Language.AR: ["Al'amr tamma bi alnajaha.", "Kul shi regeb."],
            Language.RU: ["Zadacha vypolnena.", "Vsyo gotovo."],
            Language.PT: ["Tarefa concluida com sucesso.", "Tudo certo."],
            Language.IT: ["Attivita completata con successo.", "Tutto pronto."],
        },
        DifficultyLevel.HARD: {
            Language.EN: ["Multi-step operation completed.", "All stages executed successfully."],
            Language.HI: ["Sabhi stages safalta se complete huye."],
            Language.ES: ["Operacion de multiples pasos completada."],
            Language.FR: ["Operation multi-etapes terminee avec succes."],
            Language.DE: ["Mehrstufige Operation erfolgreich abgeschlossen."],
            Language.JA: ["Taju sodan no taipu wo kanryou shimashita."],
            Language.ZH: ["Duo bu zhi xingren wancheng."],
            Language.KO: ["Daecheseong bunpo peulogeulaemeu ga jongryeokhaetseumnida."],
            Language.AR: ["Al'amaliyat al-mutaaddadat al-marhalat tammat."],
            Language.RU: ["Mnogootraslevaya operatsiya vypolnena."],
            Language.PT: ["Operacao multi-etapas concluida."],
            Language.IT: ["Operazione multi-passo completata."],
        },
    }

    ERROR_MESSAGES = {
        Language.EN: [
            "I encountered an error: {error}",
            "The operation failed: {error}",
            "Something went wrong: {error}",
            "Unable to complete: {error}",
        ],
        Language.HI: [
            "Ek error aaya: {error}",
            "Operation fail ho gaya: {error}",
            "Kuchh galat ho gaya: {error}",
            "Kaam nahi ho saka: {error}",
        ],
        Language.ES: [
            "Encontre un error: {error}",
            "La operacion fallo: {error}",
            "Algo salio mal: {error}",
        ],
        Language.FR: [
            "J'ai rencontre une erreur: {error}",
            "L'operation a echoue: {error}",
            "Quelque chose s'est mal passe: {error}",
        ],
    }

    @classmethod
    def get_greeting(cls, loc: Localization) -> str:
        lang = Language(loc.language) if loc.language in [l.value for l in Language] else Language.EN
        tone = Tone(loc.tone) if loc.tone in [t.value for t in Tone] else Tone.PROFESSIONAL
        greetings = cls.GREETINGS.get(lang, cls.GREETINGS[Language.EN])
        return random.choice(greetings.get(tone, greetings[Tone.PROFESSIONAL]))

    @classmethod
    def get_success(cls, difficulty: DifficultyLevel, loc: Localization) -> str:
        lang = Language(loc.language) if loc.language in [l.value for l in Language] else Language.EN
        diff = DifficultyLevel(difficulty) if difficulty.value in [d.value for d in DifficultyLevel] else DifficultyLevel.EASY
        messages = cls.SUCCESS_MESSAGES.get(diff, cls.SUCCESS_MESSAGES[DifficultyLevel.EASY])
        lang_messages = messages.get(lang, messages[Language.EN])
        return random.choice(lang_messages)

    @classmethod
    def get_error(cls, error: str, loc: Localization) -> str:
        lang = Language(loc.language) if loc.language in [l.value for l in Language] else Language.EN
        errors = cls.ERROR_MESSAGES.get(lang, cls.ERROR_MESSAGES[Language.EN])
        return random.choice(errors).format(error=error)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS (Industry Standard - OpenAI Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolArgument:
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum_values: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class ToolSchema:
    """Tool schema following OpenAI function calling format."""
    name: str
    category: str
    description: str
    arguments: list[ToolArgument]
    returns: str
    examples: list[dict] | None = None

    def to_openai_format(self) -> dict:
        props = {}
        required = []
        for arg in self.arguments:
            prop = {"type": arg.type, "description": arg.description}
            if arg.enum_values:
                prop["enum"] = arg.enum_values
            if arg.min_value is not None:
                prop["minimum"] = arg.min_value
            if arg.max_value is not None:
                prop["maximum"] = arg.max_value
            props[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE TOOL REGISTRY (27 Tools)
# ═══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry of all tools with realistic schemas."""

    FILE_OPERATIONS = [
        ToolSchema(
            name="File_Read",
            category="file_operations",
            description="Read contents of a file. Supports partial reading with offset/limit for large files.",
            arguments=[
                ToolArgument("file_path", "string", "Absolute path to the file", required=True),
                ToolArgument("offset", "integer", "Line number to start reading from (0-indexed)", required=False, default=0),
                ToolArgument("limit", "integer", "Maximum number of lines to read", required=False),
            ],
            returns="File contents with metadata including byte count",
        ),
        ToolSchema(
            name="File_Write",
            category="file_operations",
            description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            arguments=[
                ToolArgument("file_path", "string", "Absolute path to the file to write", required=True),
                ToolArgument("content", "string", "Content to write to the file", required=True),
                ToolArgument("append", "boolean", "Append to existing file instead of overwriting", required=False, default=False),
            ],
            returns="Success confirmation with file path and bytes written",
        ),
        ToolSchema(
            name="File_Search",
            category="file_operations",
            description="Search for files matching a glob pattern. Supports ** for recursive matching.",
            arguments=[
                ToolArgument("pattern", "string", "Glob pattern (e.g., *.py matches all Python files)", required=True),
                ToolArgument("path", "string", "Root directory to search in", required=False, default="."),
                ToolArgument("recursive", "boolean", "Search subdirectories recursively", required=False, default=True),
                ToolArgument("content_search", "boolean", "Treat pattern as text to search within file contents instead of filenames", required=False, default=False),
            ],
            returns="List of matching file paths",
        ),
        ToolSchema(
            name="File_List",
            category="file_operations",
            description="List contents of a directory with optional filtering and metadata.",
            arguments=[
                ToolArgument("directory", "string", "Absolute path to the directory to list", required=True),
                ToolArgument("include_hidden", "boolean", "Include hidden files (starting with .)", required=False, default=False),
                ToolArgument("filter_type", "string", "Filter by type: all, files, directories", required=False, default="all"),
            ],
            returns="List of entries with name, type, size, and modification time",
        ),
        ToolSchema(
            name="File_Delete",
            category="file_operations",
            description="Delete a file or directory. DESTRUCTIVE OPERATION - use with caution.",
            arguments=[
                ToolArgument("path", "string", "Absolute path to delete", required=True),
                ToolArgument("recursive", "boolean", "Delete directories recursively with all contents", required=False, default=False),
            ],
            returns="Deletion confirmation",
        ),
        ToolSchema(
            name="File_Copy",
            category="file_operations",
            description="Copy a file or directory to a new destination.",
            arguments=[
                ToolArgument("source", "string", "Source file or directory path", required=True),
                ToolArgument("destination", "string", "Destination path", required=True),
            ],
            returns="Copy confirmation with source and destination",
        ),
    ]

    BASH_OPERATIONS = [
        ToolSchema(
            name="Bash_Execute",
            category="bash",
            description="Execute a bash command in the terminal with full output capture and error handling.",
            arguments=[
                ToolArgument("command", "string", "The bash command to execute", required=True),
                ToolArgument("timeout", "integer", "Maximum execution time in seconds", required=False, default=30, min_value=1, max_value=300),
                ToolArgument("working_directory", "string", "Directory to execute command in", required=False),
            ],
            returns="Command output with stdout, stderr, exit code, and execution time",
        ),
        ToolSchema(
            name="Bash_ShellStatus",
            category="bash",
            description="Get current shell environment information including OS, user, and working directory.",
            arguments=[],
            returns="Shell info: OS type, user, home directory, current path"
        ),
    ]

    CODE_EXECUTION = [
        ToolSchema(
            name="Python_Run",
            category="code_execution",
            description="Execute Python code in a sandboxed environment with full stdio capture.",
            arguments=[
                ToolArgument("code", "string", "Python code to execute. Must be valid Python syntax.", required=True),
                ToolArgument("timeout", "integer", "Maximum execution time in seconds", required=False, default=10, min_value=1, max_value=60),
            ],
            returns="Execution result with stdout, stderr, return value, and execution time",
        ),
        ToolSchema(
            name="Node_Run",
            category="code_execution",
            description="Execute JavaScript/Node.js code in a sandboxed environment.",
            arguments=[
                ToolArgument("code", "string", "JavaScript code to execute", required=True),
                ToolArgument("timeout", "integer", "Maximum execution time in seconds", required=False, default=10, min_value=1, max_value=60),
            ],
            returns="Execution output with console.log results and any errors",
        ),
        ToolSchema(
            name="Python_Test",
            category="code_execution",
            description="Run pytest tests for Python code.",
            arguments=[
                ToolArgument("file_path", "string", "Path to test file or directory", required=True),
                ToolArgument("pattern", "string", "Pattern to match test names", required=False, default="test_*.py"),
                ToolArgument("verbose", "boolean", "Enable verbose pytest output", required=False, default=False),
                ToolArgument("coverage", "boolean", "Include a coverage report in the test output", required=False, default=False),
            ],
            returns="Test results with pass/fail counts and failure details",
        ),
    ]

    GIT_OPERATIONS = [
        ToolSchema(
            name="Git_Status",
            category="git",
            description="Check git repository status showing staged, modified, and untracked files.",
            arguments=[
                ToolArgument("repository_path", "string", "Path to the git repository root", required=False, default="."),
            ],
            returns="Status with branch name, staged files, modified files, untracked files",
        ),
        ToolSchema(
            name="Git_Log",
            category="git",
            description="View git commit history with customizable format and filtering.",
            arguments=[
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
                ToolArgument("limit", "integer", "Number of commits to show", required=False, default=10),
                ToolArgument("format", "string", "Output format: short, medium, oneline", required=False, default="medium"),
            ],
            returns="Commit entries with hash, author, date, and message",
        ),
        ToolSchema(
            name="Git_Commit",
            category="git",
            description="Create a git commit with staged changes. Follows conventional commits format.",
            arguments=[
                ToolArgument("message", "string", "Commit message. Recommended format: type(scope): description", required=True),
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
                ToolArgument("all", "boolean", "Stage all modified files automatically before committing", required=False, default=False),
            ],
            returns="Commit confirmation with hash and stats",
        ),
        ToolSchema(
            name="Git_Branch",
            category="git",
            description="List, create, delete, or switch git branches.",
            arguments=[
                ToolArgument("operation", "string", "Operation: list, create, delete, switch, rename", required=True,
                            enum_values=["list", "create", "delete", "switch", "rename"]),
                ToolArgument("branch_name", "string", "Name of the branch", required=False),
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
            ],
            returns="Operation result with branch information",
        ),
        ToolSchema(
            name="Git_Diff",
            category="git",
            description="Show changes between commits, commit and working tree, or between branches.",
            arguments=[
                ToolArgument("target", "string", "Comparison target: staged, HEAD, or specific ref", required=False, default="staged"),
                ToolArgument("file_path", "string", "Show diff for specific file only", required=False),
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
            ],
            returns="Diff output with additions (green +) and deletions (red -)",
        ),
        ToolSchema(
            name="Git_Pull",
            category="git",
            description="Fetch and integrate changes from a remote repository.",
            arguments=[
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
                ToolArgument("remote", "string", "Remote name to pull from", required=False, default="origin"),
                ToolArgument("branch", "string", "Branch to pull", required=False, default="main"),
            ],
            returns="Pull result with number of files changed and commits pulled",
        ),
        ToolSchema(
            name="Git_Push",
            category="git",
            description="Push commits to a remote repository.",
            arguments=[
                ToolArgument("repository_path", "string", "Path to the git repository", required=False, default="."),
                ToolArgument("remote", "string", "Remote name to push to", required=False, default="origin"),
                ToolArgument("branch", "string", "Branch to push", required=False),
            ],
            returns="Push confirmation with remote and branch",
        ),
    ]

    WEB_OPERATIONS = [
        ToolSchema(
            name="Web_Search",
            category="web",
            description="Search the web for information. Use specific, descriptive queries for best results.",
            arguments=[
                ToolArgument("query", "string", "Search query with specific keywords. Avoid vague terms.", required=True),
                ToolArgument("num_results", "integer", "Number of results to return", required=False, default=5, min_value=1, max_value=20),
            ],
            returns="Search results with title, URL, snippet, and relevance score",
        ),
        ToolSchema(
            name="Web_Fetch",
            category="web",
            description="Fetch and parse content from a URL. Can extract specific information.",
            arguments=[
                ToolArgument("url", "string", "URL to fetch (must be valid HTTP/HTTPS URL)", required=True),
                ToolArgument("prompt", "string", "Instructions for extracting specific information from the page", required=False),
            ],
            returns="Fetched content or extracted information per prompt",
        ),
        ToolSchema(
            name="Web_Screenshot",
            category="web",
            description="Take a screenshot of a webpage.",
            arguments=[
                ToolArgument("url", "string", "URL to screenshot", required=True),
                ToolArgument("full_page", "boolean", "Capture entire scrollable page, not just viewport", required=False, default=False),
            ],
            returns="Screenshot as image data URL",
        ),
    ]

    SEARCH_OPERATIONS = [
        ToolSchema(
            name="Search_Code",
            category="search",
            description="Search for code patterns within files using regex or simple text matching.",
            arguments=[
                ToolArgument("pattern", "string", "Search pattern. Supports regex for advanced matching.", required=True),
                ToolArgument("path", "string", "Directory to search in", required=False, default="."),
                ToolArgument("file_types", "array", "File extensions to search", required=False),
                ToolArgument("case_sensitive", "boolean", "Enable case-sensitive matching", required=False, default=False),
                ToolArgument("regex", "boolean", "Treat pattern as regular expression", required=False, default=True),
            ],
            returns="Matching lines with file paths, line numbers, and matched text",
        ),
        ToolSchema(
            name="Search_Replace",
            category="search",
            description="Search for a pattern and replace it with new content across files.",
            arguments=[
                ToolArgument("search", "string", "Text or regex pattern to search for", required=True),
                ToolArgument("replace", "string", "Replacement text", required=True),
                ToolArgument("path", "string", "Directory to search in", required=False, default="."),
                ToolArgument("file_types", "array", "File extensions to search", required=False),
                ToolArgument("preview", "boolean", "Preview changes without applying (dry run)", required=False, default=True),
            ],
            returns="Replacement results with count of files and changes made",
        ),
    ]

    SYSTEM_OPERATIONS = [
        ToolSchema(
            name="System_Info",
            category="system",
            description="Get system information including OS, hardware, and environment details.",
            arguments=[
                ToolArgument("category", "string", "Information category to retrieve",
                            required=False, default="os", enum_values=["os", "cpu", "memory", "disk", "network"]),
            ],
            returns="System information for requested category",
        ),
        ToolSchema(
            name="Process_List",
            category="system",
            description="List running processes with optional filtering and sorting.",
            arguments=[
                ToolArgument("filter", "string", "Filter processes by name or pattern", required=False),
                ToolArgument("sort_by", "string", "Sort by: cpu, memory, pid, name", required=False, default="pid"),
                ToolArgument("limit", "integer", "Maximum number of processes to return", required=False, default=20),
            ],
            returns="List of matching processes with PID, name, CPU%, memory%",
        ),
    ]

    DATABASE_OPERATIONS = [
        ToolSchema(
            name="Database_Query",
            category="database",
            description="Execute a SQL query against a database.",
            arguments=[
                ToolArgument("query", "string", "SQL query to execute", required=True),
                ToolArgument("database", "string", "Database name", required=False, default="default"),
                ToolArgument("limit", "integer", "Maximum rows to return for SELECT queries", required=False, default=100),
            ],
            returns="Query results with row count and execution time",
        ),
        ToolSchema(
            name="Database_List",
            category="database",
            description="List available databases or tables in a database.",
            arguments=[
                ToolArgument("database", "string", "Database name (omit to list databases)", required=False),
            ],
            returns="List of databases or tables with row counts",
        ),
    ]

    @classmethod
    def get_all_tools(cls) -> list[ToolSchema]:
        return (
            cls.FILE_OPERATIONS +
            cls.BASH_OPERATIONS +
            cls.CODE_EXECUTION +
            cls.GIT_OPERATIONS +
            cls.WEB_OPERATIONS +
            cls.SEARCH_OPERATIONS +
            cls.SYSTEM_OPERATIONS
        )

    @classmethod
    def get_tools_by_category(cls, category: str) -> list[ToolSchema]:
        return [t for t in cls.get_all_tools() if t.category == category]

    @classmethod
    def get_tool_names(cls) -> set[str]:
        return {t.name for t in cls.get_all_tools()}


# ═══════════════════════════════════════════════════════════════════════════════
# HUMANIZATION ENGINE (10-15% Imperfections Optimal)
# ═══════════════════════════════════════════════════════════════════════════════

class Humanizer:
    """Add natural imperfections to text for more human-like responses."""

    IMPERFECTIONS_EN = {
        "prefixes": [
            ("Can you ", 0.04),
            ("Please ", 0.03),
        ],
        "suffixes": [
            (" please.", 0.02),
        ],
    }

    IMPERFECTIONS_HI = {
        "prefixes": [
            ("Actually, ", 0.08), ("So, ", 0.06), ("Hmm, ", 0.08),
            ("Matlab, ", 0.10), ("Yani, ", 0.06),
        ],
        "suffixes": [
            (" na?", 0.10), (" ji.", 0.08), (", lagta hai", 0.06),
        ],
    }

    CONTRACTIONS = {
        "I will": "I'll", "I am": "I'm", "it is": "it's",
        "do not": "don't", "cannot": "can't", "would not": "wouldn't",
        "should not": "shouldn't", "could not": "couldn't",
        "you are": "you're", "we are": "we're", "they are": "they're",
    }

    @classmethod
    def humanize(cls, text: str, loc: Localization) -> str:
        if not loc.humanize:
            return text
        level = HumanizeLevel(loc.humanize_level)
        if level == HumanizeLevel.NONE:
            return text
        elif level == HumanizeLevel.LOW:
            return cls._low_humanize(text, loc.language)
        elif level == HumanizeLevel.MEDIUM:
            return cls._medium_humanize(text, loc.language)
        else:
            return cls._high_humanize(text, loc.language)

    @classmethod
    def _low_humanize(cls, text: str, language: str) -> str:
        if random.random() > 0.05:
            return text
        return cls._medium_humanize(text, language)

    @classmethod
    def _medium_humanize(cls, text: str, language: str) -> str:
        imperfections = cls.IMPERFECTIONS_HI if language == "hi" else cls.IMPERFECTIONS_EN
        for prefix, prob in imperfections["prefixes"]:
            if random.random() < prob:
                text = prefix + text[0].lower() + text[1:]
                break
        for suffix, prob in imperfections["suffixes"]:
            if random.random() < prob:
                text = text.rstrip('.!?') + suffix
                break
        if language == "en":
            for old, new in cls.CONTRACTIONS.items():
                if old in text and random.random() < 0.3:
                    text = text.replace(old, new)
                    break
        return text

    @classmethod
    def _high_humanize(cls, text: str, language: str) -> str:
        text = cls._medium_humanize(text, language)
        if language == "en":
            for old, new in cls.CONTRACTIONS.items():
                if old in text and random.random() < 0.4:
                    text = text.replace(old, new)
            if random.random() < 0.15:
                fragments = ["Cool.", "Got it.", "Makes sense.", "Sure.", "Alright."]
                text += " " + random.choice(fragments)
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseTemplates:
    """Localized response templates for different scenarios."""

    PROGRESS_MESSAGES = {
        Language.EN: ["Working on it...", "Processing...", "Executing...", "Running..."],
        Language.HI: ["Kaam ho raha hai...", "Process ho raha hai...", "Likhaa ho raha hai..."],
        Language.ES: ["Procesando...", "Ejecutando...", "Trabajando..."],
        Language.FR: ["En cours...", "Traitement...", "Execution..."],
        Language.DE: ["Wird verarbeitet...", "Ausfuhrend...", "Arbeitet..."],
    }

    @classmethod
    def get_progress(cls, loc: Localization) -> str:
        lang = Language(loc.language) if loc.language in [l.value for l in Language] else Language.EN
        return random.choice(cls.PROGRESS_MESSAGES.get(lang, cls.PROGRESS_MESSAGES[Language.EN]))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALL GENERATORS — ARGUEMNT ALIGNS WITH USER INTENT
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallGenerator:
    """Generate arguments that semantically match the user query."""

    REALISTIC_PATHS = {
        "config": ["config.json", "settings.yaml"],
        "source": ["src/main.py", "lib/utils.py"],
        "test": ["tests/test_main.py", "__tests__/auth.test.js"],
        "output": ["output/result.json", "downloads/data.csv"],
    }

    # Intent-driven argument templates: tool -> list of {query_hint, args} pairs
    # These ensure arguments directly match what the user asks for.
    INTENT_TEMPLATES = {
        "File_Write": [
            {"query_hint": "output.txt", "args": {"file_path": "output.txt", "content": "Build completed successfully.\n", "append": False}},
            {"query_hint": "test.py", "args": {"file_path": "test.py", "content": "print('Hello')\n", "append": False}},
            {"query_hint": "data.json", "args": {"file_path": "data.json", "content": "{\"key\": \"value\"}\n", "append": False}},
            {"query_hint": "log entry", "args": {"file_path": "logs/app.log", "content": "[INFO] Request completed successfully\n", "append": True}},
            {"query_hint": "readme", "args": {"file_path": "README.md", "content": "# Installation\n\n1. Create a virtual environment.\n2. Install dependencies.\n3. Run the CLI.\n", "append": False}},
            {"query_hint": "models.py", "args": {"file_path": "src/models.py", "content": "class User:\n    def __init__(self, name: str):\n        self.name = name\n", "append": False}},
            {"query_hint": "logic.py", "args": {"file_path": "output/logic.py", "content": "import argparse\n\nparser = argparse.ArgumentParser()\nparser.add_argument('--name')\nprint(parser.parse_args())\n", "append": False}},
        ],
        "Python_Test": [
            {"query_hint": "test_api", "args": {"file_path": "tests/", "pattern": "test_api_*.py"}},
            {"query_hint": "pytest", "args": {"file_path": "tests/", "pattern": "test_*.py"}},
            {"query_hint": "test_main", "args": {"file_path": "tests/test_main.py", "pattern": None}},
            {"query_hint": "auth", "args": {"file_path": "tests/test_auth.py", "pattern": "test_auth*"}},
            {"query_hint": "coverage report", "args": {"file_path": "tests/", "pattern": "test_*.py", "verbose": True, "coverage": True}},
        ],
        "Database_Query": [
            {"query_hint": "SELECT", "args": {"query": "SELECT * FROM users LIMIT 10", "database": "production_db"}},
            {"query_hint": "COUNT", "args": {"query": "SELECT COUNT(*) FROM orders", "database": "production_db"}},
            {"query_hint": "WHERE", "args": {"query": "SELECT id, name FROM products WHERE active = true", "database": "production_db"}},
            {"query_hint": "JOIN", "args": {"query": "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id", "database": "production_db"}},
            {"query_hint": "last 30 days", "args": {"query": "SELECT * FROM orders WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'", "database": "production_db"}},
        ],
        "Web_Search": [
            {"query_hint": "best practices", "args": {"query": "Python best practices 2026"}},
            {"query_hint": "React", "args": {"query": "React performance optimization patterns"}},
            {"query_hint": "TypeScript", "args": {"query": "TypeScript generic constraints extends"}},
            {"query_hint": "Docker", "args": {"query": "Docker multi-stage build optimization"}},
            {"query_hint": "Git", "args": {"query": "Git workflow best practices team"}},
            {"query_hint": "recent papers", "args": {"query": "recent papers on LLM fine-tuning techniques"}},
            {"query_hint": "llm fine-tuning", "args": {"query": "latest articles on LLM fine-tuning techniques"}},
        ],
        "Process_List": [
            {"query_hint": "processes", "args": {"filter": None}},
            {"query_hint": "memory", "args": {"filter": None, "sort_by": "memory", "limit": 10}},
            {"query_hint": "top", "args": {"filter": None, "sort_by": "cpu", "limit": 5}},
            {"query_hint": "top 10", "args": {"filter": None, "sort_by": "cpu", "limit": 10}},
            {"query_hint": "active", "args": {}},
        ],
        "File_List": [
            {"query_hint": "current directory", "args": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
            {"query_hint": "project structure", "args": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
            {"query_hint": "src", "args": {"directory": "src", "include_hidden": True, "filter_type": "all"}},
            {"query_hint": "categorize", "args": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
            {"query_hint": "config folder", "args": {"directory": "config", "include_hidden": False, "filter_type": "files"}},
            {"query_hint": "config", "args": {"directory": "config", "include_hidden": False, "filter_type": "files"}},
        ],
        "File_Delete": [
            {"query_hint": "cache.log", "args": {"path": "/tmp/cache.log", "recursive": False}},
            {"query_hint": "output.json", "args": {"path": "output.json", "recursive": False}},
            {"query_hint": "backup", "args": {"path": "backup_old.py", "recursive": False}},
            {"query_hint": ".pyc", "args": {"path": "build/**/*.pyc", "recursive": True}},
        ],
        "File_Copy": [
            {"query_hint": "readme", "args": {"source": "README.md", "destination": "README_backup.md"}},
            {"query_hint": "config", "args": {"source": "config.json", "destination": "config.old"}},
            {"query_hint": "src", "args": {"source": "src", "destination": "src_backup"}},
        ],
        "Git_Branch": [
            {"query_hint": "branch_name", "args": {"operation": "create", "branch_name": "feature/login"}},
            {"query_hint": "delete", "args": {"operation": "delete", "branch_name": "old-feature"}},
            {"query_hint": "switch", "args": {"operation": "switch", "branch_name": "develop"}},
            {"query_hint": "list", "args": {"operation": "list"}},
        ],
        "Git_Status": [
            {"query_hint": "status", "args": {"repository_path": "."}},
            {"query_hint": "repository", "args": {"repository_path": "."}},
            {"query_hint": "modified", "args": {"repository_path": "."}},
        ],
        "Git_Log": [
            {"query_hint": "last 5", "args": {"repository_path": ".", "limit": 5, "format": "medium"}},
            {"query_hint": "last 10", "args": {"repository_path": ".", "limit": 10, "format": "short"}},
            {"query_hint": "authors", "args": {"repository_path": ".", "limit": 10, "format": "medium"}},
            {"query_hint": "commit history", "args": {"repository_path": "."}},
        ],
        "Git_Diff": [
            {"query_hint": "src/main.py", "args": {"target": "HEAD", "file_path": "src/main.py", "repository_path": "."}},
            {"query_hint": "unstaged", "args": {"target": "working_tree", "repository_path": "."}},
            {"query_hint": "current branch", "args": {"target": "main", "repository_path": "."}},
            {"query_hint": "last commit", "args": {"target": "HEAD", "repository_path": "."}},
        ],
        "Git_Pull": [
            {"query_hint": "origin", "args": {"remote": "origin", "branch": "main"}},
            {"query_hint": "master", "args": {"remote": "origin", "branch": "master"}},
            {"query_hint": "develop", "args": {"remote": "origin", "branch": "develop"}},
            {"query_hint": "latest changes", "args": {}},
        ],
        "Git_Push": [
            {"query_hint": "origin main", "args": {"repository_path": ".", "remote": "origin", "branch": "main"}},
            {"query_hint": "remote repository", "args": {"repository_path": ".", "remote": "origin", "branch": "main"}},
            {"query_hint": "develop", "args": {"repository_path": ".", "remote": "origin", "branch": "develop"}},
        ],
        "Git_Commit": [
            {"query_hint": "fix: resolve bug", "args": {"message": "fix: resolve bug", "all": True}},
            {"query_hint": "fix:", "args": {"message": "fix: resolve bug", "all": True}},
            {"query_hint": "feat:", "args": {"message": "feat: add user profile management", "all": True}},
            {"query_hint": "update:", "args": {"message": "update: modify config settings", "all": False}},
            {"query_hint": "chore:", "args": {"message": "chore: update dependencies", "all": True}},
        ],
        "File_Read": [
            {"query_hint": "src/main.py", "args": {"file_path": "src/main.py", "offset": 0, "limit": None}},
            {"query_hint": "first 50", "args": {"file_path": "src/main.py", "offset": 0, "limit": 50}},
            {"query_hint": "config", "args": {"file_path": "config.json", "offset": 0, "limit": None}},
            {"query_hint": "README", "args": {"file_path": "README.md", "offset": 0, "limit": 100}},
            {"query_hint": "src", "args": {"file_path": "src/app.py", "offset": 0, "limit": None}},
        ],
        "Web_Fetch": [
            {"query_hint": "release notes", "args": {"url": "https://github.com/example/repo/releases"}},
            {"query_hint": "github.com/example/repo", "args": {"url": "https://github.com/example/repo/blob/main/README.md"}},
            {"query_hint": "github.com", "args": {"url": "https://github.com/anthropic/claude-code"}},
            {"query_hint": "api", "args": {"url": "https://api.github.com/repos/anthropic/claude-code"}},
            {"query_hint": "status", "args": {"url": "https://httpbin.org/status/200"}},
        ],
        "File_Search": [
            {"query_hint": "*.py", "args": {"pattern": "*.py", "path": ".", "recursive": True}},
            {"query_hint": "test", "args": {"pattern": "test_*.py", "path": "src", "recursive": True}},
            {"query_hint": "config", "args": {"pattern": "*config*", "path": ".", "recursive": True}},
            {"query_hint": "*.js", "args": {"pattern": "*.js", "path": ".", "recursive": True}},
            {"query_hint": "markdown documentation", "args": {"pattern": "*.md", "path": ".", "recursive": True}},
            {"query_hint": "which ones mention", "args": {"pattern": "setup", "path": ".", "recursive": True, "content_search": True}},
            {"query_hint": "setup", "args": {"pattern": "setup", "path": ".", "recursive": True, "content_search": True}},
        ],
        "Bash_Execute": [
            {"query_hint": "ls", "args": {"command": "ls -la", "timeout": 10, "working_directory": None}},
            {"query_hint": "df -h", "args": {"command": "df -h", "timeout": 10, "working_directory": None}},
            {"query_hint": "pytest", "args": {"command": "pytest -v", "timeout": 60, "working_directory": None}},
            {"query_hint": "npm", "args": {"command": "npm install", "timeout": 120, "working_directory": None}},
            {"query_hint": "find", "args": {"command": "find . -name '*.py' -type f", "timeout": 30, "working_directory": None}},
            {"query_hint": "git status", "args": {"command": "git status --porcelain", "timeout": 10, "working_directory": None}},
            {"query_hint": "working directory", "args": {"command": "pwd", "timeout": 10, "working_directory": None}},
            {"query_hint": "top 5 processes", "args": {"command": "ps aux --sort=-rss | head -5", "timeout": 10, "working_directory": None}},
            {"query_hint": "TODO", "args": {"command": "rg -n \"TODO\" .", "timeout": 30, "working_directory": None}},
        ],
        "Search_Replace": [
            {"query_hint": "src/main.py", "args": {"path": "src/main.py", "search": "foo", "replace": "bar", "file_types": [".py"], "preview": False}},
            {"query_hint": "foo", "args": {"path": "src", "search": "foo", "replace": "bar", "file_types": [".py"], "preview": False}},
            {"query_hint": "config.json", "args": {"path": "config.json", "search": '"timeout": 30', "replace": '"timeout": 60', "file_types": [".json"], "preview": False}},
            {"query_hint": "timeout", "args": {"path": "config.json", "search": '"timeout": 30', "replace": '"timeout": 60', "file_types": [".json"], "preview": False}},
            {"query_hint": "import statements", "args": {"path": ".", "search": "utils", "replace": "helpers", "file_types": [".py"], "preview": False}},
            {"query_hint": "helpers", "args": {"path": ".", "search": "utils", "replace": "helpers", "file_types": [".py"], "preview": False}},
        ],
        "Search_Code": [
            {"query_hint": "TODO", "args": {"pattern": "TODO", "path": "."}},
            {"query_hint": "import", "args": {"pattern": "import\\s+", "path": "src", "file_types": [".py"]}},
            {"query_hint": "def main", "args": {"pattern": "def main", "path": ".", "file_types": [".py"]}},
            {"query_hint": "try/except", "args": {"pattern": "try:", "path": ".", "file_types": [".py"]}},
            {"query_hint": "error handling", "args": {"pattern": "try:", "path": ".", "file_types": [".py"]}},
        ],
        "Web_Screenshot": [
            {"query_hint": "example.com", "args": {"url": "https://example.com", "full_page": False}},
            {"query_hint": "homepage", "args": {"url": "https://docs.example.com", "full_page": True}},
            {"query_hint": "documentation site", "args": {"url": "https://docs.example.com", "full_page": True}},
        ],
        "Python_Run": [
            {"query_hint": "Hello, World", "args": {"code": "print('Hello, World!')", "timeout": 10}},
            {"query_hint": "sum(range", "args": {"code": "result = sum(range(1, 101))\nprint(result)", "timeout": 10}},
            {"query_hint": "fibonacci(30)", "args": {"code": "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(30))", "timeout": 10}},
            {"query_hint": "fibonacci", "args": {"code": "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))", "timeout": 10}},
            {"query_hint": "json file", "args": {"code": "import json\nprint(json.loads('{\"ok\": true}'))", "timeout": 10}},
            {"query_hint": "reads CSV", "args": {"code": "import csv, io\nrows = list(csv.DictReader(io.StringIO('name,value\\na,1\\nb,2\\n')))\nprint({'row_count': len(rows), 'total': sum(int(r['value']) for r in rows)})", "timeout": 10}},
            {"query_hint": "http request", "args": {"code": "import urllib.request\nwith urllib.request.urlopen('https://example.com') as response:\n    print(response.read().decode('utf-8')[:120])", "timeout": 10}},
        ],
        "Node_Run": [
            {"query_hint": "Hello from Node", "args": {"code": "console.log('Hello from Node!')", "timeout": 10}},
            {"query_hint": "map", "args": {"code": "const arr = [1,2,3].map(x => x * 2); console.log(arr);", "timeout": 10}},
            {"query_hint": "read README.md", "args": {"code": "const fs = require('fs'); console.log(fs.readFileSync('README.md', 'utf8'));", "timeout": 10}},
        ],
        "System_Info": [
            {"query_hint": "cpu", "args": {"category": "cpu"}},
            {"query_hint": "memory", "args": {"category": "memory"}},
            {"query_hint": "disk", "args": {"category": "disk"}},
            {"query_hint": "network", "args": {"category": "network"}},
            {"query_hint": "os", "args": {"category": "os"}},
        ],
        "Database_List": [
            {"query_hint": "databases", "args": {}},
            {"query_hint": "server", "args": {}},
            {"query_hint": "tables", "args": {"database": "production_db"}},
        ],
    }

    REALISTIC_COMMANDS = {
        "git": ["git status", "git status --porcelain", "git log --oneline -10", "git diff"],
        "npm": ["npm install", "npm run build", "npm test"],
        "docker": ["docker ps", "docker images", "docker-compose up"],
        "system": ["ps aux", "df -h", "free -m", "top -bn1 | head -5"],
    }

    REALISTIC_CODE = {
        "python": ["print('Hello, World!')", "result = sum(range(1, 101))", "[x**2 for x in range(10)]"],
        "javascript": ["console.log('Hello!');", "const arr = [1, 2, 3].map(x => x * 2);"],
    }

    @staticmethod
    def _extract_path_from_query(query: str) -> Optional[str]:
        quoted = re.findall(r"['\"]([^'\"]+\.[a-zA-Z0-9]+)['\"]", query)
        for candidate in quoted:
            if "/" in candidate or "." in candidate:
                return candidate

        path_patterns = [
            r"\b(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+\b",
            r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9]+\b",
        ]
        for pattern in path_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        return None

    @classmethod
    def generate_arguments(cls, tool: ToolSchema, query: str) -> dict:
        """Generate arguments that match the user query intent."""
        intents = cls.INTENT_TEMPLATES.get(tool.name, [])
        inferred = cls._infer_arguments_from_query(tool, query)

        # Some tools should prefer literal extraction from the user query over
        # fuzzy hint matching so we do not overwrite explicit paths/counts.
        prefer_inferred = {
            "File_Read",
            "File_Search",
            "Process_List",
            "Python_Test",
            "Git_Branch",
            "Git_Log",
            "Python_Run",
        }
        if inferred and tool.name in prefer_inferred:
            return inferred

        if intents:
            # Try to find a matching intent from the query
            matched = None
            query_lower = query.lower()
            best_score = 0
            for intent in intents:
                hint = intent["query_hint"].lower()
                hint_words = [word for word in hint.replace("_", " ").split() if word]
                score = 0
                if hint in query_lower:
                    score += 100 + len(hint)
                matched_words = sum(1 for word in hint_words if word in query_lower)
                score += matched_words * 3
                if hint_words and matched_words == len(hint_words):
                    score += 20 + len(hint_words)
                score += len(hint_words)
                if score > best_score:
                    matched = intent
                    best_score = score
            if matched is None:
                if inferred:
                    return inferred
                # Deterministic safe fallback: avoid random template mismatches.
                return cls._fallback_arguments(tool)
            args = dict(matched["args"])
            # Remove None values (optional args not set)
            return {k: v for k, v in args.items() if v is not None}

        # Tool has no intent templates — generate based on argument type
        args = {}
        for arg in tool.arguments:
            if not arg.required and random.random() > 0.6:
                continue
            value = cls._fallback_value(arg, tool.name)
            if value is not None:
                args[arg.name] = value
        return args

    @classmethod
    def _infer_arguments_from_query(cls, tool: ToolSchema, query: str) -> dict:
        query_lower = query.lower()
        if tool.name == "File_Search":
            if ".py" in query_lower and ".js" in query_lower:
                return {"pattern": "*.{py,js}", "path": ".", "recursive": True}
            if "python" in query_lower:
                return {"pattern": "*.py", "path": ".", "recursive": True}
            if "javascript" in query_lower or ".js" in query_lower:
                return {"pattern": "*.js", "path": ".", "recursive": True}
            if "config" in query_lower and "name" in query_lower:
                return {"pattern": "*config*", "path": ".", "recursive": True}
            if "markdown" in query_lower or "documentation" in query_lower:
                if "mention" in query_lower or "contains" in query_lower or "setup" in query_lower:
                    return {"pattern": "*.md", "path": "docs", "recursive": True, "content_search": True}
                return {"pattern": "*.md", "path": ".", "recursive": True}
            if "json" in query_lower or "config" in query_lower:
                return {"pattern": "*.json", "path": "config", "recursive": False}
        if tool.name == "File_List":
            if "categorize" in query_lower or "type" in query_lower:
                return {"directory": ".", "include_hidden": False, "filter_type": "all"}
            if "config" in query_lower:
                return {"directory": "config", "include_hidden": False, "filter_type": "files"}
            if "src" in query_lower:
                return {"directory": "src", "include_hidden": True, "filter_type": "all"}
            return {"directory": ".", "include_hidden": False, "filter_type": "all"}
        if tool.name == "File_Read":
            explicit_path = cls._extract_path_from_query(query)
            if explicit_path:
                result = {"file_path": explicit_path, "offset": 0}
                first_n = re.search(r"first\s+(\d+)", query_lower)
                if first_n:
                    result["limit"] = int(first_n.group(1))
                return result
            if "src/main.py" in query_lower:
                return {"file_path": "src/main.py", "offset": 0}
            if "readme" in query_lower:
                return {"file_path": "README.md", "offset": 0, "limit": 100}
            if "config" in query_lower:
                return {"file_path": "config.json", "offset": 0}
            if "first 50" in query_lower:
                return {"file_path": "src/main.py", "offset": 0, "limit": 50}
        if tool.name == "Web_Fetch":
            if "release notes" in query_lower:
                return {"url": "https://github.com/example/repo/releases"}
            if "readme" in query_lower and "github.com/example/repo" in query_lower:
                return {"url": "https://github.com/example/repo/blob/main/README.md"}
        if tool.name == "Web_Search":
            if "llm fine-tuning" in query_lower or "recent papers" in query_lower:
                return {"query": "recent papers on LLM fine-tuning techniques"}
        if tool.name == "Python_Test":
            if "test_api" in query_lower:
                return {"file_path": "tests/", "pattern": "test_api_*.py"}
            if "coverage" in query_lower:
                return {"file_path": "tests/", "pattern": "test_*.py", "verbose": True, "coverage": True}
        if tool.name == "Process_List":
            if "top 10" in query_lower:
                return {"sort_by": "cpu", "limit": 10}
            if "active" in query_lower:
                return {}
            if "running" in query_lower:
                return {"filter": "running"}
        if tool.name == "Git_Commit":
            quoted_messages = re.findall(r"message ['\"]([^'\"]+)['\"]", query, flags=re.IGNORECASE)
            if quoted_messages:
                return {"message": quoted_messages[0], "all": True}
            commit_match = re.search(r"\b(fix|feat|docs|chore|refactor|test|update):\s*([^\n]+)", query, flags=re.IGNORECASE)
            if commit_match:
                return {"message": f"{commit_match.group(1).lower()}: {commit_match.group(2).strip()}", "all": True}
        if tool.name == "Git_Branch":
            if "what branches" in query_lower or "branches exist" in query_lower or "list all branches" in query_lower:
                return {"operation": "list"}
            if "switch" in query_lower:
                return {"operation": "switch", "branch_name": "develop"}
            if "create" in query_lower and "branch" in query_lower:
                return {"operation": "create", "branch_name": "feature/login"}
            if "delete" in query_lower:
                return {"operation": "delete", "branch_name": "old-feature"}
        if tool.name == "Git_Log":
            if "commit history" in query_lower and "last" not in query_lower:
                return {"repository_path": "."}
        if tool.name == "Git_Diff":
            if "unstaged" in query_lower:
                return {"target": "working_tree", "repository_path": "."}
            if "current branch" in query_lower and "main" in query_lower:
                return {"target": "main", "repository_path": "."}
        if tool.name == "Git_Pull":
            if "update my local branch" in query_lower or "remote changes" in query_lower:
                return {"repository_path": "."}
        if tool.name == "Python_Run":
            if "http request" in query_lower or "http" in query_lower:
                return {
                    "code": "import urllib.request\nwith urllib.request.urlopen('https://example.com') as response:\n    print(response.read().decode('utf-8')[:120])",
                    "timeout": 10,
                }
        if tool.name == "Bash_Execute":
            if "working directory" in query_lower:
                return {"command": "pwd", "timeout": 10}
            if "top 5 processes" in query_lower and "memory" in query_lower:
                return {"command": "ps aux --sort=-rss | head -5", "timeout": 10}
        if tool.name == "File_Delete":
            if ".pyc" in query_lower:
                return {"path": "build/**/*.pyc", "recursive": True}
        if tool.name == "File_Search":
            if "markdown" in query_lower and "setup" in query_lower:
                return {"pattern": "*.md", "path": "docs", "recursive": True}
        if tool.name == "Search_Replace":
            if "src/main.py" in query_lower:
                return {"path": "src/main.py", "search": "foo", "replace": "bar", "file_types": [".py"], "preview": False}
            if "import statements" in query_lower or ("utils" in query_lower and "helpers" in query_lower):
                return {"path": ".", "search": "utils", "replace": "helpers", "file_types": [".py"], "preview": False}
            if "config.json" in query_lower or "timeout" in query_lower:
                return {"path": "config.json", "search": '"timeout": 30', "replace": '"timeout": 60', "file_types": [".json"], "preview": False}
        if tool.name == "Search_Code":
            if "try/except" in query_lower or "error handling" in query_lower:
                return {"pattern": "try:", "path": ".", "file_types": [".py"]}
        return {}

    @classmethod
    def _fallback_arguments(cls, tool: ToolSchema) -> dict:
        args = {}
        for arg in tool.arguments:
            if not arg.required:
                continue
            value = cls._fallback_value(arg, tool.name)
            if value is not None:
                args[arg.name] = value
        return args

    @classmethod
    def _fallback_value(cls, arg: ToolArgument, tool_name: str) -> Any:
        """Safe fallback value generation — only when no intent template exists."""
        name_lower = arg.name.lower()

        # Path args: only for file tools, not bash/execution
        if ("path" in name_lower or "file" in name_lower or "directory" in name_lower):
            if tool_name in ("File_Read", "File_Write", "File_Search", "File_List",
                            "File_Delete", "File_Copy", "Search_Replace", "Search_Code",
                            "Python_Test"):
                return cls._generate_path(arg.name)
            return None

        if "command" in name_lower:
            return cls._generate_command(tool_name)

        if "code" in name_lower or "script" in name_lower:
            return cls._generate_code(tool_name)

        if "query" in name_lower or "search" in name_lower or "pattern" in name_lower:
            return random.choice([
                "Python async patterns", "React hooks guide", "Docker compose",
                "Git workflow", "TypeScript generics", "REST API design",
            ])

        if "message" in name_lower or "commit" in name_lower:
            prefixes = ["fix:", "feat:", "refactor:", "docs:", "chore:", "test:"]
            return f"{random.choice(prefixes)} {random.choice(['add feature', 'fix bug', 'update docs'])}"

        if arg.type == "boolean":
            return random.choice([True, False])

        if arg.type == "integer":
            return cls._generate_int(arg.name, arg)

        if arg.enum_values:
            return random.choice(arg.enum_values)

        # Never generate fake/sample strings — only set if required
        if arg.required:
            if arg.type == "string":
                if "url" in name_lower:
                    return "https://example.com"
                return None

        if arg.type == "array":
            return [".py", ".js"] if "file_type" in name_lower else []

        return None

    @classmethod
    def _generate_path(cls, name: str) -> str:
        name_lower = name.lower()
        if "config" in name_lower:
            return random.choice(cls.REALISTIC_PATHS["config"])
        elif "source" in name_lower or "src" in name_lower:
            return random.choice(cls.REALISTIC_PATHS["source"])
        elif "test" in name_lower:
            return random.choice(cls.REALISTIC_PATHS["test"])
        elif "output" in name_lower or "destination" in name_lower:
            return random.choice(cls.REALISTIC_PATHS["output"])
        return random.choice(cls.REALISTIC_PATHS["source"])

    @classmethod
    def _generate_command(cls, name: str) -> str:
        name_lower = name.lower()
        if "git" in name_lower:
            return random.choice(cls.REALISTIC_COMMANDS["git"])
        elif "npm" in name_lower or "node" in name_lower:
            return random.choice(cls.REALISTIC_COMMANDS["npm"])
        elif "docker" in name_lower:
            return random.choice(cls.REALISTIC_COMMANDS["docker"])
        return random.choice(cls.REALISTIC_COMMANDS["system"])

    @classmethod
    def _generate_code(cls, name: str) -> str:
        name_lower = name.lower()
        if "python" in name_lower or "py" in name_lower:
            return random.choice(cls.REALISTIC_CODE["python"])
        elif "javascript" in name_lower or "js" in name_lower:
            return random.choice(cls.REALISTIC_CODE["javascript"])
        return random.choice(cls.REALISTIC_CODE["python"])

    @classmethod
    def _generate_int(cls, name: str, arg: ToolArgument) -> int:
        name_lower = name.lower()
        if "limit" in name_lower or "num_results" in name_lower or "count" in name_lower:
            return random.randint(1, min(arg.max_value or 50, 50))
        elif "offset" in name_lower or "skip" in name_lower:
            return random.randint(0, 100)
        elif "timeout" in name_lower:
            return random.choice([10, 30, 60])
        return random.randint(1, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATORS (15% Error Rate)
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Generate realistic tool responses including errors."""

    ERROR_TYPES = {
        "file_not_found": {"error": "FileNotFoundError: [Errno 2] No such file or directory: '{path}'", "code": "ENOENT"},
        "permission_denied": {"error": "PermissionError: [Errno 13] Permission denied: '{path}'", "code": "EACCES"},
        "is_directory": {"error": "IsADirectoryError: expected file, got directory: '{path}'", "code": "EISDIR"},
        "invalid_json": {"error": "JSONDecodeError: Expecting value: line 1 column 1", "code": "JSON_ERROR"},
        "timeout": {"error": "TimeoutError: Command timed out after {timeout} seconds", "code": "ETIMEDOUT"},
        "syntax_error": {"error": "SyntaxError: invalid syntax", "code": "SYNTAX_ERROR"},
        "network_error": {"error": "ConnectionError: Failed to connect to host", "code": "ECONNREFUSED"},
        "git_not_repo": {"error": "Fatal: not a git repository (or any of the parent directories): .git", "code": "NOT_GIT_REPO"},
        "empty_result": {"error": "No results found", "code": "EMPTY_RESULT"},
    }

    SUCCESS_TEMPLATES = {
        "File_Read": {"template": {"path": "{file_path}", "bytes_read": 1234, "lines_read": 42, "content_preview": "# Sample file...\nLine 1\nLine 2"}},
        "File_Write": {"template": {"path": "{file_path}", "bytes_written": 567, "lines_written": 12}},
        "File_List": {"template": {"entries": [{"name": "src", "type": "directory"}, {"name": "README.md", "type": "file"}, {"name": "main.py", "type": "file"}], "total": 3}},
        "File_Search": {"template": {"matches": ["src/main.py", "src/utils.py", "tests/test_main.py"], "total": 3}},
        "File_Delete": {"template": {"path": "{path}", "deleted": True}},
        "File_Copy": {"template": {"source": "{source}", "destination": "{destination}", "copied": True}},
        "Bash_Execute": {"template": {"stdout": "Executed: {command}\n", "stderr": "", "exit_code": 0}},
        "Git_Status": {"template": {"branch": "main", "is_dirty": True, "staged": ["README.md"], "modified": ["src/main.py"], "untracked": ["new.py"]}},
        "Git_Commit": {"template": {"branch": "main", "commit_hash": "abc1234", "message": "{message}", "files_changed": 2}},
        "Git_Log": {"template": {"commits": [{"hash": "abc1234", "message": "fix: resolve bug", "author": "Dev <dev@example.com>"}, {"hash": "def5678", "message": "feat: add feature", "author": "Dev <dev@example.com>"}]}},
        "Git_Branch": {"template": {"current": "main", "branches": ["main", "develop", "feature/x"]}},
        "Git_Diff": {"template": {"files": [{"path": "src/main.py", "additions": 5, "deletions": 2}]}},
        "Git_Push": {"template": {"remote": "{remote}", "branch": "{branch}", "pushed": True}},
        "Git_Pull": {"template": {"remote": "{remote}", "branch": "{branch}", "files_updated": 3, "insertions": 45}},
        "Bash_ShellStatus": {"template": {"os": "Linux x86_64", "shell": "/bin/bash", "home": "/home/sridhar", "cwd": "/home/sridhar/beastcli-eng1", "user": "sridhar"}},
        "Python_Test": {"template": {"file_path": "{file_path}", "tests_run": 5, "passed": 5, "failed": 0, "skipped": 0, "exit_code": 0}},
        "Web_Search": {"template": {"results": [{"title": "Result Title", "url": "https://example.com", "snippet": "A relevant article about..."}], "total": 5}},
        "Web_Fetch": {"template": {"url": "{url}", "status": 200, "content_length": 2048, "content_type": "text/html"}},
        "Web_Screenshot": {"template": {"url": "{url}", "captured": True, "width": 1920, "height": 1080}},
        "Python_Run": {"template": {"stdout": "Hello, World!\n", "stderr": "", "return_value": None, "exit_code": 0}},
        "Node_Run": {"template": {"stdout": "Hello, World!\n", "stderr": "", "exit_code": 0}},
        "Search_Code": {"template": {"matches": [{"file": "src/main.py", "line": 42, "context": "def main():"}], "total": 1}},
        "Search_Replace": {"template": {"file": "{path}", "replacements": 1, "changed": True}},
        "System_Info": {"template": {"os": "Linux x86_64", "python_version": "3.11.0", "cpu_count": 8, "memory_total_gb": 32}},
        "Process_List": {"template": {"processes": [{"pid": 1234, "name": "python", "cpu": "2.5%"}, {"pid": 5678, "name": "node", "cpu": "1.2%"}], "total": 2}},
        "Database_Query": {"template": {"rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2, "columns": ["id", "name"]}},
        "Database_List": {"template": {"databases": ["production", "test_db"], "total": 2}},
    }

    @classmethod
    def generate_response(cls, tool: ToolSchema, args: dict, success: bool = True) -> str:
        if success:
            return cls._generate_success(tool, args)
        return cls._generate_error(tool, args)

    @classmethod
    def _generate_success(cls, tool: ToolSchema, args: dict) -> str:
        # Return industry-standard tool_result format
        if tool.name == "Bash_ShellStatus":
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "shell": "/bin/zsh",
                    "user": "sridhar",
                    "home_directory": "/Users/sridhar",
                    "current_directory": "/Users/sridhar/project",
                    "platform": "macOS",
                }),
            }
            return json.dumps(result)
        if tool.name == "Python_Run":
            code = args.get("code", "")
            stdout = "Hello, World!\n"
            if "fibonacci(30)" in code:
                stdout = "832040\n"
            elif "fibonacci" in code:
                stdout = "55\n"
            elif "sum(range(1, 101))" in code:
                stdout = "5050\n"
            elif "csv.DictReader" in code:
                stdout = "{'row_count': 2, 'total': 3}\n"
            elif "json.loads" in code:
                stdout = "{'ok': True}\n"
            elif "urllib.request.urlopen" in code:
                stdout = "<!doctype html><html><head><title>Example Domain</title></head><body>Example Domain</body></html>\n"
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "stdout": stdout,
                    "stderr": "",
                    "return_value": None,
                    "exit_code": 0,
                }),
            }
            return json.dumps(result)
        if tool.name == "Node_Run":
            code = args.get("code", "")
            if "readfilesync('readme.md'" in code.lower():
                stdout = "# Project Setup\n\nInstall dependencies and run the CLI.\n"
            else:
                stdout = "Hello from Node!\n" if "Hello from Node" in code else "[ 2, 4, 6 ]\n"
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "stdout": stdout,
                    "stderr": "",
                    "exit_code": 0,
                }),
            }
            return json.dumps(result)
        if tool.name == "Web_Fetch":
            url = args.get("url", "")
            output = {
                "url": url,
                "status": 200,
                "content_length": 2048,
                "content_type": "text/html",
            }
            if "readme" in url.lower():
                output["summary"] = "README covers installation, usage, and development commands."
            elif "releases" in url.lower():
                output["summary"] = "Latest release notes mention bug fixes, dependency updates, and CLI improvements."
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps(output),
            }
            return json.dumps(result)
        if tool.name == "Git_Log":
            limit = min(max(int(args.get("limit", 5) or 5), 1), 10)
            commits = [
                {"hash": "abc1234", "message": "fix: resolve bug", "author": "Dev <dev@example.com>"},
                {"hash": "def5678", "message": "feat: add feature", "author": "Dev <dev@example.com>"},
                {"hash": "7890abc", "message": "docs: refresh README", "author": "Dev <dev@example.com>"},
                {"hash": "fedc321", "message": "test: add API coverage", "author": "Dev <dev@example.com>"},
                {"hash": "13579bd", "message": "refactor: simplify parser", "author": "Dev <dev@example.com>"},
                {"hash": "2468ace", "message": "chore: bump dependencies", "author": "Dev <dev@example.com>"},
                {"hash": "1122aabb", "message": "feat: add CLI flag", "author": "Dev <dev@example.com>"},
                {"hash": "3344ccdd", "message": "fix: handle empty config", "author": "Dev <dev@example.com>"},
                {"hash": "5566eeff", "message": "perf: speed up scans", "author": "Dev <dev@example.com>"},
                {"hash": "7788gghh", "message": "build: update workflow", "author": "Dev <dev@example.com>"},
            ][:limit]
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"commits": commits}),
            }
            return json.dumps(result)
        if tool.name == "File_List":
            directory = args.get("directory", ".")
            if directory == "config" or directory.endswith("/config"):
                entries = [
                    {"name": "config.json", "type": "file"},
                    {"name": "settings.yaml", "type": "file"},
                ]
            elif directory == "src" or directory.endswith("/src"):
                entries = [
                    {"name": "app.py", "type": "file"},
                    {"name": "main.py", "type": "file"},
                    {"name": ".env.example", "type": "file"},
                ]
            else:
                entries = [
                    {"name": "src", "type": "directory"},
                    {"name": "config", "type": "directory"},
                    {"name": "README.md", "type": "file"},
                    {"name": "main.py", "type": "file"},
                ]
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"entries": entries, "total": len(entries)}),
            }
            return json.dumps(result)
        if tool.name == "Bash_Execute":
            command = args.get("command", "")
            stdout = f"Executed: {command}\n"
            if command == "pwd":
                stdout = "/Users/sridhar/project\n"
            elif "ps aux --sort=-rss | head -5" in command:
                stdout = (
                    "USER PID %CPU %MEM COMMAND\n"
                    "sridhar 4101 3.2 12.8 node\n"
                    "sridhar 3880 1.4 9.1 python3\n"
                    "sridhar 2450 0.8 5.7 chrome\n"
                    "sridhar 1780 0.3 4.2 code\n"
                )
            elif "git status --porcelain" in command:
                stdout = " M src/main.py\n?? new.py\n"
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "stdout": stdout,
                    "stderr": "",
                    "exit_code": 0,
                }),
            }
            return json.dumps(result)
        if tool.name == "File_Search":
            pattern = args.get("pattern", "*")
            if pattern == "*.md" and args.get("path") == "docs" and args.get("content_search"):
                matches = ["docs/setup.md"]
            elif pattern == "*.md" and args.get("path") == "docs":
                matches = ["docs/setup.md", "docs/architecture.md"]
            elif args.get("content_search"):
                matches = ["README.md", "docs/setup.md"]
            else:
                matches = {
                    "*.md": ["README.md", "docs/setup.md", "docs/architecture.md"],
                    "*.py": ["src/main.py", "src/utils.py", "tests/test_main.py"],
                    "*.js": ["src/app.js", "src/utils.js"],
                    "*.{py,js}": ["src/main.py", "src/utils.py", "src/app.js", "src/utils.js"],
                    "test_*.py": ["tests/test_main.py", "tests/test_auth.py"],
                    "*config*": ["config", "config.json", "settings/config.dev.json"],
                }.get(pattern, ["src/main.py"])
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"matches": matches, "total": len(matches)}),
            }
            return json.dumps(result)
        if tool.name == "Git_Pull":
            remote = args.get("remote", "origin")
            branch = args.get("branch", "main")
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "remote": remote,
                    "branch": branch,
                    "files_updated": 3,
                    "insertions": 45,
                }),
            }
            return json.dumps(result)
        if tool.name == "Python_Test":
            coverage_enabled = args.get("coverage", False)
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({
                    "passed": 24,
                    "failed": 0,
                    "pattern": args.get("pattern", "test_*.py"),
                    "verbose": args.get("verbose", False),
                    "coverage": {"enabled": coverage_enabled, "percent": 87} if coverage_enabled else None,
                    "exit_code": 0,
                }),
            }
            return json.dumps(result)
        if tool.name == "Search_Code":
            pattern = args.get("pattern", "")
            matches = [{
                "file": "src/main.py",
                "line": 42,
                "context": "def main():",
            }]
            if pattern == "TODO":
                matches = [{
                    "file": "src/tasks.py",
                    "line": 18,
                    "context": "# TODO: replace mock client with real implementation",
                }]
            elif pattern == "import\\s+":
                matches = [{
                    "file": "src/main.py",
                    "line": 3,
                    "context": "import argparse",
                }]
            elif pattern == "def main":
                matches = [
                    {"file": "src/main.py", "line": 42, "context": "def main():"},
                    {"file": "tests/test_main.py", "line": 9, "context": "def main():"},
                ]
            elif pattern == "try:":
                matches = [
                    {"file": "src/api.py", "line": 27, "context": "try:"},
                    {"file": "src/worker.py", "line": 54, "context": "try:"},
                ]
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"matches": matches, "total": len(matches)}),
            }
            return json.dumps(result)
        if tool.name == "Database_Query":
            query = args.get("query", "")
            rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
            columns = ["id", "name"]
            if "orders" in query.lower():
                rows = [
                    {"order_id": 101, "status": "paid", "created_at": "2026-04-12"},
                    {"order_id": 102, "status": "shipped", "created_at": "2026-04-18"},
                ]
                columns = ["order_id", "status", "created_at"]
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"rows": rows, "total": len(rows), "columns": columns}),
            }
            return json.dumps(result)
        if tool.name == "Git_Branch":
            operation = args.get("operation", "list")
            branches = ["main", "develop", "feature/x"]
            branch_name = args.get("branch_name")
            current = "main"
            if operation == "create" and branch_name and branch_name not in branches:
                branches.append(branch_name)
            if operation == "switch" and branch_name:
                current = branch_name
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps({"current": current, "branches": branches, "operation": operation}),
            }
            return json.dumps(result)

        template = cls.SUCCESS_TEMPLATES.get(tool.name)
        if template:
            response = template["template"].copy()
            for key, value in args.items():
                for resp_key, resp_value in response.items():
                    if isinstance(resp_value, str) and "{" + key + "}" in resp_value:
                        response[resp_key] = resp_value.replace("{" + key + "}", str(value))
            # Guard: any remaining {placeholder} gets replaced with arg value or default
            for resp_key, resp_value in response.items():
                if isinstance(resp_value, str):
                    for ph in re.findall(r"\{([a-zA-Z_]+)\}", resp_value):
                        response[resp_key] = resp_value.replace(
                            "{" + ph + "}",
                            str(args.get(ph, "/path/to/file"))
                        )
            result = {
                "type": "tool_result",
                "tool_call_id": "{{TOOL_CALL_ID}}",
                "output": json.dumps(response),
            }
            return json.dumps(result)
        return json.dumps({
            "type": "tool_result",
            "tool_call_id": "{{TOOL_CALL_ID}}",
            "output": json.dumps({"result": f"{tool.name} completed", "success": True}),
        })

    @classmethod
    def _generate_error(cls, tool: ToolSchema, args: dict) -> str:
        tool_specific_errors = {
            "Bash_ShellStatus": ["timeout"],
            "Bash_Execute": ["timeout", "permission_denied"],
            "Python_Run": ["timeout", "syntax_error"],
            "Node_Run": ["timeout", "syntax_error"],
            "Python_Test": ["timeout", "empty_result"],
            "File_List": ["file_not_found", "permission_denied"],
            "File_Read": ["file_not_found", "permission_denied"],
            "File_Write": ["permission_denied"],
            "File_Delete": ["file_not_found", "permission_denied"],
            "File_Copy": ["file_not_found", "permission_denied"],
            "File_Search": ["empty_result"],
            "Search_Code": ["empty_result"],
            "Search_Replace": ["empty_result"],
            "Web_Search": ["network_error"],
            "Web_Fetch": ["network_error"],
            "Web_Screenshot": ["network_error"],
            "Git_Status": ["git_not_repo"],
            "Git_Log": ["git_not_repo"],
            "Git_Commit": ["git_not_repo"],
            "Git_Branch": ["git_not_repo"],
            "Git_Diff": ["git_not_repo"],
            "Git_Pull": ["git_not_repo", "network_error"],
            "Git_Push": ["git_not_repo", "network_error"],
            "System_Info": ["timeout"],
            "Process_List": ["timeout"],
            "Database_Query": ["network_error", "syntax_error"],
            "Database_List": ["network_error"],
        }
        error_type = random.choice(tool_specific_errors.get(tool.name, ["timeout"]))
        if tool.name in {"Python_Run", "Node_Run"} and error_type == "syntax_error":
            code = args.get("code", "")
            likely_valid = any(
                token in code for token in (
                    "print(",
                    "console.log(",
                    "urllib.request.urlopen",
                    "csv.DictReader",
                    "json.loads",
                    "fibonacci(",
                    ".map(",
                    "readFileSync(",
                )
            )
            if likely_valid:
                error_type = "timeout"
        error_info = cls.ERROR_TYPES[error_type]
        error_message = error_info["error"]
        replacements = {
            "path": args.get("path") or args.get("file_path") or ".",
            "timeout": args.get("timeout", 30),
            "remote": args.get("remote", "origin"),
            "branch": args.get("branch", "main"),
        }
        for key, value in replacements.items():
            error_message = error_message.replace("{" + key + "}", str(value))
        for key, value in args.items():
            error_message = error_message.replace("{" + key + "}", str(value))
        # Guard: any remaining {placeholder} gets a default value
        for ph in re.findall(r"\{([a-zA-Z_]+)\}", error_message):
            error_message = error_message.replace("{" + ph + "}", str(args.get(ph, "?")))
        return json.dumps({
            "type": "tool_result",
            "tool_call_id": "{{TOOL_CALL_ID}}",
            "output": "",
            "error": error_message,
            "exit_code": 1,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# USER QUERY TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class QueryTemplates:
    """Natural language query templates for each tool."""

    TEMPLATES = {
        "File_Read": {
            DifficultyLevel.EASY: [
                "Read the file at config.json",
                "Show me what's in main.py",
                "Can you open and display src/app.py?",
                "What does the setup.py file contain?",
            ],
            DifficultyLevel.MEDIUM: [
                "Can you read the first 50 lines of this file?",
                "What's in the README?",
                "Show me the contents of the lib/utils.py file",
            ],
            DifficultyLevel.HARD: [
                "Read the configuration file and tell me what the database connection string is",
                "Find and read the main entry point of the project",
            ],
            DifficultyLevel.EXPERT: [
                "Read src/main.py and summarize what it does",
            ],
        },
        "File_Write": {
            DifficultyLevel.EASY: [
                "Write this content to output.txt",
                "Create a new file called test.py with the content: print('Hello')",
                "Save the following to data.json: {\"key\": \"value\"}",
            ],
            DifficultyLevel.MEDIUM: [
                "Append this log entry to the existing file",
                "Create README.md with installation instructions",
            ],
            DifficultyLevel.HARD: [
                "Create a new Python module at src/models.py with a User class",
            ],
            DifficultyLevel.EXPERT: [
                "Write a complete CLI tool to output/logic.py with argument parsing",
            ],
        },
        "File_List": {
            DifficultyLevel.EASY: [
                "List the files in the current directory",
                "What's in the current directory?",
                "Show me the project structure",
            ],
            DifficultyLevel.MEDIUM: [
                "List all files in the src directory including hidden ones",
                "What files are in the config folder?",
            ],
            DifficultyLevel.HARD: [
                "List all files recursively in the project and show me the tree structure",
            ],
            DifficultyLevel.EXPERT: [
                "List and categorize all files by type (source, test, config, docs)",
            ],
        },
        "File_Search": {
            DifficultyLevel.EASY: [
                "Find all Python files in this project",
                "Search for *.json files in the config directory",
            ],
            DifficultyLevel.MEDIUM: [
                "Find all files containing the word 'config' in their name",
                "Search for test files matching the pattern test_*.py",
            ],
            DifficultyLevel.HARD: [
                "Find all markdown documentation files and tell me which ones mention 'setup'",
            ],
            DifficultyLevel.EXPERT: [
                "Search recursively for all files with .py and .js extensions and group by directory",
            ],
        },
        "File_Delete": {
            DifficultyLevel.EASY: [
                "Delete the temporary file /tmp/cache.log",
                "Remove output.json from the current directory",
            ],
            DifficultyLevel.MEDIUM: [
                "Delete the backup file backup_old.py",
            ],
            DifficultyLevel.HARD: [
                "Clean up all .pyc files in the build directory",
            ],
        },
        "File_Copy": {
            DifficultyLevel.EASY: [
                "Copy README.md to README_backup.md",
                "Duplicate the config file as config.old",
            ],
            DifficultyLevel.MEDIUM: [
                "Copy the entire src directory to src_backup",
            ],
        },
        "Bash_Execute": {
            DifficultyLevel.EASY: [
                "Run `ls -la` in the current directory",
                "Check disk usage with df -h",
                "Show me the current working directory",
            ],
            DifficultyLevel.MEDIUM: [
                "Find all Python files modified in the last 7 days",
                "Count the number of lines in all .py files",
                "Show me the top 5 processes by memory usage",
            ],
            DifficultyLevel.HARD: [
                "Run the test suite and tell me if all tests pass",
                "Install dependencies from requirements.txt and verify",
            ],
            DifficultyLevel.EXPERT: [
                "Analyze the codebase, find all TODO comments, and summarize them",
            ],
        },
        "Bash_ShellStatus": {
            DifficultyLevel.EASY: [
                "What is the current shell environment?",
                "Show me the current OS, user, and working directory",
            ],
            DifficultyLevel.MEDIUM: [
                "What shell am I using and what is my home directory?",
            ],
        },
        "Git_Status": {
            DifficultyLevel.EASY: [
                "What's the current git status?",
                "Show me uncommitted changes",
                "Are there any changes in the repository?",
            ],
            DifficultyLevel.MEDIUM: [
                "Are there any files that need to be committed?",
                "Check what files have been modified since the last commit",
            ],
        },
        "Git_Commit": {
            DifficultyLevel.EASY: [
                "Commit the changes with message 'fix: resolve bug'",
                "Stage and commit all changes with the message 'update: modify config'",
            ],
            DifficultyLevel.MEDIUM: [
                "Commit all changes with a descriptive message following conventional commits",
            ],
            DifficultyLevel.HARD: [
                "Stage the modified files and commit with a properly formatted commit message",
            ],
        },
        "Git_Log": {
            DifficultyLevel.EASY: [
                "Show me the last 5 commits",
                "Show me the commit history",
            ],
            DifficultyLevel.MEDIUM: [
                "Display the last 10 commits in short format",
                "Show me the commit history with dates and authors",
            ],
        },
        "Git_Branch": {
            DifficultyLevel.EASY: [
                "List all branches in this repository",
                "What branches exist?",
            ],
            DifficultyLevel.MEDIUM: [
                "Create a new branch called feature/login",
                "Switch to the develop branch",
            ],
            DifficultyLevel.HARD: [
                "Delete the old feature branch that is fully merged",
            ],
        },
        "Git_Diff": {
            DifficultyLevel.EASY: [
                "Show me the changes in src/main.py",
                "What's different from the last commit?",
            ],
            DifficultyLevel.MEDIUM: [
                "Show all unstaged changes across the entire project",
            ],
            DifficultyLevel.HARD: [
                "Compare the current branch with main and show me the summary",
            ],
        },
        "Git_Push": {
            DifficultyLevel.EASY: [
                "Push commits to origin main",
                "Push my changes to the remote repository",
            ],
            DifficultyLevel.MEDIUM: [
                "Push the develop branch to the origin remote",
            ],
        },
        "Git_Pull": {
            DifficultyLevel.EASY: [
                "Pull the latest changes from origin main",
                "Update my local branch with remote changes",
            ],
        },
        "Web_Search": {
            DifficultyLevel.EASY: [
                "Search for Python best practices 2026",
                "Find information about React performance optimization",
            ],
            DifficultyLevel.MEDIUM: [
                "Search for articles about async/await patterns in Python",
                "Find documentation on TypeScript generics",
            ],
            DifficultyLevel.HARD: [
                "Search for recent papers or articles on LLM fine-tuning techniques",
            ],
            DifficultyLevel.EXPERT: [
                "Research best practices for building production-grade CLI tools in Rust",
            ],
        },
        "Web_Fetch": {
            DifficultyLevel.EASY: [
                "Fetch the README from github.com/example/repo",
                "Get the content of https://api.example.com/status",
            ],
            DifficultyLevel.MEDIUM: [
                "Fetch the latest release notes from the project documentation",
            ],
        },
        "Web_Screenshot": {
            DifficultyLevel.EASY: [
                "Take a screenshot of https://example.com",
                "Take a screenshot of the documentation homepage",
            ],
        },
        "Python_Run": {
            DifficultyLevel.EASY: [
                "Run this Python code: print('Hello, World!')",
                "Execute: result = sum(range(1, 101)); print(result)",
            ],
            DifficultyLevel.MEDIUM: [
                "Calculate fibonacci(30) using recursion",
                "Run a Python script that parses a JSON file",
            ],
            DifficultyLevel.HARD: [
                "Execute a Python script that makes an HTTP request and prints the response",
            ],
            DifficultyLevel.EXPERT: [
                "Run a data processing pipeline that reads CSV, transforms data, and outputs statistics",
            ],
        },
        "Node_Run": {
            DifficultyLevel.EASY: [
                "Run this JavaScript: console.log('Hello from Node!')",
                "Execute: const arr = [1,2,3].map(x => x * 2); console.log(arr);",
            ],
            DifficultyLevel.MEDIUM: [
                "Run a Node script that reads README.md and prints its contents",
            ],
        },
        "Python_Test": {
            DifficultyLevel.EASY: [
                "Run pytest on tests/test_main.py",
                "Execute the test suite for the auth module",
            ],
            DifficultyLevel.MEDIUM: [
                "Run all tests matching the pattern test_api_*.py",
            ],
            DifficultyLevel.HARD: [
                "Run tests with verbose output and show me the coverage report",
            ],
        },
        "Search_Code": {
            DifficultyLevel.EASY: [
                "Find all occurrences of 'TODO' in the codebase",
                "Search for 'def main' in all Python files",
            ],
            DifficultyLevel.MEDIUM: [
                "Find lines containing 'import' statements across the project",
            ],
            DifficultyLevel.HARD: [
                "Search for all error handling patterns (try/except blocks) in the code",
            ],
        },
        "Search_Replace": {
            DifficultyLevel.EASY: [
                "Replace all occurrences of 'foo' with 'bar' in src/main.py",
            ],
            DifficultyLevel.MEDIUM: [
                "In the config.json file, replace the timeout value from 30 to 60",
            ],
            DifficultyLevel.HARD: [
                "Update all import statements from 'utils' to 'helpers' across the project",
            ],
        },
        "System_Info": {
            DifficultyLevel.EASY: [
                "Show me the OS details",
                "Show me the memory details",
            ],
            DifficultyLevel.MEDIUM: [
                "Show me the CPU details and tell me how many cores are available",
            ],
        },
        "Process_List": {
            DifficultyLevel.EASY: [
                "Show me the running processes",
                "List all active processes",
            ],
            DifficultyLevel.MEDIUM: [
                "Show me the top 10 processes by CPU usage",
            ],
        },
        "Database_Query": {
            DifficultyLevel.EASY: [
                "Run the query: SELECT * FROM users LIMIT 10",
                "Execute: SELECT COUNT(*) FROM orders",
            ],
            DifficultyLevel.MEDIUM: [
                "Query the database for all orders placed in the last 30 days",
            ],
        },
        "Database_List": {
            DifficultyLevel.EASY: [
                "List all databases on this server",
                "Show me all available databases",
            ],
        },
    }

    @classmethod
    def get_query(cls, tool: ToolSchema, difficulty: DifficultyLevel, loc: Localization) -> str:
        lang = Language(loc.language) if loc.language in [l.value for l in Language] else Language.EN
        diff = DifficultyLevel(difficulty) if difficulty.value in [d.value for d in DifficultyLevel] else DifficultyLevel.EASY
        tool_templates = cls.TEMPLATES.get(tool.name, {})
        diff_templates = tool_templates.get(diff, tool_templates.get(DifficultyLevel.EASY, [f"Execute {tool.name}"]))
        query = random.choice(diff_templates)
        if loc.humanize:
            query = Humanizer.humanize(query, loc)
        return query


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SystemPromptGenerator:
    """Generate system prompts with localization as SUPREME."""

    @classmethod
    def generate(cls, loc: Localization, tool_count: int = 27) -> str:
        # Minimal system prompt — tools list provided via tools field, not here
        return (
            f"You are an expert CLI assistant with access to {tool_count} tools. "
            f"Respond in language: {loc.language}, tone: {loc.tone}, formality: {loc.formality}. "
            f"When a user asks you to do something, call the appropriate tool(s) with correct arguments. "
            f"After getting tool results, provide a clear, specific final answer."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    Chat message in the conversation.

    IMPORTANT: tool_calls do NOT include 'id' field.
    Tool call IDs are SYSTEM-GENERATED at execution time.
    Including 'id' in tool_calls would teach models to hallucinate IDs.
    """
    role: str  # system, user, assistant, tool
    content: str | None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None  # Only in tool responses
    name: str | None = None          # Only in tool responses

    def to_dict(self) -> dict:
        """
        Unsloth-compatible format:
        - Assistant messages have content as JSON string (tool_call or final_answer)
        - Tool messages have content as JSON string (tool_result) + tool_call_id
        """
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        # tool_call_id ONLY in tool responses (system-generated)
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class DatasetExample:
    """A complete training example. Format compatible with Unsloth fine-tuning."""
    messages: list[Message]
    localization: Localization
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "localization": self.localization.to_dict(),
            "tools": [t.to_openai_format() for t in ToolRegistry.get_all_tools()],
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetValidator:
    """Validate dataset examples against best practices."""

    PATH_KEYS = {
        "file_path", "path", "directory", "source", "destination", "repository_path"
    }

    @classmethod
    def validate_example(cls, example: DatasetExample) -> tuple[bool, list[str]]:
        errors = []
        if not example.messages or len(example.messages) < 3:
            errors.append("Insufficient messages (need at least 3: system, user, assistant)")
        if not example.localization:
            errors.append("Missing localization configuration")
        if example.messages and example.messages[0].role != "system":
            errors.append("First message must be 'system' role")

        user_text = " ".join(
            (msg.content or "") for msg in example.messages if msg.role == "user"
        ).lower()

        # Check assistant messages have valid JSON content (tool_call or final_answer)
        for msg in example.messages:
            if msg.role == "assistant" and msg.content:
                try:
                    obj = json.loads(msg.content)
                    if not isinstance(obj, dict) or "type" not in obj:
                        errors.append(f"Assistant content missing 'type' field: {msg.content[:80]}")
                    elif obj["type"] not in ("tool_call", "final_answer"):
                        errors.append(f"Assistant content has invalid type: {obj['type']}")
                    if obj.get("type") == "tool_call":
                        if not obj.get("tool_name"):
                            errors.append(f"tool_call missing 'tool_name': {msg.content[:80]}")
                        if "arguments" not in obj:
                            errors.append(f"tool_call missing 'arguments': {msg.content[:80]}")
                        if "tool_call_id" in obj or "id" in obj:
                            errors.append(f"tool_call contains system-generated id field: {msg.content[:80]}")
                        valid_names = ToolRegistry.get_tool_names()
                        if obj.get("tool_name") not in valid_names:
                            errors.append(f"Invalid tool name: {obj.get('tool_name')}")
                        else:
                            schema = next(t for t in ToolRegistry.get_all_tools() if t.name == obj["tool_name"])
                            args = obj.get("arguments", {})
                            allowed = {arg.name for arg in schema.arguments}
                            required = {arg.name for arg in schema.arguments if arg.required}
                            missing = sorted(required - set(args.keys()))
                            extra = sorted(set(args.keys()) - allowed)
                            if missing:
                                errors.append(f"tool_call missing required arguments: {obj['tool_name']} -> {missing}")
                            if extra:
                                errors.append(f"tool_call has unexpected arguments: {obj['tool_name']} -> {extra}")
                            for key, value in args.items():
                                if (
                                    key in cls.PATH_KEYS and
                                    isinstance(value, str) and
                                    value.startswith("/Users/sridhar/project") and
                                    not any(token in user_text for token in (
                                        "/users/", "absolute path", "full path",
                                        "current directory", "working directory",
                                        "project root", "repository root"
                                    )) and
                                    not value.replace("/Users/sridhar/project/", "") in user_text
                                ):
                                    errors.append(f"tool_call uses ungrounded absolute path: {obj['tool_name']} -> {key}={value}")
                    if obj.get("type") == "final_answer":
                        if not obj.get("content") or len(obj.get("content", "").strip()) < 5:
                            errors.append(f"final_answer too short/generic: {msg.content[:80]}")
                        generic = {
                            "operation completed successfully.",
                            "task finished. all steps completed.",
                            "task completed successfully.",
                        }
                        if obj.get("content", "").strip().lower() in generic or "executed without errors" in obj.get("content", "").lower():
                            errors.append(f"final_answer too generic: {msg.content[:80]}")
                except json.JSONDecodeError:
                    errors.append(f"Assistant content not valid JSON: {msg.content[:80]}")

        # Check tool messages have valid JSON with type=tool_result
        for msg in example.messages:
            if msg.role == "tool" and msg.content:
                try:
                    obj = json.loads(msg.content)
                    if obj.get("type") != "tool_result":
                        errors.append(f"Tool content has wrong type: {msg.content[:80]}")
                    if not obj.get("tool_call_id"):
                        errors.append(f"tool_result missing 'tool_call_id': {msg.content[:80]}")
                    if "output" not in obj and "error" not in obj:
                        errors.append(f"tool_result missing 'output' or 'error': {msg.content[:80]}")
                    serialized = json.dumps(obj, ensure_ascii=False)
                    if any(token in serialized for token in ("{file_path}", "/path/to/copy", "command output here")):
                        errors.append(f"tool_result contains placeholder content: {msg.content[:80]}")
                except json.JSONDecodeError:
                    errors.append(f"Tool content not valid JSON: {msg.content[:80]}")

        return len(errors) == 0, errors

    @classmethod
    def validate_batch(cls, examples: list[DatasetExample]) -> tuple[list[DatasetExample], dict]:
        valid = []
        stats = {"total": len(examples), "valid": 0, "invalid": 0, "errors_by_type": {}}

        for i, ex in enumerate(examples):
            is_valid, errors = cls.validate_example(ex)
            if is_valid:
                valid.append(ex)
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                for error in errors:
                    error_type = error.split(":")[0] if ":" in error else error
                    stats["errors_by_type"][error_type] = stats["errors_by_type"].get(error_type, 0) + 1

        return valid, stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class ComprehensiveDatasetPipeline:
    """
    Production-grade dataset generation pipeline.

    Based on TOUCAN, ToolACE, and APIGen methodologies:
    - Multi-stage generation
    - Quality filtering
    - Diverse localization
    - Curriculum learning
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.tools = ToolRegistry.get_all_tools()
        self.validator = DatasetValidator()

    def generate_single(
        self,
        localization: Localization,
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        include_error: bool = False,
        multi_tool: bool = False,
    ) -> DatasetExample:
        """Generate a training example in Unsloth-compatible format.

        Message schema:
        - system: System prompt
        - user: User request
        - assistant: {"type":"tool_call","tool_name":"...","arguments":{...}}
        - tool: {"type":"tool_result","tool_call_id":"...","output":"..."}
        - (optionally more tool_call/tool pairs for multi_tool samples)
        - assistant: {"type":"final_answer","content":"..."}

        Multi-tool samples (multi_tool=True) generate 2-3 related tool calls
        per sample to reduce structural duplication and teach tool chaining.
        """
        # Keep samples tightly aligned to the user request. Multi-tool support
        # should only be re-enabled once explicit chain templates exist.
        selected_tools = [random.choice(self.tools)]

        system_prompt = SystemPromptGenerator.generate(localization, len(self.tools))
        user_query = QueryTemplates.get_query(selected_tools[0], difficulty, localization)

        # Build messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_query),
        ]

        all_success = True
        tool_results_data = []

        for tool in selected_tools:
            args = ToolCallGenerator.generate_arguments(tool, user_query)
            success = not include_error or random.random() > 0.15
            if not success:
                all_success = False

            system_call_id = "call_%s" % uuid.uuid4().hex[:12]
            tool_call_content = json.dumps({
                "type": "tool_call",
                "tool_name": tool.name,
                "arguments": args,
            }, ensure_ascii=False)

            tool_response = ResponseGenerator.generate_response(tool, args, success)
            # Reuse the same system_call_id from the tool_call above
            tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", system_call_id)

            messages.append(Message(role="assistant", content=tool_call_content))
            messages.append(Message(role="tool", content=tool_result_content, tool_call_id=system_call_id, name=tool.name))
            tool_results_data.append({"tool": tool.name, "args": args, "response": tool_result_content, "success": success, "user_query": user_query})

        # Build final answer — specific and grounded in tool results
        final_response = self._build_final_answer(tool_results_data, localization, all_success)
        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)
        messages.append(Message(role="assistant", content=final_answer_content))

        return DatasetExample(
            messages=messages,
            localization=localization,
            metadata={
                "difficulty": difficulty.value,
                "tool_category": selected_tools[0].category,
                "tool_name": selected_tools[0].name,
                "tools_used": [t.name for t in selected_tools],
                "num_tools": len(selected_tools),
                "success": all_success,
                "generated_at": datetime.utcnow().isoformat(),
                "generator_version": "4.0",
            }
        )

    @staticmethod
    def _build_final_answer(
        tool_results_data: list[dict],
        localization: Localization,
        all_success: bool,
    ) -> str:
        """Build a specific, grounded final answer based on tool results."""
        if not all_success:
            for tool_result in tool_results_data:
                response = json.loads(tool_result.get("response", "{}"))
                error = response.get("error")
                if error:
                    if "No results found" in error:
                        args = tool_result.get("args", {})
                        user_query = tool_result.get("user_query", "").lower()
                        if tool_result.get("tool") == "Search_Code":
                            if "try/except" in user_query or "error handling" in user_query:
                                return "No try/except-style error-handling matches were found in the codebase."
                            return f"No code matches were found for `{args.get('pattern', 'the pattern')}`."
                        if tool_result.get("tool") == "File_Search":
                            return f"No files were found matching `{args.get('pattern', 'the pattern')}`."
                    return LocalizationContent.get_error(error, localization)
            return LocalizationContent.get_error("Operation failed", localization)
        summaries = []
        for tool_result in tool_results_data:
            tool_name = tool_result.get("tool", "")
            args = tool_result.get("args", {})
            user_query = tool_result.get("user_query", "").lower()
            response = json.loads(tool_result.get("response", "{}"))
            raw_output = response.get("output", "")
            try:
                payload = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            except json.JSONDecodeError:
                payload = {"raw_output": raw_output}
            if tool_name == "Bash_ShellStatus":
                if "home directory" in user_query:
                    summaries.append(
                        f"Shell: {payload.get('shell', '/bin/zsh')}, user: {payload.get('user', 'sridhar')}, "
                        f"home: {payload.get('home_directory', '/Users/sridhar')}."
                    )
                else:
                    summaries.append(
                        f"Platform: {payload.get('platform', 'macOS')}, shell: {payload.get('shell', '/bin/zsh')}, "
                        f"user: {payload.get('user', 'sridhar')}, home: {payload.get('home_directory', '/Users/sridhar')}, "
                        f"cwd: {payload.get('current_directory', '/Users/sridhar/project')}."
                    )
            elif tool_name == "Bash_Execute":
                command = args.get('command', '')
                stdout = payload.get('stdout', '').strip()
                if command == "pwd" and stdout:
                    summaries.append(f"Current working directory: {stdout}")
                elif "ps aux --sort=-rss | head -5" in command and stdout:
                    summaries.append(f"Top processes by memory usage:\n{stdout}")
                else:
                    summaries.append(
                        f"Ran `{command}` successfully with exit code {payload.get('exit_code', 0)}."
                    )
            elif tool_name == "File_Read":
                target = payload.get('path', args.get('file_path', 'the file'))
                if any(phrase in user_query for phrase in ["summarize", "what it does", "what this file does", "contain", "contents"]):
                    preview = payload.get("content_preview", "").strip()
                    if target.endswith("src/main.py"):
                        summaries.append(f"{target} appears to be the main entry point.\nPreview:\n{preview}")
                    elif target.endswith("README.md"):
                        summaries.append(f"{target} appears to document project setup and usage.\nPreview:\n{preview}")
                    elif target.endswith("setup.py"):
                        summaries.append(f"{target} appears to define package setup metadata.\nPreview:\n{preview}")
                    elif target.endswith("lib/utils.py"):
                        summaries.append(f"{target} content preview:\n{preview}")
                    else:
                        summaries.append(f"{target} content preview:\n{preview}")
                else:
                    actual_lines = payload.get('lines_read', args.get('limit', 0))
                    if args.get("limit"):
                        preview = payload.get("content_preview", "").strip()
                        requested = args.get("limit")
                        summaries.append(f"For the request to read the first {requested} lines, {target} had {actual_lines} lines available.\nPreview:\n{preview}")
                    else:
                        line_label = "line" if actual_lines == 1 else "lines"
                        summaries.append(f"Read {target} and returned {actual_lines} {line_label}.")
            elif tool_name == "File_Write":
                path = payload.get('path', args.get('file_path', 'the file'))
                if "user class" in user_query:
                    summaries.append(f"Created the Python module `{path}` with a `User` class definition.")
                else:
                    summaries.append(f"Wrote the requested content to `{path}`.")
            elif tool_name == "File_Delete":
                summaries.append(f"Deleted {payload.get('path', args.get('path', 'the target path'))}.")
            elif tool_name == "File_Copy":
                summaries.append(f"Copied {payload.get('source', args.get('source', 'the source'))} to {payload.get('destination', args.get('destination', 'the destination'))}.")
            elif tool_name == "File_List":
                entry_names = [entry.get("name") for entry in payload.get("entries", [])[:3]]
                if "categorize" in user_query:
                    summaries.append(
                        "Project files by type: source: src, main.py; config: config; docs: README.md."
                    )
                else:
                    summaries.append(f"Listed {payload.get('total', 0)} entries from {args.get('directory', 'the directory')}: {', '.join(entry_names)}.")
            elif tool_name == "File_Search":
                all_matches = payload.get("matches", [])
                matches = all_matches[:3]
                if ".py and .js" in user_query or ".py and .js extensions" in user_query:
                    grouped = {}
                    for match in all_matches:
                        directory = match.rsplit("/", 1)[0] if "/" in match else "."
                        grouped.setdefault(directory, []).append(match.rsplit("/", 1)[-1])
                    group_lines = [f"{directory}/" for directory in grouped]
                    for directory, files in grouped.items():
                        group_lines.append(f"files: {', '.join(files)}")
                    summaries.append(
                        f"Found {payload.get('total', 0)} Python and JavaScript files grouped by directory:\n"
                        + "\n".join(group_lines)
                    )
                elif "markdown documentation" in user_query and "setup" in user_query:
                    summaries.append(f"Found markdown documentation files mentioning setup: {', '.join(all_matches)}.")
                elif "markdown documentation" in user_query:
                    summaries.append(f"Found {payload.get('total', 0)} markdown documentation files in the docs directory: {', '.join(matches)}.")
                else:
                    summaries.append(f"Found {payload.get('total', 0)} files matching {args.get('pattern', 'the pattern')}: {', '.join(matches)}.")
            elif tool_name == "Git_Status":
                staged = payload.get('staged', [])
                modified = payload.get('modified', [])
                untracked = payload.get('untracked', [])
                summaries.append(
                    f"Git status on {payload.get('branch', 'main')}: "
                    f"staged files: {', '.join(staged) or 'none'}; "
                    f"modified files: {', '.join(modified) or 'none'}; "
                    f"untracked files: {', '.join(untracked) or 'none'}."
                )
            elif tool_name == "Git_Log":
                commits = payload.get("commits", [])
                labels = [f"{c.get('hash', '')}: {c.get('message', '')}" for c in commits]
                summaries.append(f"Fetched {len(payload.get('commits', []))} recent commits: {'; '.join(labels)}.")
            elif tool_name == "Git_Branch":
                operation = args.get("operation", "list")
                current = payload.get("current", "main")
                if operation == "switch":
                    summaries.append(f"Switched to branch `{current}` successfully.")
                elif operation == "create":
                    summaries.append(f"Created branch `{args.get('branch_name', current)}`. Current branch is `{current}`.")
                else:
                    summaries.append(f"Branch operation `{operation}` completed. Branches now include {', '.join(payload.get('branches', []))}.")
            elif tool_name == "Git_Diff":
                files = payload.get('files', [])
                if files:
                    target = args.get('target', '')
                    if target == "main":
                        summaries.append(f"Compared the current branch with main. The available diff data shows {files[0].get('path', args.get('file_path', 'the target'))} with {files[0].get('additions', 0)} additions and {files[0].get('deletions', 0)} deletions.")
                    elif target == "working_tree":
                        summaries.append(f"Unstaged diff data shows {files[0].get('path', args.get('file_path', 'the target'))} with {files[0].get('additions', 0)} additions and {files[0].get('deletions', 0)} deletions.")
                    else:
                        summaries.append(f"Compared with the last commit, {files[0].get('path', args.get('file_path', 'the target'))} shows {files[0].get('additions', 0)} additions and {files[0].get('deletions', 0)} deletions.")
                else:
                    summaries.append("Diff completed successfully.")
            elif tool_name == "Git_Pull":
                remote = payload.get('remote', args.get('remote') or 'origin')
                branch = payload.get('branch', args.get('branch') or 'main')
                if args.get('remote') or args.get('branch'):
                    summaries.append(f"Pulled from {remote}/{branch} with {payload.get('files_updated', 0)} files updated and {payload.get('insertions', 0)} insertions.")
                else:
                    summaries.append(f"Pulled remote changes into the local branch from {remote}/{branch} with {payload.get('files_updated', 0)} files updated and {payload.get('insertions', 0)} insertions.")
            elif tool_name == "Git_Push":
                summaries.append(f"Pushed to {payload.get('remote', args.get('remote', 'origin'))}/{payload.get('branch', args.get('branch', 'main'))}.")
            elif tool_name == "Git_Commit":
                summaries.append(f"Created commit `{payload.get('commit_hash', 'unknown')}` with message `{payload.get('message', args.get('message', ''))}`.")
            elif tool_name == "Web_Search":
                first = payload.get('results', [{}])[0]
                if first:
                    summaries.append(f"Search returned {payload.get('total', 0)} results for `{args.get('query', 'the query')}`. Top result: {first.get('title', 'Untitled')} ({first.get('url', 'unknown URL')}).")
                else:
                    summaries.append(f"Search returned {payload.get('total', 0)} results for `{args.get('query', 'the query')}`.")
            elif tool_name == "Web_Fetch":
                url = payload.get('url', args.get('url', 'the URL'))
                if "releases" in url:
                    summaries.append(payload.get("summary", f"Fetched release notes from {url}."))
                elif "readme" in url.lower():
                    summaries.append(payload.get("summary", f"Fetched README from {url}."))
                else:
                    summaries.append(f"Fetched {url} with status {payload.get('status', 200)} and content type {payload.get('content_type', 'unknown')}.")
            elif tool_name == "Web_Screenshot":
                summaries.append(f"Captured a screenshot of {payload.get('url', args.get('url', 'the URL'))} at {payload.get('width', 0)}x{payload.get('height', 0)}.")
            elif tool_name == "Python_Run":
                stdout = payload.get('stdout', '').strip()
                if "fibonacci" in args.get("code", ""):
                    summaries.append(f"Fibonacci result: {stdout}")
                elif "csv.DictReader" in args.get("code", "") and stdout:
                    summaries.append("Python data pipeline output: 2 rows processed with a total value of 3.")
                elif "urllib.request.urlopen" in args.get("code", "") and stdout:
                    summaries.append(f"Python HTTP request completed successfully and printed the response body preview: `{stdout}`.")
                elif stdout:
                    summaries.append(f"Output: {stdout}")
                else:
                    summaries.append("Python code ran successfully.")
            elif tool_name == "Node_Run":
                stdout = payload.get('stdout', '').strip()
                if "readfilesync('readme.md'" in args.get("code", "").lower() and stdout:
                    summaries.append(f"Printed the contents of README.md: `{stdout}`")
                else:
                    summaries.append(f"Node.js code ran successfully with exit code {payload.get('exit_code', 0)} and stdout `{stdout}`.")
            elif tool_name == "Python_Test":
                coverage = payload.get("coverage")
                if coverage and coverage.get("enabled"):
                    summaries.append(
                        f"Pytest run completed for {args.get('file_path', 'the target')} with {payload.get('passed', 0)} passing tests and {coverage.get('percent', 0)}% coverage."
                    )
                else:
                    summaries.append(f"Pytest run completed for {args.get('file_path', 'the target')} with {payload.get('passed', 0)} passing tests and no reported failures.")
            elif tool_name == "Search_Code":
                matches = payload.get("matches", [])
                if "try/except" in user_query or "error handling" in user_query:
                    if matches:
                        formatted = ", ".join(
                            f"{m.get('file', 'unknown')}:{m.get('line', 0)} (`{m.get('context', '')}`)"
                            for m in matches
                        )
                        summaries.append(f"Found {payload.get('total', 0)} try/except-style error-handling matches in the codebase: {formatted}.")
                    else:
                        summaries.append("No try/except-style error-handling matches were found in the codebase.")
                else:
                    formatted = ", ".join(
                        f"{m.get('file', 'unknown')}:{m.get('line', 0)} (`{m.get('context', '')}`)"
                        for m in matches[:2]
                    )
                    summaries.append(f"Found {payload.get('total', 0)} code matches for `{args.get('pattern', 'the pattern')}`: {formatted}.")
            elif tool_name == "Search_Replace":
                if args.get("preview"):
                    summaries.append(f"Previewed {payload.get('replacements', 0)} replacement(s) of `{args.get('search', 'the search term')}` in {payload.get('file', args.get('path', 'the target path'))}.")
                else:
                    summaries.append(f"Replaced {payload.get('replacements', 0)} occurrence(s) of `{args.get('search', 'the search term')}` in {payload.get('file', args.get('path', 'the target path'))}.")
            elif tool_name == "System_Info":
                category = args.get("category", "os")
                if category == "cpu":
                    cpu_count = payload.get('cpu_count', 0)
                    os_name = payload.get('os')
                    python_version = payload.get('python_version')
                    details = [f"{cpu_count} cores available"]
                    if os_name:
                        details.append(f"OS {os_name}")
                    if python_version:
                        details.append(f"Python {python_version}")
                    summaries.append(f"Retrieved CPU information: {', '.join(details)}.")
                elif category == "memory":
                    summaries.append(
                        f"Memory details: {payload.get('memory_total_gb', 0)} GB total, "
                        f"OS {payload.get('os', 'unknown OS')}, Python {payload.get('python_version', 'unknown')}, "
                        f"{payload.get('cpu_count', 0)} CPU cores."
                    )
                elif category == "os":
                    summaries.append(
                        f"OS details: {payload.get('os', 'unknown OS')}, Python {payload.get('python_version', 'unknown')}, "
                        f"{payload.get('cpu_count', 0)} CPU cores, {payload.get('memory_total_gb', 0)} GB memory."
                    )
                else:
                    summaries.append(f"Retrieved {category} system information successfully.")
            elif tool_name == "Process_List":
                if args.get("limit"):
                    process_names = [f"{p.get('name', 'unknown')} ({p.get(args.get('sort_by', 'cpu'), p.get('cpu', 'n/a'))})" for p in payload.get("processes", [])]
                    summaries.append(f"Available process results sorted by {args.get('sort_by', 'pid')}: {', '.join(process_names)}.")
                elif args.get("filter") == "running" or "active processes" in user_query:
                    process_names = [p.get("name", "unknown") for p in payload.get("processes", [])]
                    summaries.append(f"Retrieved {payload.get('total', 0)} active processes: {', '.join(process_names)}.")
                else:
                    summaries.append(f"Retrieved {payload.get('total', 0)} processes.")
            elif tool_name == "Database_Query":
                if "orders" in args.get("query", "").lower() and "30 days" in args.get("query", "").lower():
                    summaries.append(f"Found {payload.get('total', 0)} orders from the last 30 days in {args.get('database', 'the database')}.")
                else:
                    summaries.append(f"Query returned {payload.get('total', 0)} rows from {args.get('database', 'the database')}.")
            elif tool_name == "Database_List":
                if "databases" in payload:
                    summaries.append(f"Listed {payload.get('total', 0)} available databases: {', '.join(payload.get('databases', []))}.")
                else:
                    summaries.append(f"Listed database objects for {args.get('database', 'the requested database')}.")
            else:
                summaries.append(f"Completed {tool_name} successfully.")

        return " ".join(summaries) if summaries else "Completed the requested task successfully."

    def generate_batch(
        self,
        count: int,
        localization: Localization,
        difficulty_distribution: dict | None = None,
        error_rate: float = 0.15,
    ) -> list[DatasetExample]:
        """Generate a batch with difficulty distribution."""
        dist = difficulty_distribution or {
            DifficultyLevel.EASY: 0.30,
            DifficultyLevel.MEDIUM: 0.40,  # OPTIMAL per ToolACE research
            DifficultyLevel.HARD: 0.20,
            DifficultyLevel.EXPERT: 0.10,
        }

        examples = []
        for _ in range(count):
            difficulties = list(dist.keys())
            weights = list(dist.values())
            difficulty = random.choices(difficulties, weights=weights)[0]
            include_error = random.random() < error_rate
            example = self.generate_single(
                localization=localization,
                difficulty=difficulty,
                include_error=include_error,
            )
            examples.append(example)

        return examples

    def generate_localized_batch(
        self,
        count_per_locale: int,
        languages: list[str] | None = None,
        tones: list[str] | None = None,
        formalities: list[str] | None = None,
    ) -> list[DatasetExample]:
        """Generate examples across multiple localizations."""
        langs = languages or [l.value for l in Language]
        tns = tones or [t.value for t in Tone]
        fmls = formalities or [f.value for f in FormalityLevel]

        all_examples = []
        for lang in langs:
            for tone in tns:
                for formality in fmls:
                    loc = Localization(
                        language=lang,
                        tone=tone,
                        formality=formality,
                        humanize=True,
                        humanize_level="medium",
                    )
                    examples = self.generate_batch(count=count_per_locale, localization=loc)
                    all_examples.extend(examples)

        return all_examples

    def run_pipeline(
        self,
        total_count: int,
        localization: Localization,
        output_path: str,
        validate: bool = True,
        difficulty_distribution: dict | None = None,
    ) -> dict:
        """Run the complete pipeline."""
        print(f"Generating {total_count} examples...")
        examples = self.generate_batch(
            count=total_count,
            localization=localization,
            difficulty_distribution=difficulty_distribution,
        )
        print(f"Generated {len(examples)} examples")

        if validate:
            valid_examples, stats = self.validator.validate_batch(examples)
            print(f"Valid: {stats['valid']}/{stats['total']}")
            if stats['errors_by_type']:
                print(f"Errors by type: {stats['errors_by_type']}")
            examples = valid_examples

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

        return {"status": "success", "generated": len(examples), "output_path": output_path}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT_DIR = Path("/tmp/beastcli-eng1/output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    pipeline = ComprehensiveDatasetPipeline(seed=42)

    print("=" * 70)
    print("ULTIMATE AGENT TRAINING DATASET GENERATOR v4.0")
    print("=" * 70)

    print("\nGenerating dataset with all localizations...")
    examples = pipeline.generate_localized_batch(
        count_per_locale=25,
        languages=["en", "hi", "es", "fr", "de", "ja", "zh", "ko"],
        tones=["professional", "casual", "technical", "friendly"],
        formalities=["formal", "neutral", "informal"],
    )

    valid_examples, stats = pipeline.validator.validate_batch(examples)
    print(f"\nValidation: {stats['valid']}/{stats['total']} valid")

    random.shuffle(valid_examples)
    train_size = int(len(valid_examples) * 0.9)
    train_examples = valid_examples[:train_size]
    test_examples = valid_examples[train_size:]

    train_path = OUTPUT_DIR / "merged_train.jsonl"
    test_path = OUTPUT_DIR / "merged_test.jsonl"

    for path, data in [(train_path, train_examples), (test_path, test_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} examples to {path}")

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total examples: {len(valid_examples)}")
    print(f"Train: {len(train_examples)}, Test: {len(test_examples)}")
    print(f"Tools: {len(ToolRegistry.get_all_tools())}")
    print(f"Languages: 8, Tones: 4, Formalities: 3")
    print("=" * 70)
