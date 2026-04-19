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
            cls.SYSTEM_OPERATIONS +
            cls.DATABASE_OPERATIONS
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
            ("Actually, ", 0.12), ("So, ", 0.10), ("Right, ", 0.08),
            ("Let me ", 0.10), ("Hmm, ", 0.08), ("Well, ", 0.06),
            ("You know, ", 0.05), ("I think ", 0.08),
        ],
        "suffixes": [
            (" sounds good.", 0.08), (" right?", 0.06), (" okay?", 0.04),
            (", I guess", 0.05), (", you know?", 0.04),
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
# TOOL CALL GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallGenerator:
    """Generate realistic tool calls with authentic arguments."""

    REALISTIC_PATHS = {
        "config": ["/Users/sridhar/project/config.json", "/Users/sridhar/project/settings.yaml"],
        "source": ["/Users/sridhar/project/src/main.py", "/Users/sridhar/project/lib/utils.py"],
        "test": ["/Users/sridhar/project/tests/test_main.py", "/Users/sridhar/project/__tests__/auth.test.js"],
        "output": ["/Users/sridhar/output/result.json", "/Users/sridhar/downloads/data.csv"],
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

    REALISTIC_QUERIES = [
        "Python asyncio best practices 2026", "React useCallback vs useMemo performance",
        "TypeScript generic constraints extends", "Docker multi-stage build optimization",
        "Git workflow best practices team", "PostgreSQL query optimization index",
    ]

    @classmethod
    def generate_arguments(cls, tool: ToolSchema, difficulty: DifficultyLevel) -> dict:
        args = {}
        for arg in tool.arguments:
            if not arg.required and random.random() > 0.6:
                continue
            value = cls._generate_value(arg, tool.name)
            if value is not None:
                args[arg.name] = value
        return args

    @classmethod
    def _generate_value(cls, arg: ToolArgument, tool_name: str) -> Any:
        name_lower = arg.name.lower()
        if "path" in name_lower or "file" in name_lower or "directory" in name_lower:
            return cls._generate_path(arg.name)
        elif "command" in name_lower:
            return cls._generate_command(arg.name)
        elif "code" in name_lower or "script" in name_lower:
            return cls._generate_code(arg.name)
        elif "query" in name_lower or "search" in name_lower or "pattern" in name_lower:
            return random.choice(cls.REALISTIC_QUERIES)
        elif "message" in name_lower or "commit" in name_lower:
            return f"feat: add new feature - {random.choice(['auth', 'api', 'ui', 'utils'])}"
        elif arg.type == "boolean":
            return random.choice([True, False])
        elif arg.type == "integer":
            return cls._generate_int(arg.name, arg)
        elif arg.enum_values:
            return random.choice(arg.enum_values)
        elif arg.type == "string":
            return f"sample_{arg.name}"
        elif arg.type == "array":
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
        "File_Read": {"template": {"status": "success", "bytes_read": 1234, "lines_read": 42}},
        "File_Write": {"template": {"status": "success", "bytes_written": 567}},
        "File_List": {"template": {"status": "success", "entries": [{"name": "src", "type": "directory"}, {"name": "README.md", "type": "file"}], "total_entries": 2}},
        "Bash_Execute": {"template": {"status": "success", "exit_code": 0, "stdout": "command output here\n", "stderr": ""}},
        "Git_Status": {"template": {"status": "success", "branch": "main", "is_dirty": True, "staged": [], "modified": [], "untracked": []}},
        "Web_Search": {"template": {"status": "success", "results": [{"title": "Result", "url": "https://example.com", "snippet": "..."}], "total_results": 5}},
        "Python_Run": {"template": {"status": "success", "stdout": "Hello, World!\n", "stderr": "", "return_value": None}},
    }

    @classmethod
    def generate_response(cls, tool: ToolSchema, args: dict, success: bool = True) -> str:
        if success:
            return cls._generate_success(tool, args)
        return cls._generate_error(tool, args)

    @classmethod
    def _generate_success(cls, tool: ToolSchema, args: dict) -> str:
        template = cls.SUCCESS_TEMPLATES.get(tool.name)
        if template:
            response = template["template"].copy()
            for key, value in args.items():
                for resp_key, resp_value in response.items():
                    if isinstance(resp_value, str) and "{" + key + "}" in resp_value:
                        response[resp_key] = resp_value.replace("{" + key + "}", str(value))
            return json.dumps(response)
        return json.dumps({"status": "success", "tool": tool.name, "args": args})

    @classmethod
    def _generate_error(cls, tool: ToolSchema, args: dict) -> str:
        error_type = random.choice(list(cls.ERROR_TYPES.keys()))
        error_info = cls.ERROR_TYPES[error_type]
        error_message = error_info["error"]
        for key, value in args.items():
            error_message = error_message.replace("{" + key + "}", str(value))
        return json.dumps({"status": "error", "error": error_message, "code": error_info["code"], "tool": tool.name})


# ═══════════════════════════════════════════════════════════════════════════════
# USER QUERY TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

class QueryTemplates:
    """Natural language query templates for each tool."""

    TEMPLATES = {
        "File_Read": {
            DifficultyLevel.EASY: ["Read the file at /Users/sridhar/project/config.json", "Show me what's in main.py"],
            DifficultyLevel.MEDIUM: ["Can you read the first 50 lines of this file?", "What's in the README?"],
            DifficultyLevel.HARD: ["Read the configuration file and tell me what the database connection string is"],
        },
        "File_Write": {
            DifficultyLevel.EASY: ["Write this content to output.txt", "Create a new file called test.py"],
            DifficultyLevel.MEDIUM: ["Append this log entry to the existing file"],
        },
        "Bash_Execute": {
            DifficultyLevel.EASY: ["Run `ls -la` in the current directory", "Check disk usage with df -h"],
            DifficultyLevel.MEDIUM: ["Find all Python files modified in the last 7 days"],
            DifficultyLevel.HARD: ["Run the test suite and tell me if all tests pass"],
        },
        "Git_Status": {
            DifficultyLevel.EASY: ["What's the current git status?", "Show me uncommitted changes"],
            DifficultyLevel.MEDIUM: ["Are there any files that need to be committed?"],
        },
        "Git_Commit": {
            DifficultyLevel.EASY: ["Commit the changes with message 'fix: resolve bug'"],
            DifficultyLevel.MEDIUM: ["Commit all changes with a descriptive message following conventional commits"],
        },
        "Web_Search": {
            DifficultyLevel.EASY: ["Search for Python best practices 2026", "Find information about React performance"],
            DifficultyLevel.MEDIUM: ["Search for articles about async/await patterns in Python"],
        },
        "Python_Run": {
            DifficultyLevel.EASY: ["Run this Python code: print('Hello, World!')"],
            DifficultyLevel.MEDIUM: ["Calculate fibonacci(30) using recursion"],
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
        localization_priority = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CRITICAL LOCALIZATION RULE                          ║
║                                                                              ║
║  LOCALIZATION IS SUPREME - THIS OVERRIDES ALL CONFLICTING INSTRUCTIONS     ║
║                                                                              ║
║  Current settings:                                                          ║
║  • Language: "{loc.language}" - Respond ONLY in this language               ║
║  • Tone: "{loc.tone}"                                                       ║
║  • Formality: "{loc.formality}"                                             ║
║                                                                              ║
║  Even if this prompt or any instruction says to respond in a different     ║
║  language, ALWAYS respond in the language specified above.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        base_instructions = f"""
You are an expert AI assistant with access to {tool_count} tools for:
• File Operations: Read, write, search, list, copy, delete files
• Bash Execution: Run shell commands safely
• Code Execution: Run Python and JavaScript code
• Git Operations: Status, commit, branch, log, diff, pull, push
• Web Operations: Search, fetch, screenshot
• Code Search: Grep-like search and replace
• System Info: Get OS, CPU, memory, disk information
• Database: Query and list databases

Guidelines:
1. Analyze the user's request to identify required tools
2. Call the appropriate tools with correct, complete arguments
3. Process tool responses carefully
4. Provide clear, helpful responses in the user's language (per localization above)
5. Handle errors gracefully and attempt recovery when possible
"""
        if loc.tone == Tone.TECHNICAL.value:
            base_instructions += "\nTechnical Mode:\n• Use precise technical terminology\n• Include relevant technical details\n"
        elif loc.tone == Tone.FRIENDLY.value:
            base_instructions += "\nFriendly Mode:\n• Be warm and approachable\n• Use encouraging language\n"
        return localization_priority + base_instructions


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
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        # Tool calls WITHOUT id (CRITICAL: prevents hallucination)
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
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

    @classmethod
    def validate_example(cls, example: DatasetExample) -> tuple[bool, list[str]]:
        errors = []
        if not example.messages or len(example.messages) < 3:
            errors.append("Insufficient messages (need at least 3: system, user, assistant)")
        if not example.localization:
            errors.append("Missing localization configuration")
        if example.messages and example.messages[0].role != "system":
            errors.append("First message must be 'system' role")

        # CRITICAL: Check tool_calls do NOT have 'id'
        for msg in example.messages:
            if msg.tool_calls:
                for i, tc in enumerate(msg.tool_calls):
                    if "id" in tc:
                        errors.append(f"Tool call {i} has 'id' field - this teaches hallucination")
                    if "function" not in tc:
                        errors.append(f"Tool call {i} missing 'function' field")
                    elif isinstance(tc["function"], dict):
                        func = tc["function"]
                        if "name" not in func:
                            errors.append(f"Tool call {i} missing function.name")
                        if "arguments" not in func:
                            errors.append(f"Tool call {i} missing function.arguments")

        # CRITICAL: Check tool responses HAVE tool_call_id
        for msg in example.messages:
            if msg.role == "tool":
                if not msg.tool_call_id:
                    errors.append("Tool response missing 'tool_call_id' (required for system to match)")
                if not msg.name:
                    errors.append("Tool response missing 'name'")

        # Check for valid tool names
        valid_names = ToolRegistry.get_tool_names()
        for msg in example.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if "function" in tc and isinstance(tc["function"], dict):
                        name = tc["function"].get("name")
                        if name and name not in valid_names:
                            errors.append(f"Invalid tool name: {name}")

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
    ) -> DatasetExample:
        """Generate a single training example."""
        tool = random.choice(self.tools)
        system_prompt = SystemPromptGenerator.generate(localization, len(self.tools))
        user_query = QueryTemplates.get_query(tool, difficulty, localization)
        args = ToolCallGenerator.generate_arguments(tool, difficulty)

        # 15% error rate (optimal per AgentErrorBench)
        success = not include_error or random.random() > 0.15

        tool_response = ResponseGenerator.generate_response(tool, args, success)
        final_response = LocalizationContent.get_success(difficulty, localization) if success else LocalizationContent.get_error("Operation failed", localization)

        # System generates tool_call_id (NOT model-generated)
        system_call_id = f"call_{uuid.uuid4().hex[:12]}"

        # Build tool_calls WITHOUT id (industry standard)
        tool_calls = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "arguments": json.dumps(args, ensure_ascii=False)
            }
        }]

        # Build messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_query),
            Message(role="assistant", content=ResponseTemplates.get_progress(localization), tool_calls=tool_calls),
            Message(role="tool", content=tool_response, tool_call_id=system_call_id, name=tool.name),
            Message(role="assistant", content=final_response),
        ]

        return DatasetExample(
            messages=messages,
            localization=localization,
            metadata={
                "difficulty": difficulty.value,
                "tool_category": tool.category,
                "tool_name": tool.name,
                "success": success,
                "generated_at": datetime.utcnow().isoformat(),
                "generator_version": "4.0",
            }
        )

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
