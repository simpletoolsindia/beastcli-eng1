#!/usr/bin/env python3
"""
High-quality 100k dataset generator using 5 core tools.
Based on Claude Code research: Bash, Read, Edit, Glob, Web.

These 5 tools cover everything a CLI agent needs:
- Bash: python3, node, git, npm, cargo, find, grep, ls, etc.
- Read: file contents
- Edit: string replacement in files
- Glob: file pattern matching
- Web: Web_Search + Web_Fetch
"""
import json
import random
import uuid
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from merged_dataset_generator import (
    Localization,
    LocalizationContent,
    DifficultyLevel,
    Language,
    Tone,
    FormalityLevel,
    ToolRegistry,
    ToolSchema,
    ToolCallGenerator,
    ResponseGenerator,
    FinalAnswerGenerator,
    SystemPromptGenerator,
    Message,
    DatasetExample,
    DatasetValidator,
)

# Expanded intents for the 5 core tools (Bash covers git, npm, python, node, docker, etc.)
EXPANDED_INTENTS = {
    "Bash": [
        # Git commands
        {"q": "check git status", "a": {"command": "git status --porcelain", "timeout": 10}},
        {"q": "git log", "a": {"command": "git log --oneline -5", "timeout": 10}},
        {"q": "git diff", "a": {"command": "git diff", "timeout": 10}},
        {"q": "git add all", "a": {"command": "git add -A && git status", "timeout": 10}},
        {"q": "commit changes", "a": {"command": "git commit -m 'chore: update'", "timeout": 10}},
        {"q": "git push", "a": {"command": "git push origin main", "timeout": 30}},
        {"q": "git pull", "a": {"command": "git pull origin main", "timeout": 30}},
        {"q": "git branches", "a": {"command": "git branch -a", "timeout": 10}},
        {"q": "npm install", "a": {"command": "npm install", "timeout": 120}},
        {"q": "npm run build", "a": {"command": "npm run build", "timeout": 120}},
        {"q": "npm test", "a": {"command": "npm test", "timeout": 60}},
        {"q": "pip install", "a": {"command": "pip install -r requirements.txt", "timeout": 120}},
        {"q": "pip list", "a": {"command": "pip list", "timeout": 10}},
        {"q": "python script", "a": {"command": "python3 script.py", "timeout": 30}},
        {"q": "run pytest", "a": {"command": "pytest -v", "timeout": 60}},
        {"q": "node script", "a": {"command": "node script.js", "timeout": 30}},
        {"q": "cargo build", "a": {"command": "cargo build", "timeout": 120}},
        {"q": "cargo run", "a": {"command": "cargo run", "timeout": 60}},
        {"q": "docker ps", "a": {"command": "docker ps", "timeout": 10}},
        {"q": "docker images", "a": {"command": "docker images", "timeout": 10}},
        {"q": "list files", "a": {"command": "ls -la", "timeout": 10}},
        {"q": "find python files", "a": {"command": "find . -name '*.py' -type f | head -20", "timeout": 30}},
        {"q": "disk usage", "a": {"command": "df -h", "timeout": 10}},
        {"q": "memory info", "a": {"command": "free -m", "timeout": 10}},
        {"q": "top processes", "a": {"command": "ps aux --sort=-rss | head -10", "timeout": 10}},
        {"q": "current directory", "a": {"command": "pwd", "timeout": 5}},
        {"q": "search code", "a": {"command": "grep -rn 'TODO' . --include='*.py' | head -10", "timeout": 30}},
        {"q": "count lines", "a": {"command": "wc -l $(find . -name '*.py' -type f)", "timeout": 30}},
        {"q": "find imports", "a": {"command": "rg 'import ' src/ --type py -l | head -10", "timeout": 30}},
        {"q": "find main functions", "a": {"command": "rg 'def main' . --type py -l", "timeout": 30}},
        {"q": "git clone", "a": {"command": "git clone https://github.com/example/repo.git", "timeout": 60}},
        {"q": "docker-compose up", "a": {"command": "docker-compose up -d", "timeout": 60}},
        {"q": "curl api", "a": {"command": "curl -s https://api.github.com/", "timeout": 15}},
        {"q": "system uptime", "a": {"command": "uptime", "timeout": 5}},
        {"q": "user info", "a": {"command": "whoami && id", "timeout": 5}},
        {"q": "python version", "a": {"command": "python3 --version", "timeout": 5}},
        {"q": "node version", "a": {"command": "node --version", "timeout": 5}},
    ],
    "Read": [
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
        {"q": "read index.js", "a": {"file_path": "src/index.js"}},
        {"q": "read server.py", "a": {"file_path": "server.py"}},
        {"q": "read .gitignore", "a": {"file_path": ".gitignore"}},
        {"q": "read Makefile", "a": {"file_path": "Makefile"}},
        {"q": "read pyproject.toml", "a": {"file_path": "pyproject.toml"}},
        {"q": "read tsconfig.json", "a": {"file_path": "tsconfig.json"}},
        {"q": "read .npmrc", "a": {"file_path": ".npmrc"}},
        {"q": "read logs", "a": {"file_path": "logs/app.log"}},
        {"q": "read seed data", "a": {"file_path": "data/seed.json"}},
        {"q": "read middleware", "a": {"file_path": "src/middleware.py"}},
    ],
    "Write": [
        {"q": "write output.txt", "a": {"file_path": "output.txt", "content": "Build completed successfully.\n", "append": False}},
        {"q": "write test.py", "a": {"file_path": "test.py", "content": "def test_example():\n    assert True\n", "append": False}},
        {"q": "write config.json", "a": {"file_path": "config.json", "content": '{"version": "1.0", "debug": true}\n', "append": False}},
        {"q": "append to log", "a": {"file_path": "logs/app.log", "content": "[INFO] Request completed\n", "append": True}},
        {"q": "write README", "a": {"file_path": "README.md", "content": "# Project\n\nSetup instructions here.\n", "append": False}},
        {"q": "write models.py", "a": {"file_path": "src/models.py", "content": "class User:\n    def __init__(self, name: str):\n        self.name = name\n", "append": False}},
        {"q": "write .gitignore", "a": {"file_path": ".gitignore", "content": "__pycache__/\nnode_modules/\n.env\n", "append": False}},
        {"q": "write requirements.txt", "a": {"file_path": "requirements.txt", "content": "requests>=2.28.0\npytest>=7.0.0\n", "append": False}},
        {"q": "write index.html", "a": {"file_path": "public/index.html", "content": "<!DOCTYPE html>\n<html>\n<head><title>App</title></head>\n<body>Hello</body>\n</html>\n", "append": False}},
        {"q": "write script.sh", "a": {"file_path": "scripts/deploy.sh", "content": "#!/bin/bash\nset -e\nnpm run build\n", "append": False}},
        {"q": "write Dockerfile", "a": {"file_path": "Dockerfile", "content": "FROM python:3.11-slim\nCOPY . /app\nWORKDIR /app\nRUN pip install -r requirements.txt\n", "append": False}},
        {"q": "write docker-compose.yml", "a": {"file_path": "docker-compose.yml", "content": "version: '3'\nservices:\n  web:\n    build: .\n    ports:\n      - '3000:3000'\n", "append": False}},
    ],
    "Grep": [
        {"q": "find TODO comments", "a": {"pattern": "TODO", "path": ".", "case_sensitive": False}},
        {"q": "search for function", "a": {"pattern": "def main", "path": "src"}},
        {"q": "find imports", "a": {"pattern": "import\\s+", "path": "src", "file_types": ".py"}},
        {"q": "find console.log", "a": {"pattern": "console\\.log", "path": "src", "file_types": ".js"}},
        {"q": "search for class", "a": {"pattern": "class User", "path": "."}},
        {"q": "find error handling", "a": {"pattern": "try:", "path": "src"}},
        {"q": "find async functions", "a": {"pattern": "async def", "path": "."}},
        {"q": "search pytest", "a": {"pattern": "def test_", "path": "tests"}},
        {"q": "find TODO", "a": {"pattern": "FIXME", "path": ".", "case_sensitive": False}},
        {"q": "find decorators", "a": {"pattern": "@", "path": "src"}},
        {"q": "search regex", "a": {"pattern": "re\\.compile", "path": "."}},
        {"q": "find queries", "a": {"pattern": "SELECT.*FROM", "path": ".", "file_types": ".py"}},
    ],
    "TodoWrite": [
        {"q": "add todo for setup", "a": {"todo": "Set up project structure", "status": "pending"}},
        {"q": "mark task done", "a": {"todo": "Add authentication", "status": "completed"}},
        {"q": "add bug fix todo", "a": {"todo": "Fix memory leak in worker", "status": "pending"}},
        {"q": "update task status", "a": {"todo": "Write tests", "status": "in_progress"}},
        {"q": "add refactor todo", "a": {"todo": "Refactor database layer", "status": "pending"}},
        {"q": "add docs todo", "a": {"todo": "Document API endpoints", "status": "pending"}},
    ],
    "TodoRead": [
        {"q": "check todo list", "a": {}},
        {"q": "show tasks", "a": {}},
        {"q": "read todos", "a": {}},
        {"q": "view progress", "a": {}},
        {"q": "list tasks", "a": {}},
    ],
    "Edit": [
        {"q": "replace foo with bar", "a": {"file_path": "src/main.py", "old_string": "foo", "new_string": "bar"}},
        {"q": "change port 3000 to 8080", "a": {"file_path": "config.json", "old_string": '"port": 3000', "new_string": '"port": 8080'}},
        {"q": "update version to 2.0", "a": {"file_path": "package.json", "old_string": '"version": "1.0.0"', "new_string": '"version": "2.0.0"'}},
        {"q": "add console.log", "a": {"file_path": "src/index.js", "old_string": "console.log('start');", "new_string": "console.log('start');\nconsole.log('initialized');"}},
        {"q": "fix typo in README", "a": {"file_path": "README.md", "old_string": "instalation", "new_string": "installation"}},
        {"q": "add new import", "a": {"file_path": "src/app.py", "old_string": "import os\n", "new_string": "import os\nimport json\n"}},
        {"q": "change timeout value", "a": {"file_path": "config.yaml", "old_string": "timeout: 30", "new_string": "timeout: 60"}},
        {"q": "rename function", "a": {"file_path": "lib/utils.py", "old_string": "def get_user():", "new_string": "def fetch_user():"}},
        {"q": "add return type", "a": {"file_path": "src/models.py", "old_string": "def get_name(self):", "new_string": "def get_name(self) -> str:"}},
        {"q": "fix syntax error", "a": {"file_path": "test.py", "old_string": "pritn('hello')", "new_string": "print('hello')"}},
        {"q": "add shebang", "a": {"file_path": "script.sh", "old_string": "", "new_string": "#!/bin/bash\n"}},
        {"q": "update env var", "a": {"file_path": ".env", "old_string": "DEBUG=false", "new_string": "DEBUG=true"}},
        {"q": "add auth middleware", "a": {"file_path": "server.py", "old_string": "app = Flask(__name__)\n", "new_string": "app = Flask(__name__)\napp.config['SECRET_KEY'] = 'dev-key'\n"}},
        {"q": "update python version", "a": {"file_path": "Dockerfile", "old_string": "FROM python:3.11-slim", "new_string": "FROM python:3.12-slim"}},
        {"q": "add to requirements", "a": {"file_path": "requirements.txt", "old_string": "pytest>=7.0.0\n", "new_string": "pytest>=7.0.0\nblack>=23.0.0\n"}},
    ],
    "Glob": [
        {"q": "find python files", "a": {"pattern": "**/*.py"}},
        {"q": "find js files", "a": {"pattern": "**/*.js"}},
        {"q": "find test files", "a": {"pattern": "**/*test*.py"}},
        {"q": "find config files", "a": {"pattern": "**/config*"}},
        {"q": "find all markdown", "a": {"pattern": "**/*.md"}},
        {"q": "find yaml configs", "a": {"pattern": "**/*.yaml"}},
        {"q": "find json files", "a": {"pattern": "**/*.json"}},
        {"q": "find dockerfiles", "a": {"pattern": "**/Dockerfile*"}},
        {"q": "find shell scripts", "a": {"pattern": "**/*.sh"}},
        {"q": "find css files", "a": {"pattern": "**/*.css"}},
        {"q": "find html files", "a": {"pattern": "**/*.html"}},
        {"q": "find env files", "a": {"pattern": "**/.env*"}},
        {"q": "find gitignore", "a": {"pattern": "**/.gitignore"}},
        {"q": "find src files", "a": {"pattern": "src/**/*"}},
        {"q": "find requirements", "a": {"pattern": "**/requirements*.txt"}},
    ],
    "Glob": [
        {"q": "find python files", "a": {"pattern": "**/*.py"}},
        {"q": "find js files", "a": {"pattern": "**/*.js"}},
        {"q": "find test files", "a": {"pattern": "**/*test*.py"}},
        {"q": "find config files", "a": {"pattern": "**/config*"}},
        {"q": "find all markdown", "a": {"pattern": "**/*.md"}},
        {"q": "find yaml configs", "a": {"pattern": "**/*.yaml"}},
        {"q": "find json files", "a": {"pattern": "**/*.json"}},
        {"q": "find dockerfiles", "a": {"pattern": "**/Dockerfile*"}},
        {"q": "find shell scripts", "a": {"pattern": "**/*.sh"}},
        {"q": "find css files", "a": {"pattern": "**/*.css"}},
        {"q": "find html files", "a": {"pattern": "**/*.html"}},
        {"q": "find env files", "a": {"pattern": "**/.env*"}},
        {"q": "find gitignore", "a": {"pattern": "**/.gitignore"}},
        {"q": "find src files", "a": {"pattern": "src/**/*"}},
        {"q": "find __pycache__", "a": {"pattern": "**/__pycache__/**"}},
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
        {"q": "search Node.js best practices", "a": {"query": "Node.js best practices production", "num_results": 5}},
        {"q": "search Docker compose", "a": {"query": "Docker compose production deployment", "num_results": 5}},
        {"q": "search Python testing", "a": {"query": "Python pytest best practices", "num_results": 5}},
        {"q": "search Git branching", "a": {"query": "Git branching strategy GitFlow vs trunk", "num_results": 5}},
        {"q": "search CI/CD pipeline", "a": {"query": "CI/CD pipeline GitHub Actions best practices", "num_results": 5}},
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
        {"q": "fetch PyPI package", "a": {"url": "https://pypi.org/project/requests/"}},
        {"q": "fetch Wikipedia", "a": {"url": "https://en.wikipedia.org/wiki/Software_development"}},
    ],
}

# 48 locales for diverse language coverage
LOCALES = [
    ("en", "professional", "formal"), ("en", "professional", "neutral"), ("en", "professional", "informal"),
    ("en", "casual", "formal"), ("en", "casual", "neutral"), ("en", "casual", "informal"),
    ("en", "technical", "formal"), ("en", "technical", "neutral"), ("en", "technical", "informal"),
    ("en", "friendly", "formal"), ("en", "friendly", "neutral"), ("en", "friendly", "informal"),
    ("hi", "professional", "formal"), ("hi", "professional", "neutral"), ("hi", "casual", "neutral"),
    ("hi", "casual", "informal"), ("hi", "friendly", "formal"), ("hi", "friendly", "neutral"),
    ("es", "professional", "neutral"), ("es", "casual", "informal"),
    ("fr", "professional", "formal"), ("fr", "casual", "neutral"),
    ("de", "professional", "neutral"), ("de", "casual", "informal"),
    ("ja", "professional", "formal"), ("ja", "casual", "neutral"),
    ("zh", "professional", "neutral"), ("zh", "casual", "informal"),
    ("ko", "professional", "formal"), ("ko", "casual", "neutral"),
    ("pt", "professional", "neutral"), ("pt", "casual", "informal"),
    ("it", "professional", "neutral"), ("it", "casual", "informal"),
    ("ar", "professional", "formal"), ("ar", "casual", "informal"),
    ("ru", "professional", "neutral"), ("ru", "casual", "informal"),
]

DIFFICULTIES = [
    (DifficultyLevel.EASY, 0.30),
    (DifficultyLevel.MEDIUM, 0.40),
    (DifficultyLevel.HARD, 0.20),
    (DifficultyLevel.EXPERT, 0.10),
]


class HQDatasetGenerator:
    """Generate high-quality examples with 5 core tools."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.tools = ToolRegistry.get_all_tools()
        self.validator = DatasetValidator()
        self._used_intents: dict[str, set[str]] = {}
        self._example_count = 0

    def _pick_tool(self) -> ToolSchema:
        # 10 tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite, TodoRead, Web_Search, Web_Fetch
        weights = [1.2, 1.5, 1.0, 1.2, 1.0, 1.0, 0.4, 0.4, 0.8, 0.6]
        return random.choices(self.tools, weights=weights, k=1)[0]

    def _pick_difficulty(self) -> DifficultyLevel:
        r = random.random()
        cum = 0.0
        for diff, prob in DIFFICULTIES:
            cum += prob
            if r < cum:
                return diff
        return DifficultyLevel.MEDIUM

    def _pick_locale(self) -> tuple:
        return random.choice(LOCALES)

    def _get_diverse_intent(self, tool_name: str) -> tuple[str, dict]:
        intents = EXPANDED_INTENTS.get(tool_name, [])
        if not intents:
            return (f"use {tool_name}", {})
        used = self._used_intents.get(tool_name, set())
        unused = [i for i in intents if i["q"] not in used]
        if unused and random.random() < 0.8:
            intent = random.choice(unused)
            used.add(intent["q"])
            self._used_intents[tool_name] = used
        else:
            intent = random.choice(intents)
        return intent["q"], intent["a"]

    def generate_one(self, include_error: bool = False) -> DatasetExample:
        tool = self._pick_tool()
        difficulty = self._pick_difficulty()
        lang, tone, formality = self._pick_locale()

        loc = Localization(
            language=lang, tone=tone, formality=formality,
            humanize=True, humanize_level="medium",
        )

        query_hint, intent_args = self._get_diverse_intent(tool.name)

        system_prompt = SystemPromptGenerator.generate(loc)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query_hint),
        ]

        success = not include_error or random.random() > 0.15

        tc_id = "call_%s" % uuid.uuid4().hex[:12]
        tool_call_content = json.dumps({
            "type": "tool_call",
            "id": tc_id,
            "tool_name": tool.name,
            "arguments": intent_args,
        }, ensure_ascii=False)

        tool_response = ResponseGenerator.generate(tool, intent_args, success)
        tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", tc_id)

        messages.append(Message(role="assistant", content=tool_call_content))
        messages.append(Message(role="tool", content=tool_result_content, tool_call_id=tc_id, name=tool.name))

        final = FinalAnswerGenerator.generate(tool, intent_args, success, loc.language)
        messages.append(Message(role="assistant", content=final))

        self._example_count += 1
        return DatasetExample(
            messages=messages,
            localization=loc,
            tools=[t.to_openai_format() for t in self.tools],
            metadata={
                "difficulty": difficulty.value,
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
        max_attempts = count * 10
        while len(examples) < count and attempts < max_attempts:
            attempts += 1
            ex = self.generate_one(include_error=include_error)
            ok, _ = self.validator.validate_example(ex)
            if ok:
                examples.append(ex)
        return examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate 100k high-quality dataset")
    parser.add_argument("--count", type=int, default=100000, help="Number of examples")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--error-rate", type=float, default=0.15, help="Error rate")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    random.seed(args.seed)
    gen = HQDatasetGenerator(seed=args.seed)

    print("=" * 70)
    print("HIGH-QUALITY DATASET GENERATOR (5 TOOLS)")
    print(f"Target: {args.count:,} examples, error rate: {args.error_rate:.0%}")
    print(f"Tools: Bash, Read, Write, Glob, Web")
    print("=" * 70)

    total = args.count
    batch_size = 5000
    train_path = OUTPUT_DIR / "hq_100k_train.jsonl"
    eval_path = OUTPUT_DIR / "hq_100k_eval.jsonl"
    test_path = OUTPUT_DIR / "hq_100k_test.jsonl"

    all_valid = []
    generated = 0
    batch_num = 0

    while generated < total:
        batch_num += 1
        remaining = total - generated
        current_batch = min(batch_size, remaining)
        print(f"\nBatch {batch_num}: generating {current_batch}...", end="", flush=True)
        batch = gen.generate_batch(current_batch, include_error=(args.error_rate > 0))
        generated += len(batch)
        all_valid.extend(batch)
        print(f" done. Total: {generated}/{total}")

        if len(all_valid) >= total:
            break

    print(f"\nTotal valid: {len(all_valid)}")

    # Split: 70% train, 15% eval, 15% test
    random.shuffle(all_valid)
    all_valid = all_valid[:total]
    n_train = int(total * 0.70)
    n_eval = int(total * 0.15)
    train = all_valid[:n_train]
    eval_ = all_valid[n_train:n_train + n_eval]
    test = all_valid[n_train + n_eval:]

    for path, data in [(train_path, train), (eval_path, eval_), (test_path, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"Saved {len(data):,} examples to {path}")

    # Stats
    tool_counts = {}
    diff_counts = {}
    lang_counts = {}
    for ex in all_valid:
        tn = ex.metadata.get("tool_name", "?")
        tool_counts[tn] = tool_counts.get(tn, 0) + 1
        diff_counts[ex.metadata.get("difficulty", "?")] = diff_counts.get(ex.metadata.get("difficulty", "?"), 0) + 1
        lang_counts[ex.localization.language] = lang_counts.get(ex.localization.language, 0) + 1

    print(f"\n{'='*70}")
    print("DISTRIBUTION")
    print("=" * 70)
    print("\nTools:")
    for tn, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_valid) * 100
        bar = "█" * int(pct / 2)
        print(f"  {tn:15s}: {cnt:6d} ({pct:5.1f}%) {bar}")
    print(f"\nDifficulty:")
    for d, cnt in sorted(diff_counts.items()):
        print(f"  {d:10s}: {cnt:6d} ({cnt/len(all_valid)*100:.1f}%)")
    print(f"\nLanguages (top 8):")
    for l, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"  {l:5s}: {cnt:6d} ({cnt/len(all_valid)*100:.1f}%)")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
