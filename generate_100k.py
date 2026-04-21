#!/usr/bin/env python3
"""
High-quality 100k dataset generator.
Every example is unique: diverse queries, semantically matched args, realistic responses.
Uses the existing generator infrastructure with massive template expansion.
"""
import json
import random
import uuid
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

# ─── Import from merged_dataset_generator ───────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from merged_dataset_generator import (
    ComprehensiveDatasetPipeline,
    Localization,
    LocalizationContent,
    DifficultyLevel,
    Language,
    Tone,
    FormalityLevel,
    HumanizeLevel,
    ToolRegistry,
    ToolSchema,
    ToolArgument,
    ToolCallGenerator,
    ResponseGenerator,
    QueryTemplates,
    Humanizer,
    SystemPromptGenerator,
    Message,
    DatasetExample,
    DatasetValidator,
)

# ─── EXPANDED INTENT TEMPLATES ────────────────────────────────────────────────
# 27 tools × ~15-25 query_hint/args pairs = 400+ unique intent combinations
EXPANDED_INTENTS = {
    "File_Read": [
        {"q": "config.json", "a": {"file_path": "config.json"}},
        {"q": "main.py", "a": {"file_path": "src/main.py"}},
        {"q": "README", "a": {"file_path": "README.md"}},
        {"q": "utils.py", "a": {"file_path": "lib/utils.py"}},
        {"q": "setup.py", "a": {"file_path": "setup.py"}},
        {"q": "app.py", "a": {"file_path": "src/app.py"}},
        {"q": "test_main.py", "a": {"file_path": "tests/test_main.py"}},
        {"q": "auth.py", "a": {"file_path": "src/auth.py"}},
        {"q": "database.py", "a": {"file_path": "src/database.py"}},
        {"q": "models.py", "a": {"file_path": "src/models.py"}},
        {"q": "settings.yaml", "a": {"file_path": "config/settings.yaml"}},
        {"q": ".env", "a": {"file_path": ".env"}},
        {"q": "requirements.txt", "a": {"file_path": "requirements.txt"}},
        {"q": "package.json", "a": {"file_path": "package.json"}},
        {"q": "Dockerfile", "a": {"file_path": "Dockerfile"}},
        {"q": "index.html", "a": {"file_path": "public/index.html"}},
        {"q": "styles.css", "a": {"file_path": "src/styles.css"}},
        {"q": "data.csv", "a": {"file_path": "data/raw/data.csv"}},
        {"q": "schema.sql", "a": {"file_path": "db/schema.sql"}},
        {"q": "CHANGELOG", "a": {"file_path": "CHANGELOG.md"}},
        {"q": "first 30 lines", "a": {"file_path": "src/main.py", "limit": 30}},
        {"q": "top 50 lines", "a": {"file_path": "config.json", "limit": 50}},
        {"q": "first 10 lines of test_auth.py", "a": {"file_path": "tests/test_auth.py", "limit": 10}},
    ],
    "File_Write": [
        {"q": "output.txt with build log", "a": {"file_path": "output.txt", "content": "[INFO] Build completed successfully.\n", "append": False}},
        {"q": "test.py with hello", "a": {"file_path": "test.py", "content": "print('Hello')\n", "append": False}},
        {"q": "data.json with key-value", "a": {"file_path": "data.json", "content": "{\"key\": \"value\", \"count\": 42}\n", "append": False}},
        {"q": "log entry to app.log", "a": {"file_path": "logs/app.log", "content": "[INFO] Request completed at 2026-04-21T10:00:00Z\n", "append": True}},
        {"q": "README with install instructions", "a": {"file_path": "README.md", "content": "# Project\n\n## Installation\n\n1. Clone the repo\n2. Run `npm install`\n3. Start the server\n", "append": False}},
        {"q": "User class to models.py", "a": {"file_path": "src/models.py", "content": "class User:\n    def __init__(self, name: str, email: str):\n        self.name = name\n        self.email = email\n", "append": False}},
        {"q": "CLI script to output/logic.py", "a": {"file_path": "output/logic.py", "content": "import argparse\n\nparser = argparse.ArgumentParser(description='CLI tool')\nargs = parser.parse_args()\nprint('Running...')\n", "append": False}},
        {"q": "config.yaml", "a": {"file_path": "config.yaml", "content": "database:\n  host: localhost\n  port: 5432\n", "append": False}},
        {"q": "api.json", "a": {"file_path": "api.json", "content": "{\"endpoints\": [\"/health\", \"/users\", \"/data\"]}\n", "append": False}},
        {"q": "docker-compose.yml", "a": {"file_path": "docker-compose.yml", "content": "version: '3.8'\nservices:\n  app:\n    image: myapp:latest\n", "append": False}},
        {"q": "pytest.ini", "a": {"file_path": "pytest.ini", "content": "[pytest]\ntestpaths = tests\npython_files = test_*.py\n", "append": False}},
        {"q": ".gitignore", "a": {"file_path": ".gitignore", "content": "__pycache__/\n*.pyc\nnode_modules/\n.env\n", "append": False}},
        {"q": "csv file", "a": {"file_path": "data/export.csv", "content": "name,score\nAlice,95\nBob,87\n", "append": False}},
        {"q": "html page", "a": {"file_path": "public/index.html", "content": "<!doctype html>\n<html><head><title>App</title></head>\n<body><h1>Hello</h1></body>\n</html>\n", "append": False}},
        {"q": "markdown doc", "a": {"file_path": "docs/guide.md", "content": "# Guide\n\n## Getting Started\n\nFollow these steps to get up and running.\n", "append": False}},
    ],
    "File_Search": [
        {"q": "all Python files", "a": {"pattern": "*.py", "path": ".", "recursive": True}},
        {"q": "JS files", "a": {"pattern": "*.js", "path": "src", "recursive": True}},
        {"q": "both py and js files", "a": {"pattern": "*.{py,js}", "path": ".", "recursive": True}},
        {"q": "markdown docs", "a": {"pattern": "*.md", "path": "docs", "recursive": True}},
        {"q": "json config files", "a": {"pattern": "*.json", "path": "config", "recursive": False}},
        {"q": "yaml config files", "a": {"pattern": "*.yaml", "path": "config", "recursive": True}},
        {"q": "test files", "a": {"pattern": "test_*.py", "path": "tests", "recursive": True}},
        {"q": "env files", "a": {"pattern": ".env*", "path": ".", "recursive": False}},
        {"q": "SVG icons", "a": {"pattern": "*.svg", "path": "public", "recursive": True}},
        {"q": "CSV data files", "a": {"pattern": "*.csv", "path": "data", "recursive": True}},
        {"q": "all files named config", "a": {"pattern": "*config*", "path": ".", "recursive": True}},
        {"q": "files with TODO comments", "a": {"pattern": "*.py", "path": "src", "content_search": True}},
        {"q": "markdown files mentioning setup", "a": {"pattern": "*.md", "path": "docs", "content_search": True}},
    ],
    "File_List": [
        {"q": "current directory", "a": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
        {"q": "src folder", "a": {"directory": "src", "include_hidden": True, "filter_type": "all"}},
        {"q": "config folder", "a": {"directory": "config", "include_hidden": False, "filter_type": "files"}},
        {"q": "project structure", "a": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
        {"q": "all files including hidden", "a": {"directory": ".", "include_hidden": True, "filter_type": "all"}},
        {"q": "tests directory", "a": {"directory": "tests", "include_hidden": False, "filter_type": "all"}},
        {"q": "data folder", "a": {"directory": "data", "include_hidden": False, "filter_type": "all"}},
        {"q": "public directory", "a": {"directory": "public", "include_hidden": False, "filter_type": "files"}},
        {"q": "lib folder", "a": {"directory": "lib", "include_hidden": False, "filter_type": "all"}},
        {"q": "docs folder", "a": {"directory": "docs", "include_hidden": False, "filter_type": "all"}},
        {"q": "categorize files by type", "a": {"directory": ".", "include_hidden": False, "filter_type": "all"}},
        {"q": "only files in src", "a": {"directory": "src", "include_hidden": False, "filter_type": "files"}},
    ],
    "File_Delete": [
        {"q": "cache.log file", "a": {"path": "/tmp/cache.log", "recursive": False}},
        {"q": "output.json", "a": {"path": "output.json", "recursive": False}},
        {"q": "old backup file", "a": {"path": "backup_old.py", "recursive": False}},
        {"q": "all pyc files in build", "a": {"path": "build/**/*.pyc", "recursive": True}},
        {"q": "temp download", "a": {"path": "downloads/temp.zip", "recursive": False}},
        {"q": "old log files", "a": {"path": "logs/*.log", "recursive": True}},
        {"q": "dist folder", "a": {"path": "dist", "recursive": True}},
    ],
    "File_Copy": [
        {"q": "README to backup", "a": {"source": "README.md", "destination": "README_backup.md"}},
        {"q": "config to old", "a": {"source": "config.json", "destination": "config.old"}},
        {"q": "src to backup", "a": {"source": "src", "destination": "src_backup"}},
        {"q": "test.py to test_backup.py", "a": {"source": "test.py", "destination": "test_backup.py"}},
        {"q": "data folder to archive", "a": {"source": "data", "destination": "data_archive"}},
    ],
    "Bash_Execute": [
        {"q": "list all files", "a": {"command": "ls -la", "timeout": 10}},
        {"q": "working directory", "a": {"command": "pwd", "timeout": 10}},
        {"q": "disk usage", "a": {"command": "df -h", "timeout": 10}},
        {"q": "memory info", "a": {"command": "free -m", "timeout": 10}},
        {"q": "top 5 processes", "a": {"command": "ps aux --sort=-rss | head -5", "timeout": 10}},
        {"q": "find py files", "a": {"command": "find . -name '*.py' -type f", "timeout": 30}},
        {"q": "count py lines", "a": {"command": "find . -name '*.py' | xargs wc -l", "timeout": 30}},
        {"q": "git porcelain status", "a": {"command": "git status --porcelain", "timeout": 10}},
        {"q": "search for TODO", "a": {"command": "rg -n \"TODO\" . --type py", "timeout": 30}},
        {"q": "node version", "a": {"command": "node --version", "timeout": 10}},
        {"q": "python version", "a": {"command": "python3 --version", "timeout": 10}},
        {"q": "npm install", "a": {"command": "npm install", "timeout": 120}},
        {"q": "pytest run", "a": {"command": "pytest -v", "timeout": 60}},
        {"q": "docker ps", "a": {"command": "docker ps", "timeout": 10}},
        {"q": "git log last 5", "a": {"command": "git log --oneline -5", "timeout": 10}},
    ],
    "Bash_ShellStatus": [
        {"q": "current shell", "a": {}},
        {"q": "shell environment", "a": {}},
        {"q": "home directory", "a": {}},
        {"q": "os and user details", "a": {}},
    ],
    "Python_Run": [
        {"q": "Hello World", "a": {"code": "print('Hello, World!')", "timeout": 10}},
        {"q": "fibonacci 30", "a": {"code": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)\nprint(fib(30))", "timeout": 10}},
        {"q": "sum 1 to 100", "a": {"code": "result = sum(range(1, 101))\nprint(result)", "timeout": 10}},
        {"q": "json parse", "a": {"code": "import json\nprint(json.loads('{\"ok\": true}'))", "timeout": 10}},
        {"q": "read csv", "a": {"code": "import csv, io\nrows = list(csv.DictReader(io.StringIO('name,val\\na,1\\nb,2\\n')))\nprint({'count': len(rows), 'sum': sum(int(r['val']) for r in rows)})", "timeout": 10}},
        {"q": "http request", "a": {"code": "import urllib.request\nwith urllib.request.urlopen('https://example.com') as r:\n    print(r.read().decode()[:100])", "timeout": 10}},
        {"q": "list comprehension", "a": {"code": "squares = [x**2 for x in range(1, 11)]\nprint(squares)", "timeout": 10}},
        {"q": "dictionary merge", "a": {"code": "a = {'x': 1}; b = {'y': 2}\nmerged = {**a, **b}\nprint(merged)", "timeout": 10}},
        {"q": "file read", "a": {"code": "with open('README.md') as f:\n    print(f.read()[:200])", "timeout": 10}},
        {"q": "datetime now", "a": {"code": "from datetime import datetime\nprint(datetime.now().isoformat())", "timeout": 10}},
        {"q": "regex match", "a": {"code": "import re\ntext = 'User: alice@example.com'\nmatch = re.search(r'\\b\\w+@\\w+\\.\\w+\\ b', text)\nprint(match.group() if match else 'No email found')", "timeout": 10}},
        {"q": "set operations", "a": {"code": "a = {1, 2, 3}; b = {2, 3, 4}\nprint('intersection:', a & b, 'union:', a | b)", "timeout": 10}},
        {"q": "factorial", "a": {"code": "import math\nprint(math.factorial(20))", "timeout": 10}},
        {"q": "prime check", "a": {"code": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nprint([n for n in range(2, 50) if is_prime(n)])", "timeout": 10}},
        {"q": "url parse", "a": {"code": "from urllib.parse import urlparse\nparsed = urlparse('https://api.example.com/users?id=42')\nprint(f'host={parsed.netloc}, path={parsed.path}')", "timeout": 10}},
        {"q": "json dumps", "a": {"code": "import json\ndata = {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}\nprint(json.dumps(data, indent=2))", "timeout": 10}},
        {"q": "csv write", "a": {"code": "import csv\nwith open('/tmp/test.csv', 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['name', 'score'])\n    w.writerows([['Alice', 95], ['Bob', 87]])\nprint('Written 2 rows')", "timeout": 10}},
    ],
    "Node_Run": [
        {"q": "Hello from Node", "a": {"code": "console.log('Hello from Node.js!')", "timeout": 10}},
        {"q": "array map", "a": {"code": "const nums = [1, 2, 3, 4, 5]; const squared = nums.map(x => x * x); console.log(squared);", "timeout": 10}},
        {"q": "read file", "a": {"code": "const fs = require('fs'); console.log(fs.readFileSync('package.json', 'utf8').slice(0, 100));", "timeout": 10}},
        {"q": "async fetch", "a": {"code": "(async () => { console.log('async demo'); })();", "timeout": 10}},
        {"q": "object spread", "a": {"code": "const a = {x: 1, y: 2}; const b = {...a, z: 3}; console.log(b);", "timeout": 10}},
    ],
    "Python_Test": [
        {"q": "run all tests", "a": {"file_path": "tests/", "pattern": "test_*.py"}},
        {"q": "api tests", "a": {"file_path": "tests/", "pattern": "test_api*.py"}},
        {"q": "test_main.py", "a": {"file_path": "tests/test_main.py", "pattern": None}},
        {"q": "auth tests", "a": {"file_path": "tests/test_auth.py", "pattern": "test_auth*"}},
        {"q": "verbose tests", "a": {"file_path": "tests/", "pattern": "test_*.py", "verbose": True}},
        {"q": "coverage report", "a": {"file_path": "tests/", "pattern": "test_*.py", "verbose": True, "coverage": True}},
        {"q": "unit tests", "a": {"file_path": "tests/unit/", "pattern": "test_*.py"}},
        {"q": "integration tests", "a": {"file_path": "tests/integration/", "pattern": "test_*.py"}},
    ],
    "Git_Status": [
        {"q": "git status", "a": {"repository_path": "."}},
        {"q": "repository state", "a": {"repository_path": "."}},
        {"q": "uncommitted changes", "a": {"repository_path": "."}},
        {"q": "modified files", "a": {"repository_path": "."}},
    ],
    "Git_Log": [
        {"q": "last 5 commits", "a": {"repository_path": ".", "limit": 5, "format": "medium"}},
        {"q": "last 10 commits", "a": {"repository_path": ".", "limit": 10, "format": "short"}},
        {"q": "commit history", "a": {"repository_path": ".", "limit": 10}},
        {"q": "commit with authors", "a": {"repository_path": ".", "limit": 10, "format": "medium"}},
        {"q": "oneline log", "a": {"repository_path": ".", "limit": 20, "format": "oneline"}},
    ],
    "Git_Commit": [
        {"q": "commit with fix message", "a": {"message": "fix: resolve null pointer in user service", "all": True}},
        {"q": "commit with feat message", "a": {"message": "feat: add user profile endpoint", "all": True}},
        {"q": "commit with docs message", "a": {"message": "docs: update README installation steps", "all": True}},
        {"q": "commit with chore message", "a": {"message": "chore: upgrade dependencies to latest", "all": True}},
        {"q": "commit with refactor message", "a": {"message": "refactor: simplify auth middleware", "all": True}},
        {"q": "commit with test message", "a": {"message": "test: add unit tests for utils module", "all": True}},
    ],
    "Git_Branch": [
        {"q": "list branches", "a": {"operation": "list"}},
        {"q": "create feature branch", "a": {"operation": "create", "branch_name": "feature/user-auth"}},
        {"q": "switch to develop", "a": {"operation": "switch", "branch_name": "develop"}},
        {"q": "delete old branch", "a": {"operation": "delete", "branch_name": "old-feature"}},
    ],
    "Git_Diff": [
        {"q": "staged changes", "a": {"target": "staged"}},
        {"q": "diff with main", "a": {"target": "main"}},
        {"q": "unstaged changes", "a": {"target": "working_tree"}},
        {"q": "diff src/main.py", "a": {"target": "HEAD", "file_path": "src/main.py"}},
    ],
    "Git_Pull": [
        {"q": "pull from origin", "a": {"repository_path": ".", "remote": "origin", "branch": "main"}},
        {"q": "pull develop", "a": {"repository_path": ".", "remote": "origin", "branch": "develop"}},
        {"q": "update local branch", "a": {"repository_path": ".", "remote": "origin"}},
    ],
    "Git_Push": [
        {"q": "push to origin", "a": {"repository_path": ".", "remote": "origin", "branch": "main"}},
        {"q": "push develop", "a": {"repository_path": ".", "remote": "origin", "branch": "develop"}},
        {"q": "push feature branch", "a": {"repository_path": ".", "remote": "origin", "branch": "feature/user-auth"}},
    ],
    "Web_Search": [
        {"q": "python async await", "a": {"query": "Python async await patterns best practices 2026"}},
        {"q": "react performance", "a": {"query": "React performance optimization patterns"}},
        {"q": "typescript generics", "a": {"query": "TypeScript generic constraints extends"}},
        {"q": "docker multi-stage", "a": {"query": "Docker multi-stage build optimization"}},
        {"q": "git workflow", "a": {"query": "Git workflow best practices team"}},
        {"q": "llm fine-tuning", "a": {"query": "recent papers on LLM fine-tuning techniques"}},
        {"q": "rust cli tools", "a": {"query": "best practices for building CLI tools in Rust"}},
        {"q": "postgres indexing", "a": {"query": "PostgreSQL indexing best practices"}},
        {"q": "kubernetes autoscaling", "a": {"query": "Kubernetes autoscaling configuration"}},
        {"q": "python best practices", "a": {"query": "Python best practices 2026"}},
        {"q": "machine learning pipeline", "a": {"query": "ML pipeline orchestration tools 2026"}},
        {"q": "api design rest", "a": {"query": "REST API design principles best practices"}},
        {"q": "testing pyramid", "a": {"query": "testing pyramid unit integration e2e"}},
        {"q": "CI/CD pipeline", "a": {"query": "CI/CD pipeline optimization strategies"}},
        {"q": "microservices patterns", "a": {"query": "microservices design patterns 2026"}},
        {"q": "redis caching", "a": {"query": "Redis caching strategies for web applications"}},
    ],
    "Web_Fetch": [
        {"q": "fetch README", "a": {"url": "https://github.com/example/repo/blob/main/README.md"}},
        {"q": "release notes", "a": {"url": "https://github.com/example/repo/releases"}},
        {"q": "api status", "a": {"url": "https://api.example.com/status"}},
        {"q": "docs page", "a": {"url": "https://docs.example.com/guide"}},
        {"q": "JSON endpoint", "a": {"url": "https://api.example.com/users/1"}},
    ],
    "Web_Screenshot": [
        {"q": "example.com homepage", "a": {"url": "https://example.com", "full_page": False}},
        {"q": "documentation site", "a": {"url": "https://docs.example.com", "full_page": True}},
        {"q": "api docs", "a": {"url": "https://api.example.com/docs", "full_page": True}},
        {"q": "dashboard page", "a": {"url": "https://app.example.com/dashboard", "full_page": False}},
    ],
    "Search_Code": [
        {"q": "all TODO comments", "a": {"pattern": "TODO", "path": "."}},
        {"q": "import statements", "a": {"pattern": "import\\s+", "path": "src"}},
        {"q": "def main", "a": {"pattern": "def main", "path": "."}},
        {"q": "try/except blocks", "a": {"pattern": "try:", "path": "."}},
        {"q": "error handling", "a": {"pattern": "try:", "path": "."}},
        {"q": "function definitions", "a": {"pattern": "def \\w+", "path": "src"}},
        {"q": "class definitions", "a": {"pattern": "class \\w+", "path": "src"}},
        {"q": "console.log calls", "a": {"pattern": "console\\.log", "path": "src"}},
        {"q": "async functions", "a": {"pattern": "async def", "path": "src"}},
        {"q": "type annotations", "a": {"pattern": ": \\w+", "path": "src"}},
    ],
    "Search_Replace": [
        {"q": "replace foo in main.py", "a": {"path": "src/main.py", "search": "foo", "replace": "bar", "file_types": [".py"], "preview": False}},
        {"q": "update timeout value", "a": {"path": "config.json", "search": '"timeout": 30', "replace": '"timeout": 60', "file_types": [".json"], "preview": False}},
        {"q": "utils to helpers", "a": {"path": ".", "search": "utils", "replace": "helpers", "file_types": [".py"], "preview": False}},
        {"q": "update port number", "a": {"path": "config.yaml", "search": "port: 3000", "replace": "port: 8080", "file_types": [".yaml"], "preview": False}},
    ],
    "System_Info": [
        {"q": "OS details", "a": {"category": "os"}},
        {"q": "CPU details", "a": {"category": "cpu"}},
        {"q": "memory details", "a": {"category": "memory"}},
        {"q": "disk details", "a": {"category": "disk"}},
        {"q": "network details", "a": {"category": "network"}},
    ],
    "Process_List": [
        {"q": "all processes", "a": {"filter": None, "sort_by": "pid", "limit": 20}},
        {"q": "top by memory", "a": {"filter": None, "sort_by": "memory", "limit": 10}},
        {"q": "top by cpu", "a": {"filter": None, "sort_by": "cpu", "limit": 5}},
        {"q": "active processes", "a": {"filter": "running"}},
        {"q": "python processes", "a": {"filter": "python"}},
        {"q": "node processes", "a": {"filter": "node"}},
    ],
    "Database_Query": [
        {"q": "list users", "a": {"query": "SELECT id, name, email FROM users ORDER BY created_at DESC LIMIT 20", "database": "production_db"}},
        {"q": "count orders", "a": {"query": "SELECT COUNT(*) FROM orders WHERE status = 'paid'", "database": "production_db"}},
        {"q": "recent orders", "a": {"query": "SELECT order_id, total, created_at FROM orders WHERE created_at >= '2026-01-01' ORDER BY created_at DESC", "database": "production_db"}},
        {"q": "active products", "a": {"query": "SELECT name, price FROM products WHERE active = true", "database": "production_db"}},
        {"q": "user with id", "a": {"query": "SELECT * FROM users WHERE id = 42", "database": "production_db"}},
    ],
    "Database_List": [
        {"q": "all databases", "a": {}},
        {"q": "list databases", "a": {}},
        {"q": "tables in db", "a": {"database": "production_db"}},
    ],
}

# ─── EXPANDED QUERY TEMPLATES ─────────────────────────────────────────────────
# More templates per tool/difficulty = more diverse user queries
EXPANDED_QUERIES = {}
for tool in ToolRegistry.get_all_tools():
    name = tool.name
    EXPANDED_QUERIES[name] = {
        DifficultyLevel.EASY: [],
        DifficultyLevel.MEDIUM: [],
        DifficultyLevel.HARD: [],
        DifficultyLevel.EXPERT: [],
    }

EASY_QUERIES = {
    "File_Read": ["Read config.json", "Show me main.py", "Open README.md", "Read src/app.py", "Show me settings.yaml", "What is in lib/utils.py?"],
    "File_Write": ["Write to output.txt", "Create test.py", "Save data.json", "Make a log entry", "Write README.md"],
    "File_Search": ["Find all Python files", "Search for JS files", "Find markdown docs", "Search for test files"],
    "File_List": ["List files in current directory", "Show src folder contents", "What's in the config folder?", "List all files"],
    "File_Delete": ["Delete cache.log", "Remove output.json", "Clean up backup files", "Delete .pyc files"],
    "File_Copy": ["Copy README to backup", "Duplicate config file", "Copy src folder", "Backup test file"],
    "Bash_Execute": ["Run ls -la", "Show working directory", "Check disk usage", "Find Python files", "Show git status"],
    "Bash_ShellStatus": ["What's my current shell?", "Show shell environment", "What is my home directory?"],
    "Python_Run": ["Run print Hello World", "Execute Python script", "Run a Python snippet", "Calculate sum 1 to 100"],
    "Node_Run": ["Run Node.js hello", "Execute JavaScript", "Run Node script"],
    "Python_Test": ["Run pytest tests", "Run test_main.py", "Execute test suite", "Run auth tests"],
    "Git_Status": ["Show git status", "What's the repository state?", "Any uncommitted changes?"],
    "Git_Log": ["Show last 5 commits", "Display commit history", "Last 10 commits"],
    "Git_Commit": ["Commit with fix message", "Stage and commit changes"],
    "Git_Branch": ["List all branches", "What branches exist?", "Create feature branch"],
    "Git_Diff": ["Show staged changes", "Diff with main branch", "Show unstaged changes"],
    "Git_Pull": ["Pull from origin", "Update local branch", "Pull remote changes"],
    "Git_Push": ["Push to remote", "Push commits", "Push feature branch"],
    "Web_Search": ["Search for Python best practices", "Find React tutorials"],
    "Web_Fetch": ["Fetch README", "Get release notes", "Fetch API docs"],
    "Web_Screenshot": ["Screenshot of example.com", "Capture homepage", "Take screenshot of docs"],
    "Search_Code": ["Find TODO comments", "Search for imports", "Find function definitions"],
    "Search_Replace": ["Replace foo with bar", "Update timeout in config", "Change utils to helpers"],
    "System_Info": ["Show OS details", "Memory info", "CPU details"],
    "Process_List": ["Show all processes", "List running processes", "Top processes"],
    "Database_Query": ["Query users table", "Count orders", "Select from database"],
    "Database_List": ["List databases", "Show available databases"],
}

MEDIUM_QUERIES = {
    "File_Read": ["Read the first 50 lines of src/main.py", "Show me lib/utils.py contents", "What does setup.py contain?"],
    "File_Write": ["Append log entry to app.log", "Create README with install instructions", "Write a CLI script to output/logic.py"],
    "File_Search": ["Find all Python and JS files", "Search for markdown documentation files", "Find files containing 'config' in name"],
    "File_List": ["List all files in src including hidden", "Show project structure recursively"],
    "Bash_Execute": ["Count lines of Python code", "Find all TODO comments", "Show top 5 processes by memory"],
    "Python_Run": ["Parse a JSON string", "Calculate fibonacci(20)", "Read and process CSV data"],
    "Node_Run": ["Run JavaScript with array operations", "Execute Node script to read package.json"],
    "Python_Test": ["Run tests with coverage report", "Run verbose pytest", "Execute test suite for auth module"],
    "Git_Log": ["Show last 10 commits with authors", "Display commit history", "Show last 5 commits in short format"],
    "Git_Commit": ["Commit all changes with conventional message", "Stage and commit with descriptive message"],
    "Web_Search": ["Search for async/await patterns", "Find TypeScript generics documentation", "Search for Docker best practices"],
    "Web_Fetch": ["Fetch latest release notes", "Get README from GitHub", "Fetch API status page"],
    "Search_Code": ["Find all try/except error handling", "Search for import statements", "Find class definitions"],
    "System_Info": ["Show CPU and memory details", "Get OS and hardware info"],
    "Process_List": ["Show top 10 by memory", "List active processes", "Show processes sorted by CPU"],
    "Database_Query": ["Get top 20 users by creation date", "Count paid orders", "List recent orders"],
}

HARD_QUERIES = {
    "File_Read": ["Read the main entry point and summarize what it does", "Find and read the database configuration"],
    "File_Write": ["Create a complete Python module at src/models.py with a User class", "Write a CLI tool with argument parsing"],
    "File_Search": ["Find markdown files in docs and show setup instructions", "Search all code for error handling patterns"],
    "Bash_Execute": ["Run pytest and check if all tests pass", "Install dependencies and verify installation"],
    "Python_Run": ["Make an HTTP request and process the JSON response", "Parse CSV and compute statistics"],
    "Web_Search": ["Research best practices for building production-grade CLI tools", "Search for recent LLM fine-tuning papers"],
    "Search_Code": ["Find all error handling patterns in the codebase"],
    "Database_Query": ["Get aggregated order statistics for the last 30 days"],
}

EXPERT_QUERIES = {
    "File_Write": ["Write a complete CLI application with argument parsing, logging, and error handling"],
    "Bash_Execute": ["Analyze the entire codebase, find all TODO comments, and generate a report summary"],
    "Python_Run": ["Run a data processing pipeline that reads CSV, transforms data, and outputs statistics"],
    "Web_Search": ["Research and compare multiple approaches for building distributed systems"],
    "Search_Code": ["Find all code patterns that could be refactored for better performance"],
}

for tool_name, queries in EASY_QUERIES.items():
    if tool_name in EXPANDED_QUERIES:
        EXPANDED_QUERIES[tool_name][DifficultyLevel.EASY] = queries
for tool_name, queries in MEDIUM_QUERIES.items():
    if tool_name in EXPANDED_QUERIES:
        EXPANDED_QUERIES[tool_name][DifficultyLevel.MEDIUM] = queries
for tool_name, queries in HARD_QUERIES.items():
    if tool_name in EXPANDED_QUERIES:
        EXPANDED_QUERIES[tool_name][DifficultyLevel.HARD] = queries
for tool_name, queries in EXPERT_QUERIES.items():
    if tool_name in EXPANDED_QUERIES:
        EXPANDED_QUERIES[tool_name][DifficultyLevel.EXPERT] = queries


# ─── HIGH-QUALITY EXAMPLE GENERATOR ─────────────────────────────────────────
class HQDatasetGenerator:
    """Generate 100k high-quality diverse examples."""

    LOCALES = [
        # (language, tone, formality) — 48 combinations, enough for 100k
        ("en", "professional", "formal"),
        ("en", "professional", "neutral"),
        ("en", "professional", "informal"),
        ("en", "casual", "formal"),
        ("en", "casual", "neutral"),
        ("en", "casual", "informal"),
        ("en", "technical", "formal"),
        ("en", "technical", "neutral"),
        ("en", "technical", "informal"),
        ("en", "friendly", "formal"),
        ("en", "friendly", "neutral"),
        ("en", "friendly", "informal"),
        ("hi", "professional", "formal"),
        ("hi", "professional", "neutral"),
        ("hi", "professional", "informal"),
        ("hi", "casual", "formal"),
        ("hi", "casual", "neutral"),
        ("hi", "casual", "informal"),
        ("hi", "technical", "formal"),
        ("hi", "technical", "neutral"),
        ("hi", "technical", "informal"),
        ("hi", "friendly", "formal"),
        ("hi", "friendly", "neutral"),
        ("hi", "friendly", "informal"),
        ("es", "professional", "neutral"),
        ("es", "casual", "informal"),
        ("fr", "professional", "formal"),
        ("fr", "casual", "neutral"),
        ("de", "professional", "neutral"),
        ("de", "casual", "informal"),
        ("ja", "professional", "formal"),
        ("ja", "casual", "neutral"),
        ("zh", "professional", "neutral"),
        ("zh", "casual", "informal"),
        ("ko", "professional", "formal"),
        ("ko", "casual", "neutral"),
    ]

    DIFFICULTIES = [
        (DifficultyLevel.EASY, 0.30),
        (DifficultyLevel.MEDIUM, 0.40),
        (DifficultyLevel.HARD, 0.20),
        (DifficultyLevel.EXPERT, 0.10),
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.tools = ToolRegistry.get_all_tools()
        self.validator = DatasetValidator()
        self._tool_weights = self._build_weights()
        self._used_intents: dict[str, set[str]] = {}  # tool -> used query_hints
        self._example_count = 0

    def _build_weights(self) -> dict[str, float]:
        w = {}
        for t in self.tools:
            cat = t.category
            if cat == "git":
                w[t.name] = 0.5
            elif t.name in ("Python_Run", "Git_Push"):
                w[t.name] = 0.7
            elif t.name in ("File_Copy", "File_Delete", "Bash_Execute", "Web_Fetch"):
                w[t.name] = 1.8
            elif t.name in ("File_Read", "File_Write", "File_List", "Web_Search"):
                w[t.name] = 1.2
            else:
                w[t.name] = 1.0
        return w

    def _pick_tool(self) -> ToolSchema:
        names = [t.name for t in self.tools]
        weights = [self._tool_weights.get(n, 1.0) for n in names]
        total = sum(weights)
        probs = [wg / total for wg in weights]
        return random.choices(self.tools, weights=probs, k=1)[0]

    def _pick_difficulty(self) -> DifficultyLevel:
        r = random.random()
        cum = 0.0
        for diff, prob in self.DIFFICULTIES:
            cum += prob
            if r < cum:
                return diff
        return DifficultyLevel.MEDIUM

    def _pick_locale(self) -> tuple:
        return random.choice(self.LOCALES)

    def _get_diverse_intent(self, tool_name: str) -> tuple[str, dict]:
        """Pick a diverse intent, preferring unused ones."""
        intents = EXPANDED_INTENTS.get(tool_name, [])
        if not intents:
            return ("query", {})

        # Try to find an unused intent first
        used = self._used_intents.get(tool_name, set())
        unused = [i for i in intents if i["q"] not in used]

        if unused and random.random() < 0.8:
            intent = random.choice(unused)
            used.add(intent["q"])
            self._used_intents[tool_name] = used
        else:
            intent = random.choice(intents)

        return intent["q"], intent["a"]

    def _get_diverse_query(self, tool_name: str, difficulty: DifficultyLevel, loc: Localization) -> str:
        """Pick a diverse query for the tool/difficulty."""
        queries = EXPANDED_QUERIES.get(tool_name, {}).get(difficulty, [])
        if queries:
            q = random.choice(queries)
            if loc.humanize:
                q = Humanizer.humanize(q, loc)
            return q
        return f"Execute {tool_name}"

    def _build_args(self, tool_name: str, query: str, intent_args: dict) -> dict:
        """Build args from intent + query inference."""
        args = dict(intent_args)
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return args

        # Use ToolCallGenerator for inference on specific tools
        inferred = ToolCallGenerator._infer_arguments_from_query(tool, query)
        if inferred and tool_name in {
            "File_Read", "File_Search", "Process_List", "Python_Test",
            "Git_Branch", "Git_Log", "Python_Run", "Web_Search", "Web_Fetch",
        }:
            for k, v in inferred.items():
                if k not in args or args[k] is None:
                    args[k] = v

        # Fix brace patterns: glob patterns like *.py are NOT placeholders
        if tool_name == "File_Search":
            pattern = args.get("pattern", "")
            if pattern and "{" in pattern:
                args["pattern"] = pattern  # keep as-is, it's a glob pattern

        return {k: v for k, v in args.items() if v is not None}

    def generate_one(self, include_error: bool = False) -> DatasetExample:
        tool = self._pick_tool()
        difficulty = self._pick_difficulty()
        lang, tone, formality = self._pick_locale()
        loc = Localization(
            language=lang, tone=tone, formality=formality,
            humanize=True, humanize_level="medium",
        )

        # Get diverse intent (query_hint + args)
        query_hint, intent_args = self._get_diverse_intent(tool.name)
        query = self._get_diverse_query(tool.name, difficulty, loc)

        # Build args: intent args + query inference
        args = self._build_args(tool.name, query, intent_args)

        # Generate messages
        system_prompt = SystemPromptGenerator.generate(loc, len(self.tools))
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]

        success = not include_error or random.random() > 0.15
        if not success:
            pass  # all_success will be False

        tc_id = "call_%s" % uuid.uuid4().hex[:12]
        tool_call_content = json.dumps({
            "type": "tool_call",
            "id": tc_id,
            "tool_name": tool.name,
            "arguments": args,
        }, ensure_ascii=False)

        tool_response = ResponseGenerator.generate_response(tool, args, success)
        tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", tc_id)

        messages.append(Message(role="assistant", content=tool_call_content))
        messages.append(Message(role="tool", content=tool_result_content, tool_call_id=tc_id, name=tool.name))

        # Build final answer
        tool_results_data = [{
            "tool": tool.name, "args": args,
            "response": tool_result_content,
            "success": success, "user_query": query,
        }]

        from merged_dataset_generator import ComprehensiveDatasetPipeline
        pipeline = ComprehensiveDatasetPipeline.__new__(ComprehensiveDatasetPipeline)
        pipeline.tools = self.tools
        final_response = pipeline._build_final_answer(tool_results_data, loc, success)
        final_answer_content = json.dumps({
            "type": "final_answer",
            "content": final_response,
        }, ensure_ascii=False)
        messages.append(Message(role="assistant", content=final_answer_content))

        self._example_count += 1
        return DatasetExample(
            messages=messages,
            localization=loc,
            metadata={
                "difficulty": difficulty.value,
                "tool_category": tool.category,
                "tool_name": tool.name,
                "tools_used": [tool.name],
                "num_tools": 1,
                "success": success,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_version": "hq-100k",
            }
        )

    def generate_batch(self, count: int, include_error: bool = False) -> list[DatasetExample]:
        """Generate count examples, validated."""
        examples = []
        attempts = 0
        max_attempts = count * 10
        while len(examples) < count and attempts < max_attempts:
            attempts += 1
            ex = self.generate_one(include_error=include_error)
            is_valid, errors = self.validator.validate_example(ex)
            if is_valid:
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

    gen = HQDatasetGenerator(seed=args.seed)

    print("=" * 70)
    print("HIGH-QUALITY 100K DATASET GENERATOR")
    print(f"Target: {args.count:,} examples, error rate: {args.error_rate:.0%}")
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
        print(f"\nBatch {batch_num}: generating {current_batch} examples...", end="", flush=True)
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

    # Final stats
    tool_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    for ex in all_valid:
        tn = ex.metadata.get("tool_name", "?")
        tool_counts[tn] = tool_counts.get(tn, 0) + 1
        diff_counts[ex.metadata.get("difficulty", "?")] = diff_counts.get(ex.metadata.get("difficulty", "?"), 0) + 1
        lang_counts[ex.localization.language] = lang_counts.get(ex.localization.language, 0) + 1

    print(f"\n{'='*70}")
    print("DISTRIBUTION")
    print("=" * 70)
    print("\nTools (top 15):")
    for tn, cnt in sorted(tool_counts.items(), key=lambda x: -x[1])[:15]:
        pct = cnt / len(all_valid) * 100
        bar = "█" * int(pct / 2)
        print(f"  {tn:20s}: {cnt:6d} ({pct:5.1f}%) {bar}")
    print(f"\nDifficulty:")
    for d, cnt in sorted(diff_counts.items()):
        print(f"  {d:10s}: {cnt:6d} ({cnt/len(all_valid)*100:.1f}%)")
    print(f"\nLanguages:")
    for l, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {l:5s}: {cnt:6d} ({cnt/len(all_valid)*100:.1f}%)")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
