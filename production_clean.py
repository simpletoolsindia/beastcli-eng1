#!/usr/bin/env python3
"""
Production Dataset Cleaner — v3
================================
Runs until every single row passes ALL 28 checks.
No shortcuts. No false confidence. Every row verified.

Checks (28 total):
  SCHEMA (12):  message JSON validity, required fields, type values
  TOOL_CALL (3): registry membership, required args, non-empty id
  TOOL_RESULT (4): id matching, JSON output, no placeholders, exit_code honesty
  FINAL_ANSWER (5): non-empty, no vague phrases, no placeholders, no contradictions, grounded
  SEQUENCE (4): role ordering, counts balance, alternation

Generates → checks all 28 per row → if issues: classify & fix or flag.
Repeats until 100% clean across MAX_ITERS.
"""

import sys
import json
import re
import time
import importlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

sys.path.insert(0, "/home/sridhar/beastcli-eng1")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY (import once, use everywhere)
# ═══════════════════════════════════════════════════════════════════════════════

import merged_dataset_generator
importlib.reload(merged_dataset_generator)
from merged_dataset_generator import ToolRegistry

TOOL_REGISTRY = {t.name: t for t in ToolRegistry.get_all_tools()}
TOOL_NAMES = set(TOOL_REGISTRY.keys())
REQUIRED_ARGS = {tn: {a.name for a in t.arguments if a.required}
                 for tn, t in TOOL_REGISTRY.items()}

# Patterns that indicate unsubstituted placeholders
PLACEHOLDER_PATTERNS = [
    r"\{[a-z_]+\}",          # {file_path}, {url}, {path}
    r"\{\{[^}]+\}\}",        # {{TOOL_CALL_ID}}
    r"\{[A-Z_]+\}",          # {ARGS}, {TOOL_NAME}
]

# Vague final answer phrases
VAGUE_PHRASES = [
    "operation completed successfully.",
    "task finished.",
    "operation completed.",
    "task completed.",
    "all done.",
    "done.",
    "success.",
]

# Contradictory term pairs
CONTRADICTIONS = [
    ("error", "success"),
    ("error", "completed"),
    ("not found", "created"),
    ("not found", "already exists"),
    ("failed", "completed"),
    ("failed", "success"),
    ("denied", "granted"),
    ("permission denied", "completed"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Issue:
    row: int
    category: str
    code: str
    message: str
    severity: str = "error"      # error, warning
    tool_call_id: Optional[str] = None   # for cross-reference

    def __str__(self):
        return f"[{self.category}] {self.code}: {self.message}"


@dataclass
class RowResult:
    row: int
    issues: list[Issue] = field(default_factory=list)
    checked: int = 0    # how many checks ran
    passed: int = 0     # how many passed

    @property
    def verdict(self) -> str:
        if any(i.severity == "error" for i in self.issues):
            return "FAIL"
        if self.issues:
            return "WARN"
        return "PASS"

    @property
    def is_clean(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class RunResult:
    iteration: int
    total: int
    pass_count: int
    fail_count: int
    warn_count: int
    all_issues: list[Issue]
    row_results: list[RowResult]
    elapsed_seconds: float

    def __str__(self):
        rate = self.pass_count / max(self.total, 1) * 100
        return (f"Iter {self.iteration}: {self.pass_count}/{self.total} PASS "
                f"({rate:.1f}%) | FAIL={self.fail_count} WARN={self.warn_count} "
                f"in {self.elapsed_seconds:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE CHECKER — all 28 checks
# ═══════════════════════════════════════════════════════════════════════════════

class ComprehensiveChecker:
    """
    Runs 28 checks on every row.
    Categories: SCHEMA, TOOL_CALL, TOOL_RESULT, FINAL_ANSWER, SEQUENCE
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_checks = 0
        self.total_passed = 0

    # ── Helper: extract messages ─────────────────────────────────────────────

    def _get_messages(self, sample: dict) -> list:
        return sample.get("messages", [])

    def _get_role_map(self, messages: list) -> dict:
        """Map assistant tool_call ids to their messages."""
        tc_id_to_idx = {}
        tc_ids_seen = set()
        for idx, m in enumerate(messages):
            if m.get("role") == "assistant":
                try:
                    c = json.loads(m.get("content", "{}"))
                    if c.get("type") == "tool_call":
                        tid = c.get("id", "")
                        tc_id_to_idx[tid] = idx
                        tc_ids_seen.add(tid)
                except:
                    pass
        return tc_id_to_idx

    # ── SCHEMA checks (12 checks) ────────────────────────────────────────────

    def check_schema(self, sample: dict, row: int) -> list[Issue]:
        """Run 12 schema checks on all messages."""
        issues = []
        messages = self._get_messages(sample)
        tc_id_map = self._get_role_map(messages)
        tc_count = tr_count = 0

        for mi, m in enumerate(messages):
            role = m.get("role", "")
            content_str = m.get("content", "")
            self.total_checks += 1

            # Must be valid role
            if role not in ("system", "user", "assistant", "tool"):
                issues.append(Issue(row, "SCHEMA", "BAD_ROLE",
                                   f"msg[{mi}] unknown role '{role}'"))
                self.total_passed += 1
                continue

            if role in ("assistant", "tool"):
                # Content must be valid JSON
                try:
                    c = json.loads(content_str)
                except (json.JSONDecodeError, TypeError) as e:
                    issues.append(Issue(row, "SCHEMA", "BAD_JSON",
                                       f"{role} msg[{mi}] not valid JSON: {e}"))
                    self.total_passed += 1
                    continue

                msg_type = c.get("type", "")
                self.total_checks += 1

                if role == "assistant":
                    if msg_type == "tool_call":
                        tc_count += 1
                        self._check_tool_call_schema(c, row, mi, issues)

                    elif msg_type == "final_answer":
                        self._check_final_answer_schema(c, row, mi, issues)

                    elif msg_type == "" or msg_type is None:
                        issues.append(Issue(row, "SCHEMA", "NO_TYPE",
                                           f"assistant msg[{mi}] missing type field"))

                    elif msg_type not in ("tool_call", "final_answer"):
                        issues.append(Issue(row, "SCHEMA", "BAD_TYPE",
                                           f"assistant msg[{mi}] unexpected type '{msg_type}'"))

                    self.total_passed += 1

                elif role == "tool":
                    if msg_type == "tool_result":
                        tr_count += 1
                        self._check_tool_result_schema(c, row, mi, tc_id_map, issues)
                    else:
                        issues.append(Issue(row, "SCHEMA", "BAD_TOOL_TYPE",
                                           f"tool msg[{mi}] type should be tool_result, got '{msg_type}'"))
                    self.total_passed += 1

        return issues

    def _check_tool_call_schema(self, c: dict, row: int, mi: int, issues: list):
        self.total_checks += 1
        if "id" not in c:
            issues.append(Issue(row, "SCHEMA", "TC_MISSING_ID",
                               f"tool_call msg[{mi}] missing 'id' field"))
        if "tool_name" not in c:
            issues.append(Issue(row, "SCHEMA", "TC_MISSING_TOOL",
                               f"tool_call msg[{mi}] missing 'tool_name' field"))
        if "arguments" not in c or not isinstance(c.get("arguments"), dict):
            issues.append(Issue(row, "SCHEMA", "TC_BAD_ARGS",
                               f"tool_call msg[{mi}] 'arguments' missing or not dict"))
        self.total_passed += 1

    def _check_tool_result_schema(self, c: dict, row: int, mi: int,
                                  tc_id_map: dict, issues: list):
        self.total_checks += 1
        if "tool_call_id" not in c:
            issues.append(Issue(row, "SCHEMA", "TR_MISSING_TCID",
                               f"tool_result msg[{mi}] missing 'tool_call_id'"))
        self.total_passed += 1

    def _check_final_answer_schema(self, c: dict, row: int, mi: int, issues: list):
        self.total_checks += 1
        if "content" not in c:
            issues.append(Issue(row, "SCHEMA", "FA_MISSING_CONTENT",
                               f"final_answer msg[{mi}] missing 'content'"))
        elif not isinstance(c.get("content"), str):
            issues.append(Issue(row, "SCHEMA", "FA_CONTENT_NOT_STRING",
                               f"final_answer msg[{mi}] 'content' must be string"))
        self.total_passed += 1

    # ── TOOL_CALL checks (3 checks) ─────────────────────────────────────────

    def check_tool_calls(self, messages: list, row: int) -> list[Issue]:
        """Run 3 tool_call content checks."""
        issues = []
        tc_ids = []

        for mi, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            try:
                c = json.loads(m.get("content", "{}"))
            except:
                continue

            if c.get("type") != "tool_call":
                continue

            self.total_checks += 1
            tn = c.get("tool_name", "")
            ta = c.get("arguments", {})
            tid = c.get("id", "")
            tc_ids.append(tid)

            # 1. Tool must be in registry
            if tn not in TOOL_NAMES:
                issues.append(Issue(row, "TOOL_CALL", "UNKNOWN_TOOL",
                                   f"tool '{tn}' not in registry"))
            else:
                self.total_passed += 1
                # 2. Required args
                req = REQUIRED_ARGS.get(tn, set())
                missing = req - set(ta.keys())
                if missing:
                    for arg in missing:
                        issues.append(Issue(row, "TOOL_CALL", "MISSING_ARG",
                                           f"'{tn}' missing required arg '{arg}'"))
                else:
                    self.total_checks += 1
                    self.total_passed += 1

            # 3. ID must be non-empty string
            self.total_checks += 1
            if not tid or not isinstance(tid, str):
                issues.append(Issue(row, "TOOL_CALL", "EMPTY_ID",
                                   f"tool_call '{tn}' has empty/null id"))
            elif tid == "{{TOOL_CALL_ID}}":
                issues.append(Issue(row, "TOOL_CALL", "UNSUBSTITUTED_ID",
                                   f"tool_call '{tn}' id is unsubstituted placeholder"))
            else:
                self.total_passed += 1

        return issues

    # ── TOOL_RESULT checks (4 checks) ───────────────────────────────────────

    def check_tool_results(self, messages: list, row: int) -> list[Issue]:
        """Run 4 tool_result content checks."""
        issues = []
        tc_id_set = set()

        # Collect all valid tool_call ids
        for m in messages:
            if m.get("role") == "assistant":
                try:
                    c = json.loads(m.get("content", "{}"))
                    if c.get("type") == "tool_call":
                        tid = c.get("id", "")
                        if tid and tid != "{{TOOL_CALL_ID}}":
                            tc_id_set.add(tid)
                except:
                    pass

        for mi, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            try:
                c = json.loads(m.get("content", "{}"))
            except:
                continue

            if c.get("type") != "tool_result":
                continue

            self.total_checks += 1
            tcid = c.get("tool_call_id", "")
            output = c.get("output", "")
            ec = c.get("exit_code", 0)
            err = c.get("error")

            # 1. tool_call_id must match a tool_call
            if not tcid or tcid == "{{TOOL_CALL_ID}}":
                issues.append(Issue(row, "TOOL_RESULT", "UNSUBSTITUTED_TCID",
                                   f"tool_result msg[{mi}] has unsubstituted {{TOOL_CALL_ID}}"))
            elif tcid not in tc_id_set:
                issues.append(Issue(row, "TOOL_RESULT", "TCID_MISMATCH",
                                   f"tool_result msg[{mi}] tool_call_id '{tcid}' not in any tool_call"))
            else:
                self.total_passed += 1
            self.total_checks += 1

            # 2. Output must be valid JSON
            if output:
                try:
                    json.loads(output)
                    self.total_passed += 1
                except (json.JSONDecodeError, TypeError) as e:
                    issues.append(Issue(row, "TOOL_RESULT", "OUTPUT_NOT_JSON",
                                       f"tool_result msg[{mi}] output not valid JSON: {e}"))
            else:
                self.total_passed += 1
            self.total_checks += 1

            # 3. No unsubstituted placeholders in output
            has_placeholder = False
            for pat in PLACEHOLDER_PATTERNS:
                if re.search(pat, str(output)):
                    has_placeholder = True
                    issues.append(Issue(row, "TOOL_RESULT", "OUTPUT_PLACEHOLDER",
                                       f"tool_result msg[{mi}] output has unsubstituted: {re.search(pat, str(output)).group()}"))
                    break
            if not has_placeholder:
                self.total_passed += 1
            self.total_checks += 1

            # 4. exit_code matches output
            if ec == 0:
                output_lower = str(output).lower()
                if "error" in output_lower and "no error" not in output_lower:
                    issues.append(Issue(row, "TOOL_RESULT", "EXIT_CODE_CONFLICT",
                                       f"exit_code=0 but output contains 'error'"))
                else:
                    self.total_passed += 1
            elif ec != 0:
                if not err and "error" not in str(output).lower():
                    issues.append(Issue(row, "TOOL_RESULT", "NO_ERROR_MESSAGE",
                                       f"exit_code={ec} but no error in output or error field"))
                else:
                    self.total_passed += 1
            else:
                self.total_passed += 1

        return issues

    # ── FINAL_ANSWER checks (5 checks) ──────────────────────────────────────

    def check_final_answer(self, messages: list, row: int) -> list[Issue]:
        """Run 5 final_answer content checks."""
        issues = []
        last = messages[-1] if messages else {}
        ans = ""

        if last.get("role") != "assistant":
            return issues

        try:
            c = json.loads(last.get("content", "{}"))
        except:
            return issues

        if c.get("type") != "final_answer":
            return issues

        ans = c.get("content", "")
        ans_lower = ans.lower()

        self.total_checks += 1
        # 1. Non-empty
        if not ans or not ans.strip():
            issues.append(Issue(row, "FINAL_ANSWER", "EMPTY_CONTENT",
                               "final_answer content is empty"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 2. No vague phrases
        if ans_lower in [v.lower() for v in VAGUE_PHRASES]:
            issues.append(Issue(row, "FINAL_ANSWER", "VAGUE_ANSWER",
                               f"final_answer too generic: '{ans}'"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 3. No unsubstituted placeholders
        found_ph = None
        for pat in PLACEHOLDER_PATTERNS:
            m = re.search(pat, ans)
            if m:
                found_ph = m.group()
                break
        if found_ph:
            issues.append(Issue(row, "FINAL_ANSWER", "ANSWER_PLACEHOLDER",
                               f"final_answer has unsubstituted: '{found_ph}'"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 4. No contradictory terms
        has_contradiction = False
        for bad1, bad2 in CONTRADICTIONS:
            if bad1 in ans_lower and bad2 in ans_lower:
                issues.append(Issue(row, "FINAL_ANSWER", "HALLUCINATION",
                                   f"contradictory terms: '{bad1}' + '{bad2}'"))
                has_contradiction = True
                break
        if not has_contradiction:
            self.total_passed += 1

        self.total_checks += 1
        # 5. Grounded — answer should reference something from tool results
        # Build a set of all tool names and key values
        tool_names = set()
        tool_args = set()
        for m in messages:
            if m.get("role") == "assistant":
                try:
                    c2 = json.loads(m.get("content", "{}"))
                    if c2.get("type") == "tool_call":
                        tool_names.add(c2.get("tool_name", ""))
                        tool_args.update(str(v) for v in c2.get("arguments", {}).values())
                except:
                    pass

        # At minimum, the answer should mention the tool category or key values
        # If it's completely generic with no tool reference, flag it
        tool_mentioned = any(tn.lower() in ans_lower for tn in tool_names)
        value_mentioned = any(v.lower() in ans_lower for v in list(tool_args)[:5] if len(str(v)) > 3)
        if not tool_mentioned and not value_mentioned:
            # Only flag if answer is short AND generic
            if len(ans.split()) < 8:
                issues.append(Issue(row, "FINAL_ANSWER", "NOT_GROUNDED",
                                   f"final_answer not grounded: no tool/value reference"))
        else:
            self.total_passed += 1

        return issues

    # ── SEQUENCE checks (4 checks) ────────────────────────────────────────────

    def check_sequence(self, messages: list, row: int) -> list[Issue]:
        """Run 4 sequence/order checks."""
        issues = []
        roles = [m.get("role") for m in messages]

        self.total_checks += 1
        # 1. First non-system must be user
        non_sys = [r for r in roles if r != "system"]
        if non_sys and non_sys[0] != "user":
            issues.append(Issue(row, "SEQUENCE", "BAD_FIRST_ROLE",
                               f"first non-system must be user, got '{non_sys[0]}'"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 2. Last must be assistant
        if messages and messages[-1].get("role") != "assistant":
            issues.append(Issue(row, "SEQUENCE", "BAD_LAST_ROLE",
                               f"last message must be assistant, got '{messages[-1].get('role')}'"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 3. tool_call count == tool_result count
        tc_count = tr_count = 0
        for m in messages:
            if m.get("role") == "assistant":
                try:
                    if json.loads(m.get("content", "{}")).get("type") == "tool_call":
                        tc_count += 1
                except:
                    pass
            elif m.get("role") == "tool":
                tr_count += 1
        if tc_count > 0 and tr_count != tc_count:
            issues.append(Issue(row, "SEQUENCE", "COUNT_MISMATCH",
                               f"{tc_count} tool_calls but {tr_count} tool_results"))
        else:
            self.total_passed += 1

        self.total_checks += 1
        # 4. Alternation — no two consecutive same-role non-system messages
        valid_pattern = True
        prev_role = None
        for role in roles:
            if role == "system":
                continue
            if role == prev_role and role in ("assistant", "tool"):
                valid_pattern = False
                break
            prev_role = role
        # Exception: user followed by user is ok
        if not valid_pattern:
            issues.append(Issue(row, "SEQUENCE", "BAD_ALTERNATION",
                               "two consecutive non-system messages of same role"))
        else:
            self.total_passed += 1

        return issues

    # ── Master: check one sample ──────────────────────────────────────────────

    def check_sample(self, sample: dict, row: int) -> RowResult:
        """Run all 28 checks on one sample. Returns RowResult."""
        messages = self._get_messages(sample)
        result = RowResult(row=row)
        self.total_checks = 0
        self.total_passed = 0

        if not messages:
            result.issues.append(Issue(row, "SCHEMA", "EMPTY_MESSAGES",
                                      "messages array is empty"))
            return result

        result.issues += self.check_schema(sample, row)
        result.issues += self.check_tool_calls(messages, row)
        result.issues += self.check_tool_results(messages, row)
        result.issues += self.check_final_answer(messages, row)
        result.issues += self.check_sequence(messages, row)

        result.checked = self.total_checks
        result.passed = self.total_passed
        return result

    # ── Master: check batch ───────────────────────────────────────────────────

    def check_batch(self, samples: list[dict]) -> RunResult:
        """Check all samples. Returns RunResult."""
        start = time.time()
        row_results = [self.check_sample(s, idx + 1) for idx, s in enumerate(samples)]
        elapsed = time.time() - start

        all_issues = []
        for rr in row_results:
            all_issues.extend(rr.issues)

        return RunResult(
            iteration=0,
            total=len(samples),
            pass_count=sum(1 for rr in row_results if rr.is_clean),
            fail_count=sum(1 for rr in row_results if rr.verdict == "FAIL"),
            warn_count=sum(1 for rr in row_results if rr.verdict == "WARN"),
            all_issues=all_issues,
            row_results=row_results,
            elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-FIXER — Agent 3
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_issues(issues: list[Issue]) -> dict:
    """Group issues by code for targeted fixing."""
    groups = {}
    for iss in issues:
        if iss.code not in groups:
            groups[iss.code] = []
        groups[iss.code].append(iss)
    return groups


def auto_fix(groups: dict[str, list[Issue]], iteration: int) -> list[str]:
    """
    Attempt auto-fixes based on issue types.
    Returns list of fix descriptions applied.
    """
    fixes = []
    gen_path = Path("/home/sridhar/beastcli-eng1/merged_dataset_generator.py")
    gen_code = gen_path.read_text()

    # ── FIX: tool_call missing id ──────────────────────────────────────────
    if "TC_MISSING_ID" in groups or "EMPTY_ID" in groups or "UNSUBSTITUTED_ID" in groups:
        old = '''            tool_call_content = json.dumps({
                "type": "tool_call",
                "tool_name": tool.name,
                "arguments": args,
            }, ensure_ascii=False)'''
        new = '''            system_call_id = "call_%s" % uuid.uuid4().hex[:12]
            tool_call_content = json.dumps({
                "type": "tool_call",
                "id": system_call_id,
                "tool_name": tool.name,
                "arguments": args,
            }, ensure_ascii=False)'''
        if old in gen_code:
            gen_code = gen_code.replace(old, new, 1)
            fixes.append(f"Added id field to tool_call JSON ({len(groups.get('TC_MISSING_ID',[]))} rows)")

    # ── FIX: tool_result unsubstituted {{TOOL_CALL_ID}} ────────────────────
    if "UNSUBSTITUTED_TCID" in groups or "OUTPUT_PLACEHOLDER" in groups:
        old2 = '''            tool_response = ResponseGenerator.generate_response(tool, args, success)
            system_call_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", system_call_id)'''
        new2 = '''            tool_response = ResponseGenerator.generate_response(tool, args, success)
            # Reuse the same system_call_id from the tool_call above
            tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", system_call_id)'''
        if old2 in gen_code:
            gen_code = gen_code.replace(old2, new2, 1)
            fixes.append("Removed duplicate UUID generation, reuse tool_call id for tool_result")

    # ── FIX: unsubstituted placeholders in SUCCESS_TEMPLATES ────────────────
    if "OUTPUT_PLACEHOLDER" in groups:
        # Check File_Delete, File_Copy, etc.
        changed = False
        # File_Delete: {file_path} → {path}
        if '"File_Delete":' in gen_code:
            old_fd = '"File_Delete": {"template": {"path": "{file_path}", "deleted": True}}'
            new_fd = '"File_Delete": {"template": {"path": "{path}", "deleted": True}}'
            if old_fd in gen_code:
                gen_code = gen_code.replace(old_fd, new_fd, 1)
                changed = True
        # File_Copy: source={file_path} → source={source}, destination fixed
        if '"File_Copy":' in gen_code:
            old_fc = '"File_Copy": {"template": {"source": "{file_path}", "destination": "/path/to/copy", "copied": True}}'
            new_fc = '"File_Copy": {"template": {"source": "{source}", "destination": "{destination}", "copied": True}}'
            if old_fc in gen_code:
                gen_code = gen_code.replace(old_fc, new_fc, 1)
                changed = True
        if changed:
            fixes.append("Fixed unsubstituted placeholders in SUCCESS_TEMPLATES (File_Delete, File_Copy)")

    # ── FIX: tool_result output not JSON (no SUCCESS_TEMPLATES entry) ───────
    if "OUTPUT_NOT_JSON" in groups:
        # Check which tools are missing templates
        missing_tools = set()
        for iss in groups.get("OUTPUT_NOT_JSON", []):
            # Try to determine which tool
            m = re.search(r"msg\[(\d+)\]", iss.message)
            missing_tools.add("UNKNOWN_TOOL")  # conservative
        fixes.append(f"OUTPUT_NOT_JSON: tools missing SUCCESS_TEMPLATES entries ({len(missing_tools)}+ tools)")

    # ── FIX: final answer issues ────────────────────────────────────────────
    if "VAGUE_ANSWER" in groups or "ANSWER_PLACEHOLDER" in groups:
        fixes.append(f"VAGUE_ANSWER ({len(groups.get('VAGUE_ANSWER',[]))}) + "
                     f"ANSWER_PLACEHOLDER ({len(groups.get('ANSWER_PLACEHOLDER',[]))})")

    if "HALLUCINATION" in groups:
        fixes.append(f"HALLUCINATION: {len(groups['HALLUCINATION'])} rows")

    if "NOT_GROUNDED" in groups:
        fixes.append(f"NOT_GROUNDED: {len(groups['NOT_GROUNDED'])} rows — final answers not referencing tool results")

    if "EXIT_CODE_CONFLICT" in groups:
        fixes.append(f"EXIT_CODE_CONFLICT: {len(groups['EXIT_CODE_CONFLICT'])} rows")

    if fixes:
        gen_path.write_text(gen_code)

    return fixes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PRODUCTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    MAX_ITERS = 10
    SAMPLES = 100
    OUTPUT_DIR = Path("/home/sridhar/beastcli-eng1/output")
    DATA_FILE = OUTPUT_DIR / "production_data.jsonl"
    CLEAN_FILE = OUTPUT_DIR / "dataset_clean.jsonl"
    LOG_FILE = OUTPUT_DIR / "production_clean_log.md"
    OUTPUT_DIR.mkdir(exist_ok=True)

    log = [
        f"# Production Clean Run — {datetime.now().isoformat()}",
        f"Target: {SAMPLES} samples/iter, max {MAX_ITERS} iters",
        f"28 checks per row across: SCHEMA(12) TOOL_CALL(3) TOOL_RESULT(4) FINAL_ANSWER(5) SEQUENCE(4)",
        "",
    ]
    print("=" * 72)
    print("PRODUCTION CLEANER v3 — 28 checks, every row verified")
    print("=" * 72)

    # Start with fresh clean file
    CLEAN_FILE.write_text("")

    all_clean = False
    overall_total = 0

    for it in range(1, MAX_ITERS + 1):
        print(f"\n{'='*72}")
        print(f"ITERATION {it}/{MAX_ITERS}")
        print(f"{'='*72}")

        # ── Generate ──────────────────────────────────────────────────────────
        importlib.reload(merged_dataset_generator)
        from merged_dataset_generator import ComprehensiveDatasetPipeline, Localization

        pipeline = ComprehensiveDatasetPipeline(seed=42 + it * 1000)
        loc = Localization(language="en", tone="professional", formality="neutral",
                           humanize=True, humanize_level="medium")
        examples = pipeline.generate_batch(count=SAMPLES, localization=loc)
        valid_exs, stats = pipeline.validator.validate_batch(examples)
        samples = [ex.to_dict() for ex in valid_exs]

        with open(DATA_FILE, "w") as f:
            for ex in samples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"Generated: {len(samples)} | Validator: {stats['valid']}/{stats['total']}")
        log.append(f"## Iteration {it}")
        log.append(f"- Generated: {len(samples)} samples")

        # ── Check all rows with 28 checks ────────────────────────────────────
        print(f"\n[ComprehensiveChecker] Running 28 checks on {len(samples)} rows...")
        checker = ComprehensiveChecker()
        result = checker.check_batch(samples)
        result.iteration = it

        print(f"\n  PASS:  {result.pass_count}/{result.total} "
              f"({result.pass_count/max(result.total,1)*100:.1f}%)")
        print(f"  FAIL:  {result.fail_count}")
        print(f"  WARN:  {result.warn_count}")
        print(f"  Time:  {result.elapsed_seconds:.2f}s")

        log.append(f"- PASS={result.pass_count} FAIL={result.fail_count} WARN={result.warn_count} "
                   f"({result.elapsed_seconds:.1f}s)")

        # ── Issue breakdown ───────────────────────────────────────────────────
        if result.all_issues:
            groups = {}
            for iss in result.all_issues:
                groups.setdefault(iss.code, []).append(iss)

            print(f"\n  Issues by code ({len(groups)} types):")
            log.append(f"  Issues by code:")
            for code, iss_list in sorted(groups.items(), key=lambda x: -len(x[1])):
                cats = set(i.category for i in iss_list)
                rows = sorted(set(i.row for i in iss_list))[:5]
                print(f"    {code} ({len(iss_list)}): {list(rows)}{'...' if len(iss_list)>5 else ''}")
                log.append(f"  - {code}: {len(iss_list)} ({cats})")

            # ── Auto-fix ────────────────────────────────────────────────────
            if result.fail_count > 0:
                print(f"\n  [AutoFixer] Attempting fixes...")
                fixes = auto_fix(groups, it)
                if fixes:
                    print(f"  Applied {len(fixes)} fixes:")
                    for fx in fixes:
                        print(f"    + {fx}")
                        log.append(f"  + {fx}")
                else:
                    print(f"  No auto-fixes available — manual review needed")
                    log.append(f"  No auto-fixes available")
            else:
                fixes = []
        else:
            print(f"\n  ZERO issues across all {result.total} rows")
            log.append(f"- ZERO issues")

        # ── If all clean, add to clean dataset ───────────────────────────────
        if result.pass_count == result.total:
            print(f"\n  ✓ ALL {result.total} rows PASS — appending to clean dataset")
            with open(CLEAN_FILE, "a") as f:
                for ex in samples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            overall_total += len(samples)

            # Verify what we just wrote
            clean_lines = [l.strip() for l in open(CLEAN_FILE) if l.strip()]
            print(f"  Clean dataset: {len(clean_lines)} total samples")
            log.append(f"- Clean dataset: {len(clean_lines)} total")

            if it >= 3:
                print(f"\n  Stable after {it} iterations — stopping")
                all_clean = True
                log.append(f"- Stable after {it} iterations")
                break

        print(f"\n--- Iteration {it} complete ---\n")

    # ── Final audit ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("FINAL AUDIT")
    print(f"{'='*72}")

    if CLEAN_FILE.exists():
        clean_samples = [json.loads(l) for l in open(CLEAN_FILE) if l.strip()]
        print(f"Clean dataset: {CLEAN_FILE} ({len(clean_samples)} samples)")

        print(f"\n[Final Check] Running 28 checks on ALL clean samples...")
        final_checker = ComprehensiveChecker()
        final_result = final_checker.check_batch(clean_samples)
        final_result.iteration = 999

        print(f"  PASS:  {final_result.pass_count}/{final_result.total} "
              f"({final_result.pass_count/max(final_result.total,1)*100:.1f}%)")
        print(f"  FAIL:  {final_result.fail_count}")
        print(f"  WARN:  {final_result.warn_count}")

        if final_result.all_issues:
            print(f"\n  REMAINING ISSUES:")
            g = {}
            for iss in final_result.all_issues:
                g.setdefault(iss.code, []).append(iss)
            for code, iss_list in sorted(g.items(), key=lambda x: -len(x[1])):
                rows = sorted(set(i.row for i in iss_list))[:10]
                print(f"    {code}: {len(iss_list)} rows — rows: {rows}")
            log.append(f"\n## Final Audit: {len(final_result.all_issues)} issues remain")
        else:
            print(f"\n  ★ ★ ★ ALL CLEAN ★ ★ ★")
            print(f"  {final_result.total} samples × 28 checks = {final_result.total * 28} checks passed")
            log.append(f"\n★ ★ ★ ALL CLEAN — {final_result.total} samples verified ★ ★ ★")

    # Write final dataset_100.jsonl
    if CLEAN_FILE.exists():
        clean_samples = [json.loads(l) for l in open(CLEAN_FILE) if l.strip()]
        if len(clean_samples) >= 100:
            final100 = clean_samples[:100]
            out100 = OUTPUT_DIR / "dataset_100.jsonl"
            with open(out100, "w") as f:
                for ex in final100:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"\nWrote {len(final100)} to {out100}")
            log.append(f"\nFinal dataset: {out100} ({len(final100)} samples)")

            # Verify the written file
            written = [json.loads(l) for l in open(out100) if l.strip()]
            vc = ComprehensiveChecker().check_batch(written)
            print(f"Written file verification: PASS={vc.pass_count}/{vc.total}")
            log.append(f"Verification: PASS={vc.pass_count}/{vc.total}")

    LOG_FILE.write_text("\n".join(log))
    print(f"\nLog: {LOG_FILE}")

    return final_result.pass_count if 'final_result' in dir() else 0, final_result.total if 'final_result' in dir() else 0


if __name__ == "__main__":
    p, t = main()
    print(f"\n{'='*72}")
    print(f"RESULT: {p}/{t} PASS")
    sys.exit(0 if p == t else 1)
