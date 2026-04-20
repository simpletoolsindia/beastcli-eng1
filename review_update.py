#!/usr/bin/env python3
"""
Agent 2 — Review Update Module
==============================
Standalone critic agent for reviewing agentic dataset samples.
Reads samples from a JSONL file, checks each against a 10-point checklist,
and writes structured comments to a review-comments JSONL file.

Can run standalone or be imported as a module:
    from review_update import Reviewer, update_review
    reviewer = Reviewer()
    comments = reviewer.review_samples(samples)
"""

import sys
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# Import tool registry from the generator
sys.path.insert(0, str(Path(__file__).parent))
from merged_dataset_generator import ToolRegistry

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {t.name: t for t in ToolRegistry.get_all_tools()}
TOOL_NAMES = set(TOOL_REGISTRY.keys())

# Required-arg specs for tool-specific logic checks
TOOL_REQUIRED_ARGS = {
    "File_Write": {"file_path", "content"},
    "File_Copy": {"source", "destination"},
    "Web_Search": {"query"},
    "Bash_Execute": {"command"},
    "Python_Run": {"code"},
    "Node_Run": {"code"},
    "Python_Test": {"file_path"},
    "Git_Commit": {"message"},
    "Git_Branch": {"operation"},
    "Web_Fetch": {"url"},
    "Web_Screenshot": {"url"},
    "File_Delete": {"path"},
    "File_Read": {"file_path"},
    "File_List": {"directory"},
    "File_Search": {"pattern"},
    "Search_Code": {"pattern"},
    "Search_Replace": {"search", "replace"},
    "Database_Query": {"query"},
    "Database_List": {},
    "System_Info": {},
    "Process_List": {},
    "Git_Status": {},
    "Git_Log": {},
    "Git_Diff": {},
    "Git_Pull": {},
    "Git_Push": {},
    "Bash_ShellStatus": {},
}

# Contradiction pairs for hallucination detection
CONTRADICTION_PAIRS = [
    ("error", "success"),
    ("error", "completed"),
    ("not found", "created"),
    ("failed", "completed"),
    ("failed", "success"),
    ("denied", "granted"),
    ("does not exist", "successfully created"),
    ("permission denied", "completed successfully"),
    ("does not exist", "already exists"),
]

# Vague phrases that suggest a generic answer
VAGUE_PHRASES = [
    "task completed",
    "operation successful",
    "done",
    "all done",
    "successfully completed",
]


# ─── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"  # "error", "warning", "info"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class SampleComment:
    row: int
    issues: list[Issue]
    verdict: str  # "GOOD", "BAD", "EMPTY", "EXCEPTION"
    review_time: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "issues": [i.to_dict() for i in self.issues],
            "verdict": self.verdict,
            "review_time": self.review_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SampleComment":
        return cls(
            row=d["row"],
            issues=[Issue(**i) if isinstance(i, dict) else Issue(code="UNKNOWN", message=i)
                    for i in d.get("issues", [])],
            verdict=d.get("verdict", "UNKNOWN"),
            review_time=d.get("review_time", ""),
        )


# ─── Reviewer class ───────────────────────────────────────────────────────────

class Reviewer:
    """
    Reviews agentic dataset samples against a 10-point checklist.

    Usage:
        reviewer = Reviewer()
        comments = reviewer.review_samples(samples)

        # Or with a file:
        comments = reviewer.review_file("output/review_data.jsonl",
                                         "output/review_comments.jsonl")
    """

    def __init__(self):
        self.stats = {
            "total": 0,
            "good": 0,
            "bad": 0,
            "empty": 0,
            "exception": 0,
        }
        self.issue_counts: dict[str, int] = {}

    # ── Individual checks ────────────────────────────────────────────────────────

    def _check_user_query(self, messages: list[dict], row: int) -> list[Issue]:
        """Check 1: User query quality."""
        issues = []
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            issues.append(Issue("NO_USER", f"ROW[{row}] NO_USER: no user message found"))
            return issues
        user_query = user_msgs[0].get("content", "").strip()
        if not user_query:
            issues.append(Issue("EMPTY_USER", f"ROW[{row}] EMPTY_USER: user message content is empty"))
        elif len(user_query) < 10:
            issues.append(Issue("SHORT_QUERY",
                                 f"ROW[{row}] SHORT_QUERY: user query too short ({len(user_query)} chars)"))
        return issues

    def _check_tool_calls(self, messages: list[dict], row: int) -> list[Issue]:
        """Check 2: Tool call correctness + Check 3: Tool ID presence."""
        issues = []
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        tool_calls = []

        for ai, am in enumerate(assistant_msgs):
            content_str = am.get("content", "")
            if not content_str:
                continue
            try:
                content = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                issues.append(Issue("BAD_MSG_JSON",
                                     f"ROW[{row}] BAD_MSG_JSON: assistant msg {ai} not valid JSON"))
                continue

            if content.get("type") == "tool_call":
                tc = {
                    "idx": ai,
                    "name": content.get("tool_name", ""),
                    "args": content.get("arguments", {}),
                    "id": content.get("id", ""),
                }
                tool_calls.append(tc)

        for tc in tool_calls:
            tn, ta, tid, r = tc["name"], tc["args"], tc["id"], row

            # Tool must exist
            if tn not in TOOL_NAMES:
                issues.append(Issue("BAD_TOOL",
                                     f"ROW[{r}] BAD_TOOL: unknown tool '{tn}'"))
            else:
                # Required args from schema
                tool_def = TOOL_REGISTRY[tn]
                for arg in tool_def.arguments:
                    if arg.required and arg.name not in ta:
                        issues.append(Issue("MISSING_ARG",
                                             f"ROW[{r}] MISSING_ARG: '{tn}' missing required arg '{arg.name}'"))

            # ID must exist
            if not tid:
                issues.append(Issue("NO_ID",
                                     f"ROW[{r}] NO_ID: tool_call missing id for '{tn}'"))

            # Tool-specific logic checks
            required = TOOL_REQUIRED_ARGS.get(tn, set())
            for req_arg in required:
                if req_arg not in ta:
                    issues.append(Issue("MISSING_ARG",
                                         f"ROW[{r}] MISSING_ARG: '{tn}' missing '{req_arg}'"))

        return issues

    def _check_tool_results(self, messages: list[dict], row: int) -> list[Issue]:
        """Check 4: Tool result consistency."""
        issues = []
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        for ti, tm in enumerate(tool_msgs):
            content_str = tm.get("content", "")
            if not content_str:
                continue
            try:
                tc_data = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                issues.append(Issue("TOOL_MSG_BAD_JSON",
                                     f"ROW[{row}] TOOL_MSG_BAD_JSON: tool msg {ti} has bad JSON"))
                continue

            if tc_data.get("type") == "tool_result":
                tcid = tc_data.get("tool_call_id", "")
                if not tcid:
                    issues.append(Issue("NO_TOOL_CALL_ID",
                                         f"ROW[{row}] NO_TOOL_CALL_ID: tool_result missing tool_call_id"))
                output = tc_data.get("output", "")
                err = tc_data.get("error")
                ec = tc_data.get("exit_code", 0)

                # Exit code 0 = success; output shouldn't say "error"
                if ec == 0 and "error" in output.lower() and "no error" not in output.lower():
                    issues.append(Issue("TOOL_RESULT_CONFLICT",
                                         f"ROW[{row}] TOOL_RESULT_CONFLICT: exit_code=0 but output contains 'error'"))
                if err and ec == 0:
                    issues.append(Issue("TOOL_ERR_CONFLICT",
                                         f"ROW[{row}] TOOL_ERR_CONFLICT: has error msg but exit_code=0"))

        return issues

    def _check_final_answer(self, messages: list[dict], row: int) -> list[Issue]:
        """Check 5: Final answer correctness + Check 6: Hallucination + Check 7: Vague."""
        issues = []
        last = messages[-1] if messages else {}
        if last.get("role") != "assistant":
            issues.append(Issue("BAD_LAST_ROLE",
                                 f"ROW[{row}] BAD_LAST_ROLE: last must be assistant, got '{last.get('role')}'"))
            return issues

        content_str = last.get("content", "")
        try:
            content = json.loads(content_str)
        except (json.JSONDecodeError, TypeError) as e:
            issues.append(Issue("BAD_ANSWER_JSON",
                                 f"ROW[{row}] BAD_ANSWER_JSON: final_answer not valid JSON: {e}"))
            return issues

        if content.get("type") != "final_answer":
            issues.append(Issue("BAD_LAST_TYPE",
                                 f"ROW[{row}] BAD_LAST_TYPE: last must be final_answer, got '{content.get('type')}'"))
            return issues

        ans = content.get("content", "").strip()
        if not ans:
            issues.append(Issue("EMPTY_ANSWER",
                                 f"ROW[{row}] EMPTY_ANSWER: final_answer.content is empty"))
            return issues

        # Hallucination: contradiction detection
        ans_lower = ans.lower()
        for bad1, bad2 in CONTRADICTION_PAIRS:
            if bad1 in ans_lower and bad2 in ans_lower:
                issues.append(Issue("HALLUCINATION",
                                     f"ROW[{row}] HALLUCINATION: contradictory terms '{bad1}' and '{bad2}' in answer"))
            # "error while succeeded" pattern
            if bad1 in ans_lower and f"{bad1} while" in ans_lower:
                issues.append(Issue("HALLUCINATION",
                                     f"ROW[{row}] HALLUCINATION: 'while' contradiction in answer"))

        # Vague answer
        vague_count = sum(1 for v in VAGUE_PHRASES if v in ans_lower)
        if vague_count >= 2:
            issues.append(Issue("VAGUE_ANSWER",
                                 f"ROW[{row}] VAGUE_ANSWER: answer too generic ({vague_count} vague phrases)"))

        return issues

    def _check_sequence(self, messages: list[dict], row: int) -> list[Issue]:
        """Check 8: Message sequence + Check 9: Tool call/result balance."""
        issues = []
        roles = [m.get("role") for m in messages]
        non_sys = [r for r in roles if r != "system"]

        # First non-system must be user
        if non_sys and non_sys[0] != "user":
            issues.append(Issue("BAD_SEQUENCE",
                                 f"ROW[{row}] BAD_SEQUENCE: first non-system must be user, got '{non_sys[0]}'"))

        # Count tool_calls and tool_results
        tc_count = 0
        tr_count = 0
        for m in messages:
            if m.get("role") == "assistant":
                try:
                    c = json.loads(m.get("content", "{}"))
                    if c.get("type") == "tool_call":
                        tc_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            elif m.get("role") == "tool":
                tr_count += 1

        if tr_count != tc_count and tc_count > 0:
            issues.append(Issue("SEQUENCE_MISMATCH",
                                 f"ROW[{row}] SEQUENCE_MISMATCH: {tc_count} tool_calls but {tr_count} tool_results"))

        return issues

    # ── Core review ────────────────────────────────────────────────────────────

    def review_sample(self, sample: dict, row: int) -> SampleComment:
        """Review a single sample and return a SampleComment."""
        issues: list[Issue] = []

        try:
            messages = sample.get("messages", [])
            if not messages:
                issues.append(Issue("EMPTY", f"ROW[{row}] EMPTY: messages array is empty"))
                return SampleComment(row=row, issues=issues, verdict="EMPTY")

            issues += self._check_user_query(messages, row)
            issues += self._check_tool_calls(messages, row)
            issues += self._check_tool_results(messages, row)
            issues += self._check_final_answer(messages, row)
            issues += self._check_sequence(messages, row)

            verdict = "GOOD" if not issues else "BAD"

        except Exception as e:
            issues.append(Issue("EXCEPTION", f"ROW[{row}] EXCEPTION: {e}"))
            verdict = "EXCEPTION"

        return SampleComment(row=row, issues=issues, verdict=verdict)

    def review_samples(self, samples: list[dict]) -> list[SampleComment]:
        """Review all samples and return list of SampleComment objects."""
        self.stats = {"total": len(samples), "good": 0, "bad": 0, "empty": 0, "exception": 0}
        self.issue_counts = {}

        comments = []
        for idx, sample in enumerate(samples):
            comment = self.review_sample(sample, row=idx + 1)
            comments.append(comment)

            # Update stats — single dispatch per verdict
            v = comment.verdict.lower()
            if v == "good":
                self.stats["good"] += 1
            elif v == "bad":
                self.stats["bad"] += 1
            elif v == "empty":
                self.stats["empty"] += 1
            elif v == "exception":
                self.stats["exception"] += 1

            # Count issue types
            for issue in comment.issues:
                self.issue_counts[issue.code] = self.issue_counts.get(issue.code, 0) + 1

        return comments

    # ── File I/O ───────────────────────────────────────────────────────────────

    def review_file(
        self,
        data_path: str | Path,
        comment_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
    ) -> list[SampleComment]:
        """
        Load samples from a JSONL file, review them, write comments.

        Args:
            data_path: Path to input JSONL (one sample per line)
            comment_path: Path to output comments JSONL. Defaults to
                          data_path.parent / "review_comments.jsonl"
            report_path: Optional path to write a markdown report

        Returns:
            List of SampleComment objects
        """
        data_path = Path(data_path)
        if comment_path:
            comment_path = Path(comment_path)
        else:
            comment_path = data_path.parent / "review_comments.jsonl"
        if report_path:
            report_path = Path(report_path)

        # Load samples
        samples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        samples.append({"_parse_error": line})

        # Review
        comments = self.review_samples(samples)

        # Write comments
        with open(comment_path, "w") as f:
            for c in comments:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        # Write report
        report = self.build_report()
        if report_path:
            report_path.write_text(report)
        else:
            print(report)

        print(f"\nWrote {len(comments)} comments to {comment_path}")
        return comments

    def build_report(self) -> str:
        """Build a markdown report from current review stats."""
        lines = [
            f"# Review Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Total**: {self.stats['total']} | "
            f"Good: {self.stats['good']} | "
            f"Bad: {self.stats['bad']} | "
            f"Empty: {self.stats['empty']} | "
            f"Exception: {self.stats['exception']}",
            "",
        ]

        if self.issue_counts:
            lines.append("## Issues by Type")
            lines.append("")
            lines.append("| Code | Count |")
            lines.append("|------|-------|")
            for code, count in sorted(self.issue_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| `{code}` | {count} |")
            lines.append("")

        good_rate = self.stats["good"] / max(self.stats["total"], 1) * 100
        threshold = 5.0
        status = "PASS" if (100 - good_rate) < threshold else "NEEDS_FIX"
        lines.append(f"## Verdict: **{status}** (error rate: {100 - good_rate:.1f}%, threshold: <{threshold}%)")

        return "\n".join(lines)


# ─── Update review ─────────────────────────────────────────────────────────────

def update_review(
    data_path: str | Path,
    comment_path: str | Path,
    *,
    overwrite_good: bool = True,
    strict: bool = False,
) -> list[SampleComment]:
    """
    Run a review update cycle: read existing comments, re-review samples,
    and merge or overwrite results.

    Args:
        data_path: Input samples JSONL
        comment_path: Existing (or target) comments JSONL
        overwrite_good: If True, re-review GOOD rows too (in case generator changed)
        strict: If True, treat warnings as errors

    Returns:
        Updated list of SampleComment objects
    """
    reviewer = Reviewer()
    comment_path = Path(comment_path)

    # Load existing comments if file exists
    existing = {}
    if comment_path.exists():
        with open(comment_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sc = SampleComment.from_dict(json.loads(line))
                        existing[sc.row] = sc
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Load samples
    samples = []
    sample_rows = {}
    with open(data_path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                    sample_rows[len(samples)] = idx
                except json.JSONDecodeError:
                    samples.append({"_parse_error": line})

    # Review only non-GOOD rows (unless overwrite_good)
    if overwrite_good:
        all_rows = list(range(1, len(samples) + 1))
    else:
        all_rows = [r for r in range(1, len(samples) + 1) if existing.get(r, SampleComment(row=r, issues=[], verdict="")).verdict != "GOOD"]

    for row in all_rows:
        reviewer.review_sample(samples[row - 1], row)

    comments = reviewer.review_samples(samples)

    # Write back
    with open(comment_path, "w") as f:
        for c in comments:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"Updated review: {len(comments)} samples, {reviewer.stats['good']} good, "
          f"{reviewer.stats['bad']} bad")
    return comments


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent 2 — Review Update Module")
    parser.add_argument("data_file", nargs="?", default=None,
                        help="Input JSONL file with samples")
    parser.add_argument("-o", "--output", default=None,
                        help="Output comments JSONL file")
    parser.add_argument("-r", "--report", default=None,
                        help="Output markdown report file")
    parser.add_argument("--update", action="store_true",
                        help="Update mode: re-review only non-GOOD rows")
    parser.add_argument("--no-overwrite-good", action="store_true",
                        help="Don't re-review rows that were GOOD in existing comments")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-row verdicts")

    args = parser.parse_args()

    if args.data_file:
        data_path = Path(args.data_file)
    else:
        # Default: use the loop v2 data file
        data_path = OUTPUT_DIR / "review_data.jsonl"
        if not data_path.exists():
            data_path = BASE_DIR / "output" / "review_data.jsonl"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    comment_path = Path(args.output) if args.output else data_path.parent / "review_comments.jsonl"
    report_path = Path(args.report) if args.report else None

    if args.update:
        update_review(
            data_path,
            comment_path,
            overwrite_good=not args.no_overwrite_good,
        )
    else:
        reviewer = Reviewer()
        reviewer.review_file(data_path, comment_path, report_path)

        if args.verbose:
            print("\n--- Per-row verdicts ---")
            for idx, sample in enumerate(
                json.loads(line) for line in open(data_path)
                if line.strip()
            ):
                print(f"  Row {idx + 1}: {reviewer.review_sample(sample, idx + 1).verdict}")


if __name__ == "__main__":
    main()
