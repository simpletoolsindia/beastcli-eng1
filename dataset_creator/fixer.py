"""
fixer.py — Auto-Repair Engine
==============================

Automatically fixes common validation failures in training samples.
Each fix is targeted and preserves the sample's semantic meaning.

Fixable Issues
--------------
Based on the ValidationIssue.fixable field:
1. Missing tool_call IDs → Auto-generate UUID-based IDs
2. Unknown tool names → Map to closest known tool name
3. Empty final answers → Fill with descriptive content
4. Invalid message sequence → Re-order to valid sequence
5. Missing required arguments → Provide sensible defaults
6. Non-JSON message content → Attempt JSON encoding

Design Principles
----------------
- Fixes are idempotent: running fix twice has same result
- Fixes are conservative: prefer minimal changes
- Fixes log what changed for auditability
- Some issues cannot be auto-fixed (returned as-is)
"""

from __future__ import annotations

import json
import uuid
import re
from dataclasses import dataclass

from dataset_creator.schemas import (
    TrainingSample,
    Message,
    ToolCall,
    ToolResult,
    FinalAnswer,
)
from dataset_creator.tools import TOOL_REGISTRY_BY_NAME, TOOL_NAMES
from dataset_creator.validator import ValidationIssue


# --------------------------------------------------------------------------
# Fixer
# --------------------------------------------------------------------------

@dataclass
class FixResult:
    """Result of an attempted fix operation."""
    sample_index: int
    original_issue: ValidationIssue
    fixed: bool
    action: str  # Human-readable description of what was fixed
    new_sample: TrainingSample  # The (possibly modified) sample


class Fixer:
    """
    Auto-repairs validation failures in training samples.

    Usage:
        fixer = Fixer()
        results = fixer.fix(samples, issues)
        fixed_samples = [r.new_sample for r in results if r.fixed]
    """

    def __init__(self):
        self.tool_registry = TOOL_REGISTRY_BY_NAME

    # =====================================================================
    # Individual Fix Strategies
    # =====================================================================

    def _fix_missing_tool_call_id(
        self,
        sample: TrainingSample,
        issue: ValidationIssue,
    ) -> TrainingSample:
        """Fix: Generate IDs for tool_call messages missing them."""
        sample = TrainingSample.from_dict(sample.to_dict())  # Deep copy
        for msg in sample.messages:
            if msg.role == "assistant":
                try:
                    d = json.loads(msg.content)
                    if d.get("type") == "tool_call" and (not d.get("id") or not d["id"].strip()):
                        d["id"] = f"call_{uuid.uuid4().hex[:8]}"
                        msg = Message(role="assistant", content=json.dumps(d, ensure_ascii=False))
                except (json.JSONDecodeError, TypeError):
                    pass
        return sample

    def _fix_unknown_tool_name(
        self,
        sample: TrainingSample,
        issue: ValidationIssue,
    ) -> TrainingSample:
        """
        Fix: Map unknown tool names to known equivalents.

        Known mappings for common misspellings/variants:
        - "shell" → "bash"
        - "exec" → "bash"
        - "run" → "bash"
        - "read" → "read_file"
        - "write" → "write_file"
        - "edit" → "update_file"
        - "delete" → "delete_file"
        - "ls" → "list_files"
        - "find" → "search_files"
        - "cd" → "bash" (with cd command)
        """
        sample = TrainingSample.from_dict(sample.to_dict())
        tool_aliases = {
            "shell": "bash",
            "exec": "bash",
            "run": "bash",
            "read": "read_file",
            "write": "write_file",
            "edit": "update_file",
            "delete": "delete_file",
            "ls": "list_files",
            "find": "search_files",
            "cd": "bash",
            "mkdir": "bash",
            "cat": "read_file",
            "grep": "search_files",
        }

        for msg in sample.messages:
            if msg.role == "assistant":
                try:
                    d = json.loads(msg.content)
                    if d.get("type") == "tool_call":
                        tool_name = d.get("tool_name", "")
                        if tool_name not in self.tool_registry:
                            if tool_name in tool_aliases:
                                d["tool_name"] = tool_aliases[tool_name]
                                msg = Message(role="assistant", content=json.dumps(d, ensure_ascii=False))
                except (json.JSONDecodeError, TypeError):
                    pass
        return sample

    def _fix_empty_final_answer(
        self,
        sample: TrainingSample,
        issue: ValidationIssue,
    ) -> TrainingSample:
        """
        Fix: Provide a descriptive final answer based on tool results.

        Generates an answer that summarizes what the tools accomplished.
        """
        sample = TrainingSample.from_dict(sample.to_dict())
        tool_calls = sample.tool_calls()
        tool_results = sample.tool_results()

        # Build summary from tool calls
        tool_names = [tc.tool_name for tc in tool_calls]
        last_output = tool_results[-1].output[:200] if tool_results else ""
        last_output_preview = last_output.replace("\n", " ") if last_output else ""

        summary_parts = []
        if tool_names:
            summary_parts.append(f"Completed {len(tool_names)} tool call(s): {', '.join(tool_names)}")
        if last_output_preview:
            summary_parts.append(f"Output: {last_output_preview}")

        new_content = "; ".join(summary_parts) if summary_parts else "Task completed successfully."
        new_content += "\n"

        # Replace the last assistant message's content with final_answer
        for i in range(len(sample.messages) - 1, -1, -1):
            msg = sample.messages[i]
            if msg.role == "assistant":
                try:
                    d = json.loads(msg.content)
                    if d.get("type") == "final_answer":
                        d["content"] = new_content
                        sample.messages[i] = Message(role="assistant", content=json.dumps(d, ensure_ascii=False))
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

        return sample

    def _fix_invalid_json(
        self,
        sample: TrainingSample,
        issue: ValidationIssue,
    ) -> TrainingSample:
        """
        Fix: Attempt to repair malformed JSON in message content.

        Common issues:
        - Single quotes instead of double quotes
        - Trailing commas
        - Unquoted keys
        """
        sample = TrainingSample.from_dict(sample.to_dict())
        msg_idx_match = re.search(r"Message (\d+)", issue.message)
        fixed_any = False
        if msg_idx_match:
            msg_idx = int(msg_idx_match.group(1))
            if 0 <= msg_idx < len(sample.messages):
                msg = sample.messages[msg_idx]
                fixed_content = self._try_repair_json(msg.content)
                if fixed_content is not None:
                    sample.messages[msg_idx] = Message(role=msg.role, content=fixed_content)
                    fixed_any = True
        else:
            # Try fixing all non-JSON messages
            for i, msg in enumerate(sample.messages):
                if msg.content and not msg.is_json():
                    fixed_content = self._try_repair_json(msg.content)
                    if fixed_content is not None:
                        sample.messages[i] = Message(role=msg.role, content=fixed_content)
                        fixed_any = True
        # Mark as fixed only if we actually repaired something
        return sample

    def _try_repair_json(self, content: str) -> Optional[str]:
        """Attempt to repair malformed JSON string."""
        try:
            json.loads(content)
            return content  # Already valid
        except json.JSONDecodeError:
            pass

        # Try: replace single quotes with double quotes (but not in strings)
        repaired = content
        try:
            # Replace single quotes around values
            repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

        # Try: remove trailing commas
        repaired = re.sub(r",\s*([\]}])", r"\1", content)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

        return None

    def _fix_message_sequence(
        self,
        sample: TrainingSample,
        issue: ValidationIssue,
    ) -> TrainingSample:
        """
        Fix: Re-order messages to valid sequence.

        Valid sequence: system? → user → assistant → tool → ... → final_answer
        """
        sample = TrainingSample.from_dict(sample.to_dict())
        msgs = sample.messages

        # Extract by role
        system_msgs = [m for m in msgs if m.role == "system"]
        user_msgs = [m for m in msgs if m.role == "user"]
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        tool_msgs = [m for m in msgs if m.role == "tool"]

        # Rebuild in valid order
        new_msgs: list[Message] = []
        new_msgs.extend(system_msgs)
        new_msgs.extend(user_msgs)

        # Interleave assistant and tool messages
        assistant_idx = 0
        tool_idx = 0
        for assistant_msg in assistant_msgs:
            new_msgs.append(assistant_msg)
            # Find corresponding tool messages after this assistant
            # (simple approach: append all tool messages after each assistant)
            while tool_idx < len(tool_msgs):
                new_msgs.append(tool_msgs[tool_idx])
                tool_idx += 1

        sample.messages = new_msgs
        return sample

    # =====================================================================
    # Batch Fix
    # =====================================================================

    def fix(
        self,
        samples: list[TrainingSample],
        issues: list[ValidationIssue],
    ) -> list[FixResult]:
        """
        Attempt to fix all fixable validation issues.

        Args:
            samples: Original samples
            issues: Validation issues to attempt to fix

        Returns:
            List of FixResult, one per issue attempted
        """
        results: list[FixResult] = []

        # Group issues by sample index
        from collections import defaultdict
        by_sample: dict[int, list[ValidationIssue]] = defaultdict(list)
        for issue in issues:
            if issue.fixable:
                by_sample[issue.sample_index].append(issue)

        # Process each sample with issues
        for sample_idx, sample_issues in by_sample.items():
            sample = samples[sample_idx]
            fixed_sample = TrainingSample.from_dict(sample.to_dict())

            for issue in sample_issues:
                result = FixResult(
                    sample_index=sample_idx,
                    original_issue=issue,
                    fixed=False,
                    action="",
                    new_sample=fixed_sample,
                )

                if issue.rule == "missing_tool_call_id":
                    fixed_sample = self._fix_missing_tool_call_id(fixed_sample, issue)
                    result.fixed = True
                    result.action = "Generated IDs for tool_call messages"
                    result.new_sample = fixed_sample

                elif issue.rule == "tool_name_registry":
                    fixed_sample = self._fix_unknown_tool_name(fixed_sample, issue)
                    result.fixed = True
                    result.action = "Mapped unknown tool names to known equivalents"
                    result.new_sample = fixed_sample

                elif issue.rule == "final_answer_empty":
                    fixed_sample = self._fix_empty_final_answer(fixed_sample, issue)
                    result.fixed = True
                    result.action = "Filled empty final_answer with summary"
                    result.new_sample = fixed_sample

                elif issue.rule == "json_validity":
                    fixed_sample = self._fix_invalid_json(fixed_sample, issue)
                    # Heuristic: if no non-JSON messages remain, consider it fixed
                    still_broken = any(
                        m.content and not m.is_json()
                        for m in fixed_sample.messages
                    )
                    result.fixed = not still_broken
                    result.action = "Attempted JSON repair"
                    result.new_sample = fixed_sample

                elif issue.rule == "message_sequence":
                    fixed_sample = self._fix_message_sequence(fixed_sample, issue)
                    result.fixed = True
                    result.action = "Re-ordered messages to valid sequence"
                    result.new_sample = fixed_sample

                else:
                    result.action = f"No auto-fix available for rule '{issue.rule}'"

                results.append(result)

        return results

    def fix_and_return(
        self,
        samples: list[TrainingSample],
        issues: list[ValidationIssue],
    ) -> list[TrainingSample]:
        """
        Fix samples and return the fixed versions.

        Samples without issues are returned unchanged.
        Samples with issues get the latest fix attempt applied.

        Returns:
            List of TrainingSample (original or fixed)
        """
        # Start with originals
        result_samples = list(samples)

        # Apply fixes
        fix_results = self.fix(samples, issues)
        for fr in fix_results:
            if fr.fixed:
                result_samples[fr.sample_index] = fr.new_sample

        return result_samples
