from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib import error, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
OUTPUT_DIR = Path("output/verifications")


@dataclass
class VerificationIssue:
    severity: str
    focus: str
    comment: str


@dataclass
class VerificationResult:
    row_number: int
    tool_name: str
    user_message: str
    is_correct: bool
    summary: str
    llm_response: str
    issues: list[VerificationIssue]

    def to_rows(self) -> list[dict]:
        if not self.issues:
            return [{
                "row_number": self.row_number,
                "tool_name": self.tool_name,
                "user_message": self.user_message,
                "is_correct": self.is_correct,
                "severity": "",
                "focus": "",
                "comment": "",
                "summary": self.summary,
                "llm_response": self.llm_response,
            }]
        rows = []
        for issue in self.issues:
            rows.append({
                "row_number": self.row_number,
                "tool_name": self.tool_name,
                "user_message": self.user_message,
                "is_correct": self.is_correct,
                "severity": issue.severity,
                "focus": issue.focus,
                "comment": issue.comment,
                "summary": self.summary,
                "llm_response": self.llm_response,
            })
        return rows


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return items


def discover_jsonl_files(root: str | Path = "output") -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.rglob("*.jsonl"))


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    payload = _get_json(f"{base_url.rstrip('/')}/api/tags")
    models = []
    for item in payload.get("models", []):
        name = item.get("name")
        if name:
            models.append(name)
    return models


def verify_dataset(
    input_path: str | Path,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_rows: Optional[int] = None,
    only_incorrect: bool = True,
    progress_callback: Optional[Callable[[int, int, VerificationResult], None]] = None,
) -> tuple[list[VerificationResult], Path]:
    samples = load_jsonl(input_path)
    if max_rows is not None:
        samples = samples[:max_rows]

    results: list[VerificationResult] = []
    total = len(samples)
    for index, sample in enumerate(samples, 1):
        result = verify_sample(index, sample, model=model, base_url=base_url)
        if not only_incorrect or not result.is_correct:
            results.append(result)
        if progress_callback:
            progress_callback(index, total, result)

    csv_path = write_results_csv(results, input_path)
    return results, csv_path


def verify_sample(row_number: int, sample: dict, model: str, base_url: str = DEFAULT_OLLAMA_URL) -> VerificationResult:
    messages = sample.get("messages", [])
    user_message = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
    tool_call_message = next(
        (msg for msg in messages if msg.get("role") == "assistant" and _assistant_message_type(msg.get("content", "")) == "tool_call"),
        None,
    )
    tool_result_message = next((msg for msg in messages if msg.get("role") == "tool"), None)
    final_answer_message = next(
        (msg for msg in reversed(messages) if msg.get("role") == "assistant" and _assistant_message_type(msg.get("content", "")) == "final_answer"),
        None,
    )

    tool_call = _safe_json(tool_call_message.get("content", "") if tool_call_message else "")
    tool_result = _safe_json(tool_result_message.get("content", "") if tool_result_message else "")
    final_answer = _safe_json(final_answer_message.get("content", "") if final_answer_message else "")

    prompt = build_verification_prompt(
        row_number=row_number,
        sample=sample,
        user_message=user_message,
        tool_call=tool_call,
        tool_result=tool_result,
        final_answer=final_answer,
    )
    llm_response_text, llm_response = generate_ollama_json(prompt=prompt, model=model, base_url=base_url)
    return normalize_verification_result(
        row_number=row_number,
        user_message=user_message,
        tool_name=tool_call.get("tool_name", sample.get("metadata", {}).get("tool_name", "")),
        payload=llm_response,
        llm_response_text=llm_response_text,
    )


def build_verification_prompt(
    row_number: int,
    sample: dict,
    user_message: str,
    tool_call: dict,
    tool_result: dict,
    final_answer: dict,
) -> str:
    review_target = {
        "row_number": row_number,
        "user_message": user_message,
        "system_message": next((msg.get("content", "") for msg in sample.get("messages", []) if msg.get("role") == "system"), ""),
        "tool_call": tool_call,
        "tool_result": tool_result,
        "final_answer": final_answer,
        "metadata": sample.get("metadata", {}),
    }
    return (
        "You are a strict dataset verification reviewer for tool-calling training data.\n"
        "Review whether the assistant behavior is correct for this single row.\n"
        "Focus on:\n"
        "1. whether the tool call matches the user's request\n"
        "2. whether arguments are grounded in the conversation\n"
        "3. whether the tool result looks consistent with the tool call\n"
        "4. whether the final answer is grounded in the tool result\n"
        "5. whether there are hidden assumptions, hallucinated paths, unsupported parameters, or mismatched outputs\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "is_correct": true or false,\n'
        '  "summary": "one short sentence",\n'
        '  "issues": [\n'
        '    {"severity": "critical|major|minor", "focus": "tool_call|tool_result|final_answer|schema", "comment": "specific review comment"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- If the row is fully acceptable, set is_correct=true and issues=[]\n"
        "- If there is any wrong tool choice, wrong argument, wrong result grounding, or wrong final answer, set is_correct=false\n"
        "- Comments must be specific and actionable\n"
        "- Do not add markdown fences or any extra text\n\n"
        f"Row data:\n{json.dumps(review_target, ensure_ascii=False, indent=2)}"
    )


def generate_ollama_json(prompt: str, model: str, base_url: str = DEFAULT_OLLAMA_URL) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }
    response = _post_json(f"{base_url.rstrip('/')}/api/generate", payload)
    text = response.get("response", "")
    parsed = _safe_json(text)
    if not parsed:
        raise ValueError(f"Ollama returned non-JSON verification output: {text[:200]}")
    return text, parsed


def write_results_csv(results: list[VerificationResult], input_path: str | Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"{input_name}_verification_{timestamp}.csv"
    fieldnames = [
        "row_number",
        "tool_name",
        "user_message",
        "is_correct",
        "severity",
        "focus",
        "comment",
        "summary",
        "llm_response",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result.to_rows():
                writer.writerow(row)
    return csv_path


def normalize_verification_result(
    row_number: int,
    user_message: str,
    tool_name: str,
    payload: dict,
    llm_response_text: str,
) -> VerificationResult:
    issues = []
    for item in payload.get("issues", []) or []:
        issues.append(
            VerificationIssue(
                severity=str(item.get("severity", "minor")).lower(),
                focus=str(item.get("focus", "tool_call")).lower(),
                comment=str(item.get("comment", "")).strip(),
            )
        )
    is_correct = bool(payload.get("is_correct", not issues))
    summary = str(payload.get("summary", "")).strip() or ("Row looks correct." if is_correct else "Row has verification issues.")
    return VerificationResult(
        row_number=row_number,
        tool_name=tool_name,
        user_message=user_message,
        is_correct=is_correct,
        summary=summary,
        llm_response=llm_response_text.strip(),
        issues=issues,
    )


def _assistant_message_type(content: str) -> str:
    data = _safe_json(content)
    return str(data.get("type", ""))


def _safe_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {raw[:200]}") from exc


def _get_json(url: str) -> dict:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {raw[:200]}") from exc
