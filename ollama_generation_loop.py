#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from dataset_verifier import (
    DEFAULT_OLLAMA_URL,
    VerificationResult,
    load_jsonl,
    verify_sample,
    write_results_csv,
)
from regenerate_dataset import generate_diverse_dataset


@dataclass
class RoundSummary:
    round_number: int
    seed: int
    requested_candidates: int
    generated_candidates: int
    accepted_in_round: int
    rejected_in_round: int
    accepted_total: int


def write_jsonl(samples: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return path


def write_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_ollama_generation_loop(
    target_count: int,
    output_path: str | Path,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    seed: int = 42,
    max_rounds: int = 8,
    candidate_multiplier: float = 1.5,
    min_batch_size: int = 25,
    keep_candidates: bool = True,
    round_callback: Optional[Callable[[dict], None]] = None,
    review_progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    output_path = Path(output_path)
    loop_dir = output_path.parent / f"{output_path.stem}_ollama_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)

    accepted_samples: list[dict] = []
    accepted_signatures: set[str] = set()
    rejected_results: list[VerificationResult] = []
    round_summaries: list[RoundSummary] = []
    global_row_number = 0

    for round_number in range(1, max_rounds + 1):
        if len(accepted_samples) >= target_count:
            break

        remaining = target_count - len(accepted_samples)
        requested_candidates = max(min_batch_size, math.ceil(remaining * candidate_multiplier))
        round_seed = seed + round_number - 1
        candidate_path = loop_dir / f"round_{round_number:02d}_candidates.jsonl"

        print(f"\n=== Ollama Loop Round {round_number}/{max_rounds} ===")
        print(f"Seed: {round_seed}")
        print(f"Need: {remaining} more accepted rows")
        print(f"Generating candidate batch: {requested_candidates}")
        if round_callback:
            round_callback({
                "event": "round_start",
                "round_number": round_number,
                "max_rounds": max_rounds,
                "seed": round_seed,
                "remaining": remaining,
                "requested_candidates": requested_candidates,
                "accepted_total": len(accepted_samples),
                "target_count": target_count,
            })

        generate_diverse_dataset(
            count=requested_candidates,
            output_path=str(candidate_path),
            seed=round_seed,
            validate=True,
        )
        samples = load_jsonl(candidate_path)
        if round_callback:
            round_callback({
                "event": "generation_complete",
                "round_number": round_number,
                "generated_candidates": len(samples),
                "accepted_total": len(accepted_samples),
                "target_count": target_count,
            })

        accepted_in_round = 0
        rejected_in_round = 0
        review_total = len(samples)
        for review_index, sample in enumerate(samples, 1):
            if len(accepted_samples) >= target_count:
                break

            global_row_number += 1
            verification = verify_sample(
                row_number=global_row_number,
                sample=sample,
                model=model,
                base_url=base_url,
            )

            signature = json.dumps(sample, ensure_ascii=False, sort_keys=True)
            if verification.is_correct and signature not in accepted_signatures:
                accepted_signatures.add(signature)
                accepted_samples.append(sample)
                accepted_in_round += 1
            else:
                rejected_results.append(verification)
                rejected_in_round += 1

            if review_progress_callback:
                review_progress_callback({
                    "event": "review_progress",
                    "round_number": round_number,
                    "review_index": review_index,
                    "review_total": review_total,
                    "accepted_in_round": accepted_in_round,
                    "rejected_in_round": rejected_in_round,
                    "accepted_total": len(accepted_samples),
                    "target_count": target_count,
                    "is_correct": verification.is_correct,
                    "tool_name": verification.tool_name,
                    "row_number": global_row_number,
                    "summary": verification.summary,
                })

        round_summary = RoundSummary(
            round_number=round_number,
            seed=round_seed,
            requested_candidates=requested_candidates,
            generated_candidates=len(samples),
            accepted_in_round=accepted_in_round,
            rejected_in_round=rejected_in_round,
            accepted_total=len(accepted_samples),
        )
        round_summaries.append(round_summary)

        print(
            f"Round {round_number} summary: accepted {accepted_in_round}, "
            f"rejected {rejected_in_round}, total accepted {len(accepted_samples)}/{target_count}"
        )
        if round_callback:
            round_callback({
                "event": "round_complete",
                "summary": asdict(round_summary),
                "accepted_total": len(accepted_samples),
                "target_count": target_count,
            })

        if not keep_candidates and candidate_path.exists():
            candidate_path.unlink()

    final_samples = accepted_samples[:target_count]
    write_jsonl(final_samples, output_path)
    rejected_csv_path = write_results_csv(rejected_results, output_path)

    report = {
        "target_count": target_count,
        "accepted_count": len(final_samples),
        "rejected_count": len(rejected_results),
        "model": model,
        "base_url": base_url,
        "seed": seed,
        "max_rounds": max_rounds,
        "candidate_multiplier": candidate_multiplier,
        "min_batch_size": min_batch_size,
        "output_path": str(output_path),
        "rejected_csv_path": str(rejected_csv_path),
        "success": len(final_samples) >= target_count,
        "rounds": [asdict(summary) for summary in round_summaries],
    }
    report_path = write_report(loop_dir / "loop_report.json", report)

    print("\n=== Ollama Loop Result ===")
    print(f"Accepted rows: {len(final_samples)}/{target_count}")
    print(f"Rejected rows logged: {len(rejected_results)}")
    print(f"Output dataset: {output_path}")
    print(f"Rejected CSV: {rejected_csv_path}")
    print(f"Loop report: {report_path}")
    if round_callback:
        round_callback({
            "event": "loop_complete",
            "report": report,
            "report_path": str(report_path),
        })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dataset, let an Ollama model review each row, and regenerate until the target count is reached."
    )
    parser.add_argument("--count", type=int, required=True, help="Target number of accepted rows")
    parser.add_argument("--output", type=str, required=True, help="Final accepted JSONL output path")
    parser.add_argument("--model", type=str, default="gemma4:latest", help="Ollama model name")
    parser.add_argument("--base-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument("--max-rounds", type=int, default=8, help="Maximum generate-review rounds")
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.5,
        help="How many candidates to generate relative to remaining target rows",
    )
    parser.add_argument("--min-batch-size", type=int, default=25, help="Minimum candidates per round")
    parser.add_argument(
        "--discard-candidates",
        action="store_true",
        help="Delete intermediate candidate JSONL files after each round",
    )
    args = parser.parse_args()

    report = run_ollama_generation_loop(
        target_count=args.count,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed,
        max_rounds=args.max_rounds,
        candidate_multiplier=args.candidate_multiplier,
        min_batch_size=args.min_batch_size,
        keep_candidates=not args.discard_candidates,
    )

    if not report["success"]:
        raise SystemExit(
            f"Loop stopped after {args.max_rounds} rounds with only {report['accepted_count']} accepted rows."
        )


if __name__ == "__main__":
    main()
