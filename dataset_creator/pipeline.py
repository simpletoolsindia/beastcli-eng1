"""
pipeline.py — End-to-End Pipeline Orchestrator
===============================================

Orchestrates the full dataset generation pipeline:
generate → validate → fix → filter → score → export

Based on Anthropic's dataset generation best practices:
1. Generate diverse samples with sufficient quantity
2. Validate rigorously to catch schema violations
3. Auto-fix common issues where possible
4. Score quality to identify high-value training samples
5. Export in the target format (JSONL for Unsloth)

Pipeline Stages
---------------
1. Generate: Create n training samples using DatasetCreator
2. Validate: Run all validation rules, collect issues
3. Fix: Attempt auto-repair of fixable issues
4. Validate Again: Re-validate after fixes to confirm
5. Score: Compute quality scores for valid samples
6. Filter: Keep only samples meeting quality threshold
7. Export: Write to JSONL with optional deduplication

Usage:
    pipeline = Pipeline(seed=42)
    result = pipeline.run(n=1000, output="training_data.jsonl")
    print(result["summary"])

    # Or use individual stages:
    samples = pipeline.generate(n=500)
    issues = pipeline.validate(samples)
    valid = pipeline.filter(samples, issues)
    scores = pipeline.score(valid)
    pipeline.export(valid, "output.jsonl")
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dataset_creator.schemas import TrainingSample
from dataset_creator.generator import DatasetCreator
from dataset_creator.validator import Validator, ValidationIssue
from dataset_creator.fixer import Fixer
from dataset_creator.quality import QualityScorer, QualityScore


# --------------------------------------------------------------------------
# Pipeline Result
# --------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Result of a full pipeline run.

    Attributes:
        total_generated: Number of samples generated
        initial_valid: Samples passing validation before fixes
        after_fixes: Samples valid after auto-fix attempt
        final_valid: Samples passing quality threshold
        exported: Number of samples written to file
        output_file: Path to exported file
        generation_stats: Template distribution info
        validation_stats: Validation issue counts
        quality_stats: Aggregate quality scores
        issues: All validation issues found (for debugging)
    """
    total_generated: int
    initial_valid: int
    after_fixes: int
    final_valid: int
    exported: int
    output_file: Optional[str]
    generation_stats: dict
    validation_stats: dict
    quality_stats: dict
    issues: list[ValidationIssue]

    def summary(self) -> dict:
        return {
            "generated": self.total_generated,
            "initial_valid": self.initial_valid,
            "after_fixes": self.after_fixes,
            "final_valid": self.final_valid,
            "exported": self.exported,
            "output_file": self.output_file,
            "pass_rate": f"{self.final_valid / max(self.total_generated, 1) * 100:.1f}%",
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PipelineResult(gen={s['generated']}, valid={s['initial_valid']}, "
            f"fixed={s['after_fixes']}, final={s['final_valid']}, "
            f"exported={s['exported']}, rate={s['pass_rate']})"
        )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class Pipeline:
    """
    End-to-end dataset generation pipeline.

    Usage:
        pipeline = Pipeline(seed=42)
        result = pipeline.run(n=1000, output="training_data.jsonl")

        # Step-by-step:
        samples = pipeline.generate(n=500)
        issues = pipeline.validate(samples)
        fixed = pipeline.fix(samples, issues)
        re_issues = pipeline.validate(fixed)
        valid = pipeline.filter(fixed, re_issues)
        scores = pipeline.score(valid)
        pipeline.export(valid, "output.jsonl")
    """

    def __init__(
        self,
        seed: int = 42,
        min_tool_calls: int = 1,
        max_tool_calls: int = 5,
        domains: Optional[list[str]] = None,
        quality_threshold: float = 40.0,
        strict_validation: bool = True,
    ):
        """
        Initialize pipeline with configuration.

        Args:
            seed: Random seed for reproducible generation
            min_tool_calls: Minimum tool calls per template
            max_tool_calls: Maximum tool calls per template
            domains: Restrict to specific domains, or None for all
            quality_threshold: Minimum quality score to export (0-100)
            strict_validation: If True, warnings also fail validation
        """
        self.seed = seed
        self.quality_threshold = quality_threshold

        self.creator = DatasetCreator(
            seed=seed,
            min_tool_calls=min_tool_calls,
            max_tool_calls=max_tool_calls,
            domains=domains,
        )
        self.validator = Validator(strict=strict_validation)
        self.fixer = Fixer()
        self.scorer = QualityScorer()

        # Track state across stages
        self._samples: list[TrainingSample] = []
        self._issues: list[ValidationIssue] = []
        self._scores: list[QualityScore] = []

    def generate(self, n: int) -> list[TrainingSample]:
        """Stage 1: Generate n training samples."""
        self._samples = self.creator.generate(n=n)
        return self._samples

    def validate(
        self,
        samples: Optional[list[TrainingSample]] = None,
    ) -> list[ValidationIssue]:
        """
        Stage 2: Validate samples and return issues.

        If samples not provided, validates previously generated samples.
        """
        if samples is not None:
            self._samples = samples
        self._issues = self.validator.validate(self._samples)
        return self._issues

    def fix(
        self,
        samples: Optional[list[TrainingSample]] = None,
        issues: Optional[list[ValidationIssue]] = None,
    ) -> list[TrainingSample]:
        """
        Stage 3: Attempt to auto-fix validation issues.

        Returns the fixed samples (original + fixed).
        """
        if samples is not None:
            self._samples = samples
        if issues is not None:
            self._issues = issues

        fixed_samples = self.fixer.fix_and_return(self._samples, self._issues)
        self._samples = fixed_samples
        return fixed_samples

    def filter(
        self,
        samples: Optional[list[TrainingSample]] = None,
        issues: Optional[list[ValidationIssue]] = None,
    ) -> list[TrainingSample]:
        """
        Stage 4: Filter to valid samples only.

        Returns samples passing validation (no error-level issues).
        """
        if samples is not None:
            self._samples = samples
        if issues is not None:
            self._issues = issues

        return self.validator.filter(self._samples, self._issues)

    def score(
        self,
        samples: Optional[list[TrainingSample]] = None,
    ) -> list[QualityScore]:
        """
        Stage 5: Score quality of samples.

        Returns quality scores for each sample.
        """
        if samples is not None:
            self._samples = samples
        self._scores = self.scorer.score_batch(self._samples)
        return self._scores

    def export(
        self,
        samples: Optional[list[TrainingSample]] = None,
        output: str = "training_data.jsonl",
        deduplicate: bool = True,
    ) -> int:
        """
        Stage 6: Export samples to JSONL file.

        Args:
            samples: Samples to export (default: scored valid samples)
            output: Output file path
            deduplicate: If True, remove exact duplicate samples

        Returns:
            Number of samples exported
        """
        if samples is not None:
            self._samples = samples

        # Deduplicate if requested
        exported_samples = self._samples
        if deduplicate:
            seen: set[str] = set()
            unique: list[TrainingSample] = []
            for s in exported_samples:
                sig = s.to_jsonl()
                if sig not in seen:
                    seen.add(sig)
                    unique.append(s)
            exported_samples = unique

        # Write JSONL
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in exported_samples:
                f.write(sample.to_jsonl() + "\n")

        self._exported_count = len(exported_samples)
        return self._exported_count

    def run(
        self,
        n: int = 1000,
        output: str = "training_data.jsonl",
        verbose: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline end-to-end.

        Args:
            n: Number of samples to generate
            output: Output JSONL file path
            verbose: Print progress to stdout

        Returns:
            PipelineResult with statistics
        """
        if verbose:
            print(f"[Pipeline] Starting with seed={self.seed}, n={n}")

        # Stage 1: Generate
        samples = self.generate(n=n)
        gen_stats = self.creator.template_stats()
        if verbose:
            print(f"[Pipeline] Generated {len(samples)} samples "
                  f"({gen_stats['total_templates']} templates)")

        # Stage 2: Validate
        issues = self.validate(samples)
        val_stats = self.validator.stats(issues)
        initial_valid = len(samples) - val_stats.get("affected_samples", 0)
        if verbose:
            print(f"[Pipeline] Validation: {val_stats['errors']} errors, "
                  f"{val_stats['warnings']} warnings, "
                  f"{initial_valid}/{len(samples)} initially valid")

        # Stage 3: Fix
        fixed_samples = self.fix(samples, issues)

        # Stage 4: Re-validate after fixes
        re_issues = self.validate(fixed_samples)
        re_val_stats = self.validator.stats(re_issues)
        after_fixes = len(fixed_samples) - re_val_stats.get("affected_samples", 0)
        if verbose:
            print(f"[Pipeline] After fixes: {re_val_stats['errors']} errors, "
                  f"{after_fixes}/{len(fixed_samples)} valid")

        # Stage 5: Filter to valid
        valid_samples = self.filter(fixed_samples, re_issues)
        if verbose:
            print(f"[Pipeline] Filtered to {len(valid_samples)} valid samples")

        # Stage 6: Score quality
        scores = self.score(valid_samples)
        quality_stats = self.scorer.batch_stats(scores)
        if verbose:
            print(f"[Pipeline] Quality avg: {quality_stats.get('overall_avg', 'N/A')}")

        # Stage 7: Filter by quality threshold
        high_quality = [
            s for s, score in zip(valid_samples, scores)
            if score.overall >= self.quality_threshold
        ]
        if verbose:
            print(f"[Pipeline] Quality threshold ({self.quality_threshold}): "
                  f"{len(high_quality)}/{len(valid_samples)} samples pass")

        # Stage 8: Export
        exported_count = 0
        if high_quality:
            exported_count = self.export(high_quality, output, deduplicate=True)
            if verbose:
                print(f"[Pipeline] Exported {exported_count} samples to {output}")

        return PipelineResult(
            total_generated=n,
            initial_valid=initial_valid,
            after_fixes=after_fixes,
            final_valid=len(valid_samples),
            exported=exported_count,
            output_file=output if exported_count > 0 else None,
            generation_stats=gen_stats,
            validation_stats=re_val_stats,
            quality_stats=quality_stats,
            issues=re_issues,
        )

    # =====================================================================
    # Utility Methods
    # =====================================================================

    def generate_and_preview(
        self,
        n: int = 5,
        show_jsonl: bool = False,
    ) -> list[TrainingSample]:
        """
        Generate n samples and print a preview.

        Useful for quick verification during development.
        """
        samples = self.generate(n=n)
        for i, sample in enumerate(samples):
            print(f"\n{'=' * 60}")
            print(f"  Sample {i} ({sample.step_count()} tool calls)")
            print(f"{'=' * 60}")
            for msg in sample.messages:
                role = msg.role.upper()
                content = msg.content
                if msg.is_json():
                    try:
                        d = json.loads(content)
                        content = json.dumps(d, indent=2)[:300]
                    except Exception:
                        pass
                else:
                    content = content[:200]
                print(f"  [{role}] {content}")
            if show_jsonl:
                print(f"\n  JSONL: {sample.to_jsonl()[:200]}...")
        return samples

    def validate_file(
        self,
        jsonl_path: str,
    ) -> dict:
        """
        Validate samples from a JSONL file.

        Returns validation statistics.
        """
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(TrainingSample.from_jsonl(line))

        issues = self.validate(samples)
        stats = self.validator.stats(issues)
        return {
            "file": jsonl_path,
            "total_samples": len(samples),
            "valid_samples": len(samples) - stats.get("affected_samples", 0),
            "stats": stats,
            "issues": issues,
        }
