"""
quality.py — Quality Scoring Engine
====================================

Scores training samples on multiple quality dimensions:
diversity, complexity, hallucination risk, and overall quality.

Based on research from:
- Magicoder (ICLR 2024): Diverse instructions > curated datasets
- OpenCodeInterpreter: Execution feedback reduces hallucinations
- Anthropic's JSON Schema enforcement: Format consistency improves training

Quality Dimensions
------------------
1. Diversity Score (0-100): How varied are the tools and domains used?
2. Complexity Score (0-100): How many tool calls? How varied are the tools?
3. Hallucination Score (0-100): Is the final answer grounded in tool results?
4. Overall Score (0-100): Weighted combination of all dimensions

Usage:
    scorer = QualityScorer()
    for sample in samples:
        score = scorer.score(sample)
        print(f"Sample {i}: {score.overall:.1f}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from dataset_creator.schemas import TrainingSample
from dataset_creator.tools import TOOL_NAMES


# --------------------------------------------------------------------------
# Quality Score
# --------------------------------------------------------------------------

@dataclass
class QualityScore:
    """
    Quality assessment for a single training sample.

    Attributes:
        diversity: Tool/domain variety (0-100)
        complexity: Reasoning depth and tool chain length (0-100)
        hallucination_risk: Groundedness in tool results (0-100, higher=better)
        overall: Weighted combination (0-100)
        details: Breakdown of scoring factors
    """
    diversity: float
    complexity: float
    hallucination_risk: float
    overall: float
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict:
        return {
            "diversity": round(self.diversity, 2),
            "complexity": round(self.complexity, 2),
            "hallucination_risk": round(self.hallucination_risk, 2),
            "overall": round(self.overall, 2),
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"QualityScore(d={self.diversity:.1f}, c={self.complexity:.1f}, "
            f"h={self.hallucination_risk:.1f}, overall={self.overall:.1f})"
        )


# --------------------------------------------------------------------------
# Quality Scorer
# --------------------------------------------------------------------------

class QualityScorer:
    """
    Scores training samples on multiple quality dimensions.

    Design principles:
    1. Scores are reproducible (seed-independent)
    2. Each dimension is independently computable
    3. Overall score weights dimensions by training importance
    4. Details dict provides interpretability for debugging
    """

    def __init__(
        self,
        diversity_weight: float = 0.3,
        complexity_weight: float = 0.3,
        hallucination_weight: float = 0.4,
    ):
        """
        Initialize scorer with dimension weights.

        Default weights favor hallucination resistance (40%) and
        complexity (30%) over diversity (30%), based on research
        showing that grounded reasoning and depth are more important
        than pure diversity for agentic tool-calling tasks.
        """
        self.diversity_weight = diversity_weight
        self.complexity_weight = complexity_weight
        self.hallucination_weight = hallucination_weight
        total = diversity_weight + complexity_weight + hallucination_weight
        # Normalize to sum to 1.0
        self.diversity_weight /= total
        self.complexity_weight /= total
        self.hallucination_weight /= total

    def score(self, sample: TrainingSample) -> QualityScore:
        """
        Compute quality score for a sample.

        Returns:
            QualityScore with individual dimensions and overall
        """
        tool_calls = sample.tool_calls()
        tool_results = sample.tool_results()
        final_answer = sample.final_answer()

        diversity = self._score_diversity(tool_calls)
        complexity = self._score_complexity(tool_calls, tool_results)
        hallucination = self._score_hallucination_risk(
            tool_calls, tool_results, final_answer
        )

        overall = (
            diversity * self.diversity_weight
            + complexity * self.complexity_weight
            + hallucination * self.hallucination_weight
        )

        return QualityScore(
            diversity=diversity,
            complexity=complexity,
            hallucination_risk=hallucination,
            overall=overall,
            details={
                "tool_count": len(tool_calls),
                "unique_tools": len({tc.tool_name for tc in tool_calls}),
                "has_final_answer": final_answer is not None,
            },
        )

    def _score_diversity(self, tool_calls: list) -> float:
        """
        Score: How diverse are the tools used?

        Factors:
        - Number of unique tools (more unique = higher score)
        - Tool category spread (filesystem, git, execution, web, utility)
        - Using less-common tools adds bonus

        Max score: 100 (uses 5+ unique tools across different categories)
        """
        if not tool_calls:
            return 0.0

        unique_tools = {tc.tool_name for tc in tool_calls}
        num_unique = len(unique_tools)

        # Base: unique tool count (max 60 points)
        base = min(num_unique * 20, 60)

        # Bonus for using tools from different categories
        categories = set()
        category_map = {
            "read_file": "filesystem", "write_file": "filesystem",
            "update_file": "filesystem", "delete_file": "filesystem",
            "list_files": "filesystem", "search_files": "filesystem",
            "git_status": "git", "git_log": "git", "git_commit": "git",
            "git_pull": "git", "git_push": "git", "git_stash": "git",
            "bash": "execution", "python_exec": "execution",
            "node_exec": "execution",
            "web_search": "web", "fetch_url": "web",
            "get_timestamp": "utility", "env_get": "utility",
        }
        for tc in tool_calls:
            cat = category_map.get(tc.tool_name, "other")
            categories.add(cat)

        # Category diversity bonus (max 40 points)
        cat_bonus = len(categories) * 10

        return min(base + cat_bonus, 100)

    def _score_complexity(self, tool_calls: list, tool_results: list) -> float:
        """
        Score: How complex is the reasoning chain?

        Factors:
        - Number of tool calls (more = higher, up to a point)
        - Tool results with actual content (not empty)
        - Mix of different tool types (not just repeated reads)

        Max score: 100
        Ideal: 3-5 diverse tool calls with meaningful results
        """
        if not tool_calls:
            return 0.0

        count = len(tool_calls)

        # Tool call count score (max 50 points)
        # Sweet spot is 2-5 calls
        if count == 1:
            count_score = 20
        elif count == 2:
            count_score = 40
        elif 3 <= count <= 4:
            count_score = 50
        elif count == 5:
            count_score = 45
        else:
            count_score = max(0, 50 - (count - 5) * 5)

        # Result content score (max 30 points)
        result_score = 0.0
        if tool_results:
            non_empty = sum(1 for r in tool_results if r.output and r.output.strip())
            ratio = non_empty / len(tool_results)
            result_score = ratio * 30

        # Reasoning chain score (max 20 points)
        # Check that tool calls aren't all the same type
        tool_names = [tc.tool_name for tc in tool_calls]
        unique_count = len(set(tool_names))
        chain_score = min(unique_count * 5, 20)

        return min(count_score + result_score + chain_score, 100)

    def _score_hallucination_risk(
        self,
        tool_calls: list,
        tool_results: list,
        final_answer: Optional,
    ) -> float:
        """
        Score: Is the final answer grounded in tool results?

        Hallucination red flags:
        - Final answer is generic (doesn't reference tool results)
        - Final answer mentions files/values not in any tool output
        - Tool results are empty but final answer claims success
        - Final answer contradicts tool results

        Max score: 100 (higher = less hallucination risk)
        """
        if not final_answer or not final_answer.content:
            return 0.0

        fa_content = final_answer.content.lower()
        fa_len = len(final_answer.content)

        # Empty or very short answer
        if fa_len < 10:
            return 10.0

        score = 50.0  # Base score

        # Check 1: Does final answer reference tool outputs?
        has_tool_reference = False
        if tool_results:
            for r in tool_results:
                output_preview = r.output[:200].lower() if r.output else ""
                # Look for overlap between FA content and results
                fa_words = set(fa_content.split())
                result_words = set(output_preview.split())
                overlap = len(fa_words & result_words)
                if overlap >= 2:
                    has_tool_reference = True
                    break

        if has_tool_reference:
            score += 25

        # Check 2: Is the final answer specific rather than generic?
        generic_patterns = [
            "task completed", "done", "finished", "successfully",
            "all done", "completed successfully", "no issues",
        ]
        is_generic = any(fa_content.startswith(p) for p in generic_patterns)
        if not is_generic:
            score += 15

        # Check 3: Reasonable length (not too short, not too long)
        if 30 <= fa_len <= 300:
            score += 10

        # Check 4: Tool results are not all errors
        if tool_results:
            error_count = sum(1 for r in tool_results if r.exit_code != 0)
            error_ratio = error_count / len(tool_results)
            if error_ratio < 0.5:
                score += 10  # Most results were successful

        return min(score, 100)

    def score_batch(
        self,
        samples: list[TrainingSample],
    ) -> list[QualityScore]:
        """Score multiple samples."""
        return [self.score(s) for s in samples]

    def filter_by_score(
        self,
        samples: list[TrainingSample],
        scores: list[QualityScore],
        min_overall: float = 50.0,
    ) -> list[TrainingSample]:
        """
        Filter samples by minimum quality threshold.

        Args:
            samples: Original samples
            scores: Precomputed scores (from score_batch)
            min_overall: Minimum overall quality score (0-100)

        Returns:
            Samples with overall >= min_overall
        """
        return [s for s, score in zip(samples, scores) if score.overall >= min_overall]

    def batch_stats(self, scores: list[QualityScore]) -> dict:
        """Compute aggregate statistics for a batch of scores."""
        if not scores:
            return {}

        def avg(key):
            return round(sum(getattr(s, key) for s in scores) / len(scores), 2)

        overalls = [s.overall for s in scores]
        return {
            "count": len(scores),
            "diversity_avg": avg("diversity"),
            "complexity_avg": avg("complexity"),
            "hallucination_risk_avg": avg("hallucination_risk"),
            "overall_avg": avg("overall"),
            "overall_min": round(min(overalls), 2),
            "overall_max": round(max(overalls), 2),
            "overall_std": round(
                (sum((o - avg("overall")) ** 2 for o in overalls) / len(scores)) ** 0.5, 2
            ),
        }
