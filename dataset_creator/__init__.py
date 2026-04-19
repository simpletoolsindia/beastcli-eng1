"""
dataset_creator — Agentic CLI Training Dataset Generator
======================================================

A Python module for generating high-quality JSONL training data
for agentic CLI agents with tool-calling capabilities.

Architecture
-----------
This module generates synthetic training samples following the
REQUIREMENTS.md schema (tool_call / tool_result / final_answer),
compatible with Anthropic fine-tuning and Unsloth training pipelines.

Quick Start
-----------
    from dataset_creator import DatasetCreator, Pipeline

    # Create samples
    creator = DatasetCreator(seed=42)
    samples = creator.generate(n=100)

    # Validate
    validator = Validator()
    issues = validator.validate(samples)
    clean = validator.filter(samples, issues)

    # Export
    pipeline = Pipeline(seed=42)
    pipeline.run(n=1000, output="training_data.jsonl")

Design Principles
---------------
1. Realistic tool outputs — simulators mirror real tool behavior
2. Diverse task templates — varied domains, complexity, failure modes
3. Strict schema enforcement — every sample validated before output
4. Quality-first generation — hallucination detection, argument validation
5. Multi-turn conversations — 2-5 tool calls per sample for reasoning depth

Based on research from:
- Anthropic Claude Cookbooks (tool_use/, finetuning/)
- OpenAI fine-tuning function calling patterns
- SWE-agent, Magicoder, OpenCodeInterpreter dataset strategies
"""

from dataset_creator.schemas import (
    ToolCall,
    ToolResult,
    FinalAnswer,
    TrainingSample,
    Message,
    Tool,
    ToolArg,
)
from dataset_creator.tools import TOOL_REGISTRY, TOOL_NAMES, simulate_tool
from dataset_creator.generator import DatasetCreator
from dataset_creator.validator import Validator, ValidationIssue
from dataset_creator.fixer import Fixer
from dataset_creator.quality import QualityScorer
from dataset_creator.pipeline import Pipeline

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "ToolCall",
    "ToolResult",
    "FinalAnswer",
    "TrainingSample",
    "Message",
    "Tool",
    "ToolArg",
    # Registry
    "TOOL_REGISTRY",
    "TOOL_NAMES",
    "simulate_tool",
    # Core
    "DatasetCreator",
    "Validator",
    "ValidationIssue",
    "Fixer",
    "QualityScorer",
    "Pipeline",
]
