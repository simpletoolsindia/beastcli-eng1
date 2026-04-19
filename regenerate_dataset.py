#!/usr/bin/env python3
"""
regenerate_dataset.py — Regenerate Beast CLI datasets with correct schema and high diversity.

Fixes applied:
1. Correct schema: tool_result (type field), tool_call (JSON string content)
2. Multi-tool samples (2-3 tools) for reduced structural duplication
3. All 26 tools covered with diverse query templates
4. 12 languages, 4 tones, 3 formalities
5. 4 difficulty levels with ToolACE distribution
6. 15% error rate (AgentErrorBench optimal)
7. Strict validation before writing

Usage:
    python regenerate_dataset.py --count 2400 --output output/merged_train.jsonl
    python regenerate_dataset.py --count 600 --output output/merged_test.jsonl --seed 999
"""

import json
import random
import argparse
from pathlib import Path

from merged_dataset_generator import (
    ComprehensiveDatasetPipeline,
    Localization,
    DifficultyLevel,
    Language,
    Tone,
    FormalityLevel,
    DatasetValidator,
    ToolRegistry,
)


def generate_diverse_dataset(
    count: int,
    output_path: str,
    seed: int = 42,
    validate: bool = True,
) -> dict:
    """Generate a high-quality diverse dataset."""
    random.seed(seed)
    pipe = ComprehensiveDatasetPipeline(seed=seed)

    languages = [l.value for l in Language]  # 12 languages
    tones = [t.value for t in Tone]          # 4 tones
    formalities = [f.value for f in FormalityLevel]  # 3 formalities

    # Difficulty distribution (ToolACE research)
    difficulty_dist = {
        DifficultyLevel.EASY: 0.30,
        DifficultyLevel.MEDIUM: 0.40,
        DifficultyLevel.HARD: 0.20,
        DifficultyLevel.EXPERT: 0.10,
    }

    # Track diversity
    tool_counts = {t.name: 0 for t in ToolRegistry.get_all_tools()}
    lang_counts = {l: 0 for l in languages}
    diff_counts = {d.value: 0 for d in DifficultyLevel}

    valid_examples = []
    invalid_count = 0

    # Distribute across locales: cycle through all combinations
    locale_options = []
    for lang in languages:
        for tone in tones:
            for form in formalities:
                locale_options.append((lang, tone, form))

    # Also create some locale-focused batches for better coverage
    locale_batches = []

    # Full locale grid: 12*4*3 = 144 combinations, cycle through them
    for i in range(count):
        lang, tone, form = locale_options[i % len(locale_options)]
        loc = Localization(
            language=lang,
            tone=tone,
            formality=form,
            humanize=True,
            humanize_level="medium",
        )

        # Pick difficulty
        difficulties = list(difficulty_dist.keys())
        weights = list(difficulty_dist.values())
        difficulty = random.choices(difficulties, weights=weights)[0]

        # Multi-tool for MEDIUM/HARD/EXPERT
        multi_tool = difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD, DifficultyLevel.EXPERT)

        # 15% error rate
        include_error = random.random() < 0.15

        try:
            example = pipe.generate_single(
                localization=loc,
                difficulty=difficulty,
                include_error=include_error,
                multi_tool=multi_tool,
            )

            if validate:
                valid, errors = DatasetValidator.validate_example(example)
                if not valid:
                    invalid_count += 1
                    if invalid_count <= 3:
                        print(f"  Rejected sample {i}: {errors[:2]}")
                    continue

            valid_examples.append(example)

            # Track diversity
            meta = example.metadata
            tool_counts[meta['tool_name']] = tool_counts.get(meta['tool_name'], 0) + 1
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            diff_counts[difficulty.value] = diff_counts.get(difficulty.value, 0) + 1

            if (i + 1) % 200 == 0:
                print(f"  Generated {i+1}/{count} (valid: {len(valid_examples)})...")

        except Exception as e:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"  Error on sample {i}: {e}")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            written += 1

    return {
        "written": written,
        "invalid_rejected": invalid_count,
        "tool_coverage": tool_counts,
        "lang_coverage": lang_counts,
        "diff_coverage": diff_counts,
    }


def main():
    parser = argparse.ArgumentParser(description='Regenerate Beast CLI dataset')
    parser.add_argument('--count', type=int, default=2400, help='Number of samples')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-validate', dest='validate', action='store_false', default=True)
    args = parser.parse_args()

    print(f"Generating {args.count} samples...")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Validate: {args.validate}")
    print()

    stats = generate_diverse_dataset(
        count=args.count,
        output_path=args.output,
        seed=args.seed,
        validate=args.validate,
    )

    print(f"\n=== Results ===")
    print(f"Written: {stats['written']}/{args.count}")
    print(f"Rejected (invalid): {stats['invalid_rejected']}")
    print(f"Tool coverage: {sum(1 for v in stats['tool_coverage'].values() if v > 0)}/{len(stats['tool_coverage'])} tools used")
    print(f"Language coverage: {sum(1 for v in stats['lang_coverage'].values() if v > 0)}/{len(stats['lang_coverage'])} languages")
    print(f"Difficulty: {dict(sorted(stats['diff_coverage'].items()))}")


if __name__ == '__main__':
    main()