#!/usr/bin/env python3
"""
Generate 10 varied datasets for manual quality review.
Each dataset has different configurations for diversity.
"""
import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from generate_100k import HQDatasetGenerator, HQDatasetGenerator
from merged_dataset_generator import (
    Localization, LocalizationContent, Language, Tone, FormalityLevel,
    ComprehensiveDatasetPipeline, ToolRegistry, ToolSchema, DatasetValidator,
    DatasetExample, Message, DifficultyLevel, SystemPromptGenerator,
    ResponseGenerator, ToolCallGenerator
)

# All available locales in HQDatasetGenerator.LOCALES
ALL_LOCALES = HQDatasetGenerator.LOCALES

# Dataset configurations
DATASET_CONFIGS = [
    {
        "name": "ds1_english_pro",
        "seed": 1001,
        "count": 1000,
        "languages": ["en"],
        "tones": ["professional"],
        "formality": ["formal", "neutral"],
        "description": "English only, professional tone, formal contexts"
    },
    {
        "name": "ds2_multilang_casual",
        "seed": 1002,
        "count": 1000,
        "languages": ["en", "hi", "es", "fr", "de"],
        "tones": ["casual"],
        "formality": ["neutral", "informal"],
        "description": "Multi-language casual tone dataset"
    },
    {
        "name": "ds3_asian_langs",
        "seed": 1003,
        "count": 1000,
        "languages": ["ja", "ko", "zh"],
        "tones": ["professional", "friendly"],
        "formality": ["formal", "neutral"],
        "description": "Asian languages dataset"
    },
    {
        "name": "ds4_tech_friendly",
        "seed": 1004,
        "count": 1000,
        "languages": ["en"],
        "tones": ["technical", "friendly"],
        "formality": ["neutral"],
        "description": "English technical friendly tone"
    },
    {
        "name": "ds5_eu_languages",
        "seed": 1005,
        "count": 1000,
        "languages": ["fr", "de", "it", "es", "pt"],
        "tones": ["professional", "casual"],
        "formality": ["formal", "neutral", "informal"],
        "description": "European languages dataset"
    },
    {
        "name": "ds6_hindi_hinglish",
        "seed": 1006,
        "count": 1000,
        "languages": ["hi"],
        "tones": ["casual", "friendly"],
        "formality": ["informal", "neutral"],
        "description": "Hindi/Hinglish mixed dataset"
    },
    {
        "name": "ds7_easy_simple",
        "seed": 1007,
        "count": 1000,
        "languages": ["en", "es", "fr"],
        "tones": ["friendly", "casual"],
        "formality": ["neutral"],
        "description": "Easy difficulty, simple tasks"
    },
    {
        "name": "ds8_hard_expert",
        "seed": 1008,
        "count": 1000,
        "languages": ["en"],
        "tones": ["technical", "professional"],
        "formality": ["formal"],
        "description": "Hard/Expert difficulty, complex tasks"
    },
    {
        "name": "ds9_all_langs",
        "seed": 1009,
        "count": 1000,
        "languages": ["en", "hi", "es", "fr", "de", "ja", "zh", "ko", "ru", "ar", "pt", "it"],
        "tones": ["professional", "casual", "technical", "friendly"],
        "formality": ["formal", "neutral", "informal"],
        "description": "All 12 languages, all tones"
    },
    {
        "name": "ds10_balanced",
        "seed": 2010,
        "count": 1000,
        "languages": ["en", "hi", "es", "fr", "de", "ja", "zh"],
        "tones": ["professional", "casual"],
        "formality": ["formal", "neutral", "informal"],
        "description": "Balanced mix of popular languages"
    },
]


def make_filtered_generator(config: dict) -> HQDatasetGenerator:
    """Create a generator that respects locale filters from config."""
    gen = HQDatasetGenerator.__new__(HQDatasetGenerator)
    gen.tools = ToolRegistry.get_all_tools()
    gen.validator = DatasetValidator()
    gen._tool_weights = {}

    for t in gen.tools:
        cat = t.category
        if cat == "git":
            gen._tool_weights[t.name] = 0.5
        elif t.name in ("Python_Run", "Git_Push"):
            gen._tool_weights[t.name] = 0.7
        elif t.name in ("File_Copy", "File_Delete", "Bash_Execute", "Web_Fetch"):
            gen._tool_weights[t.name] = 1.8
        elif t.name in ("File_Read", "File_Write", "File_List", "Web_Search"):
            gen._tool_weights[t.name] = 1.2
        else:
            gen._tool_weights[t.name] = 1.0

    gen._used_intents = {}
    gen._example_count = 0

    # Filter locales based on config
    langs = config["languages"]
    tones = config["tones"]
    forms = config["formality"]

    filtered = [
        (lang, tone, form)
        for (lang, tone, form) in ALL_LOCALES
        if lang in langs and tone in tones and form in forms
    ]

    # Fallback: if no exact match, use any locale matching the languages
    if not filtered:
        filtered = [
            (lang, tone, form)
            for (lang, tone, form) in ALL_LOCALES
            if lang in langs
        ]

    # If still empty, use all locales matching any criterion
    if not filtered:
        filtered = ALL_LOCALES

    gen._filtered_locales = filtered
    gen._diff_weights = [
        (DifficultyLevel.EASY, 0.30),
        (DifficultyLevel.MEDIUM, 0.40),
        (DifficultyLevel.HARD, 0.20),
        (DifficultyLevel.EXPERT, 0.10),
    ]

    # Copy methods from HQDatasetGenerator
    gen._pick_tool = HQDatasetGenerator._pick_tool.__get__(gen, HQDatasetGenerator)
    gen._pick_difficulty = HQDatasetGenerator._pick_difficulty.__get__(gen, HQDatasetGenerator)
    gen._pick_locale = lambda: random.choice(gen._filtered_locales)
    gen._get_diverse_intent = HQDatasetGenerator._get_diverse_intent.__get__(gen, HQDatasetGenerator)
    gen._get_diverse_query = HQDatasetGenerator._get_diverse_query.__get__(gen, HQDatasetGenerator)
    gen._build_args = HQDatasetGenerator._build_args.__get__(gen, HQDatasetGenerator)
    gen.generate_one = HQDatasetGenerator.generate_one.__get__(gen, HQDatasetGenerator)
    gen.generate_batch = HQDatasetGenerator.generate_batch.__get__(gen, HQDatasetGenerator)

    print(f"  Filtered locales: {len(gen._filtered_locales)} combinations")
    return gen


def generate_dataset(config: dict, output_base: str) -> dict:
    """Generate a single dataset with given config."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"  Languages: {config['languages']}")
    print(f"  Tones: {config['tones']}")
    print(f"  Count: {config['count']}")
    print(f"  Description: {config['description']}")
    print("=" * 60)

    random.seed(config["seed"])
    gen = make_filtered_generator(config)

    examples = gen.generate_batch(config["count"])
    print(f"Generated {len(examples)} valid examples")

    output_path = Path(output_base) / f"{name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    tool_counts = {}
    lang_counts = {}
    diff_counts = {}
    for ex in examples:
        tn = ex.metadata.get("tool_name", "?")
        tool_counts[tn] = tool_counts.get(tn, 0) + 1
        lang_counts[ex.localization.language] = lang_counts.get(ex.localization.language, 0) + 1
        diff = ex.metadata.get("difficulty", "?")
        diff_counts[diff] = diff_counts.get(diff, 0) + 1

    print(f"  Languages: {dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:5])}")

    stats = {
        "name": name,
        "total": len(examples),
        "output_path": str(output_path),
        "languages": lang_counts,
        "tools": tool_counts,
        "difficulty": diff_counts,
    }
    print(f"Saved to: {output_path}")
    return stats


def main():
    output_base = "output/datasets_for_review"
    Path(output_base).mkdir(exist_ok=True, parents=True)

    all_stats = []
    for i, config in enumerate(DATASET_CONFIGS):
        print(f"\n\n[{i+1}/10] Processing {config['name']}")
        stats = generate_dataset(config, output_base)
        all_stats.append(stats)

    # Summary report
    report_path = Path(output_base) / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Dataset Generation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total datasets: {len(all_stats)}\n\n")
        f.write("## Dataset Summary\n\n")
        for stats in all_stats:
            f.write(f"### {stats['name']}\n")
            f.write(f"- Total: {stats['total']}\n")
            f.write(f"- File: `{stats['output_path']}`\n\n")
            f.write("**Languages:**\n")
            for lang, cnt in sorted(stats['languages'].items(), key=lambda x: -x[1]):
                pct = cnt / stats['total'] * 100
                f.write(f"  - {lang}: {cnt} ({pct:.1f}%)\n")
            f.write("\n**Top Tools:**\n")
            for tool, cnt in sorted(stats['tools'].items(), key=lambda x: -x[1])[:5]:
                pct = cnt / stats['total'] * 100
                f.write(f"  - {tool}: {cnt} ({pct:.1f}%)\n")
            f.write("\n**Difficulty:**\n")
            for diff, cnt in sorted(stats['difficulty'].items()):
                pct = cnt / stats['total'] * 100
                f.write(f"  - {diff}: {cnt} ({pct:.1f}%)\n")
            f.write("\n")

    print(f"\n\n{'='*60}")
    print("ALL 10 DATASETS GENERATED")
    print(f"Report: {report_path}")
    print("=" * 60)
    for stats in all_stats:
        print(f"  - {stats['output_path']}")


if __name__ == "__main__":
    main()
