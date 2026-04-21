#!/usr/bin/env python3
"""
Generate 10 varied datasets for manual quality review.
Each dataset has different configurations for diversity.
"""
import json
import random
import uuid
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from generate_100k import HQDatasetGenerator, LOCALES as ALL_LOCALES, EXPANDED_INTENTS, DIFFICULTIES
from merged_dataset_generator import (
    Localization, LocalizationContent, DifficultyLevel,
    ToolRegistry, ToolSchema, DatasetValidator,
    DatasetExample, Message,
    SystemPromptGenerator, ResponseGenerator, FinalAnswerGenerator,
)


class FilteredGenerator:
    """HQDatasetGenerator subclass with filtered locale choices."""

    def __init__(self, config: dict):
        self._cfg = config
        self.tools = ToolRegistry.get_all_tools()
        self.validator = DatasetValidator()
        self._used_intents: dict[str, set[str]] = {}
        self._example_count = 0

        langs = config["languages"]
        tones = config["tones"]
        forms = config["formality"]

        self._filtered_locales = [
            (lang, tone, form)
            for (lang, tone, form) in ALL_LOCALES
            if lang in langs and tone in tones and form in forms
        ]
        if not self._filtered_locales:
            self._filtered_locales = [l for l in ALL_LOCALES if l[0] in langs]
        if not self._filtered_locales:
            self._filtered_locales = ALL_LOCALES

        print(f"  Filtered locales: {len(self._filtered_locales)} combinations")

    def _pick_tool(self) -> ToolSchema:
        weights = [1.2, 1.5, 1.0, 1.2, 1.0, 1.0, 0.4, 0.4, 0.8, 0.6]
        return random.choices(self.tools, weights=weights, k=1)[0]

    def _pick_difficulty(self) -> DifficultyLevel:
        r = random.random()
        cum = 0.0
        for diff, prob in DIFFICULTIES:
            cum += prob
            if r < cum:
                return diff
        return DifficultyLevel.MEDIUM

    def _pick_locale(self) -> tuple:
        return random.choice(self._filtered_locales)

    def _get_diverse_intent(self, tool_name: str) -> tuple[str, dict]:
        intents = EXPANDED_INTENTS.get(tool_name, [])
        if not intents:
            return (f"use {tool_name}", {})
        used = self._used_intents.get(tool_name, set())
        unused = [i for i in intents if i["q"] not in used]
        if unused and random.random() < 0.8:
            intent = random.choice(unused)
            used.add(intent["q"])
            self._used_intents[tool_name] = used
        else:
            intent = random.choice(intents)
        return intent["q"], intent["a"]

    def generate_one(self, include_error: bool = False) -> DatasetExample:
        tool = self._pick_tool()
        difficulty = self._pick_difficulty()
        lang, tone, formality = self._pick_locale()

        loc = Localization(
            language=lang, tone=tone, formality=formality,
            humanize=True, humanize_level="medium",
        )

        query_hint, intent_args = self._get_diverse_intent(tool.name)

        system_prompt = SystemPromptGenerator.generate(loc)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query_hint),
        ]

        success = not include_error or random.random() > 0.15

        tc_id = "call_%s" % uuid.uuid4().hex[:12]
        tool_call_content = json.dumps({
            "type": "tool_call",
            "id": tc_id,
            "tool_name": tool.name,
            "arguments": intent_args,
        }, ensure_ascii=False)

        tool_response = ResponseGenerator.generate(tool, intent_args, success)
        tool_result_content = tool_response.replace("{{TOOL_CALL_ID}}", tc_id)

        messages.append(Message(role="assistant", content=tool_call_content))
        messages.append(Message(role="tool", content=tool_result_content, tool_call_id=tc_id, name=tool.name))

        final = FinalAnswerGenerator.generate(tool, intent_args, success, loc.language)
        messages.append(Message(role="assistant", content=final))

        self._example_count += 1
        return DatasetExample(
            messages=messages,
            localization=loc,
            tools=[t.to_openai_format() for t in self.tools],
            metadata={
                "difficulty": difficulty.value,
                "tool_category": tool.category,
                "tool_name": tool.name,
                "tools_used": [tool.name],
                "num_tools": 1,
                "success": success,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_version": "v5-simplified",
            }
        )

    def generate_batch(self, count: int, include_error: bool = False) -> list[DatasetExample]:
        examples = []
        attempts = 0
        max_attempts = count * 10
        while len(examples) < count and attempts < max_attempts:
            attempts += 1
            ex = self.generate_one(include_error=include_error)
            ok, _ = self.validator.validate_example(ex)
            if ok:
                examples.append(ex)
        return examples


DATASET_CONFIGS = [
    {"name": "ds1_english_pro",      "seed": 1001, "count": 1000, "languages": ["en"],                      "tones": ["professional"],                "formality": ["formal", "neutral"],       "description": "English only, professional tone"},
    {"name": "ds2_multilang_casual",   "seed": 1002, "count": 1000, "languages": ["en", "hi", "es", "fr", "de"], "tones": ["casual"],                       "formality": ["neutral", "informal"],    "description": "Multi-language casual tone"},
    {"name": "ds3_asian_langs",         "seed": 1003, "count": 1000, "languages": ["ja", "ko", "zh"],              "tones": ["professional", "friendly"],    "formality": ["formal", "neutral"],      "description": "Asian languages"},
    {"name": "ds4_tech_friendly",       "seed": 1004, "count": 1000, "languages": ["en"],                      "tones": ["technical", "friendly"],       "formality": ["neutral"],                "description": "English technical friendly"},
    {"name": "ds5_eu_languages",        "seed": 1005, "count": 1000, "languages": ["fr", "de", "it", "es", "pt"], "tones": ["professional", "casual"],    "formality": ["formal", "neutral", "informal"], "description": "European languages"},
    {"name": "ds6_hindi_hinglish",      "seed": 1006, "count": 1000, "languages": ["hi"],                      "tones": ["casual", "friendly"],          "formality": ["informal", "neutral"],    "description": "Hindi/Hinglish"},
    {"name": "ds7_easy_simple",         "seed": 1007, "count": 1000, "languages": ["en", "es", "fr"],        "tones": ["friendly", "casual"],        "formality": ["neutral"],                "description": "Easy difficulty"},
    {"name": "ds8_hard_expert",         "seed": 1008, "count": 1000, "languages": ["en"],                      "tones": ["technical", "professional"],   "formality": ["formal"],                 "description": "Hard/Expert difficulty"},
    {"name": "ds9_all_langs",           "seed": 1009, "count": 1000, "languages": ["en", "hi", "es", "fr", "de", "ja", "zh", "ko", "ru", "ar", "pt", "it"], "tones": ["professional", "casual", "technical", "friendly"], "formality": ["formal", "neutral", "informal"], "description": "All 12 languages"},
    {"name": "ds10_balanced",           "seed": 2010, "count": 1000, "languages": ["en", "hi", "es", "fr", "de", "ja", "zh"], "tones": ["professional", "casual"],   "formality": ["formal", "neutral", "informal"], "description": "Balanced mix"},
]


def generate_dataset(config: dict, output_base: str) -> dict:
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"  Languages: {config['languages']}")
    print(f"  Tones: {config['tones']}")
    print(f"  Count: {config['count']}")
    print("=" * 60)

    random.seed(config["seed"])
    gen = FilteredGenerator(config)

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

    return {
        "name": name, "total": len(examples), "output_path": str(output_path),
        "languages": lang_counts, "tools": tool_counts, "difficulty": diff_counts,
    }


def main():
    output_base = "output/datasets_for_review"
    Path(output_base).mkdir(exist_ok=True, parents=True)

    all_stats = []
    for i, config in enumerate(DATASET_CONFIGS):
        print(f"\n[{i+1}/10] {config['name']}")
        stats = generate_dataset(config, output_base)
        all_stats.append(stats)

    report_path = Path(output_base) / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Dataset Generation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total datasets: {len(all_stats)}\n\n")
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

    print(f"\n{'='*60}")
    print("ALL 10 DATASETS GENERATED")
    print(f"Report: {report_path}")
    print("=" * 60)
    for s in all_stats:
        print(f"  - {s['output_path']}")


if __name__ == "__main__":
    main()
