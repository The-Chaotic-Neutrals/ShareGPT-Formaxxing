import json
import tempfile
import unittest
from pathlib import Path

from App.ForMaxxer.ForMaxxer import (
    _EMPTY_GEMMA_REASONING,
    _normalize_gemma_reasoning_tags,
    _normalize_gemma_reasoning_value,
    filter_dataset,
)


class GemmaReasoningNormalizationTests(unittest.TestCase):
    def test_adds_empty_thought_to_untagged_gpt_response(self):
        value, outcome = _normalize_gemma_reasoning_value("Final")
        self.assertEqual(outcome, "added")
        self.assertEqual(value, _EMPTY_GEMMA_REASONING + "Final")

    def test_preserves_valid_reasoning(self):
        original = "<|channel>thought\nReasoning\n<channel|>Final"
        self.assertEqual(
            _normalize_gemma_reasoning_value(original),
            (original, "already_tagged"),
        )

    def test_repairs_spaced_tags_and_preamble(self):
        value, outcome = _normalize_gemma_reasoning_value(
            "Preamble. < |channel>thought\nReasoning\n</ |channel>thought\nFinal"
        )
        self.assertEqual(outcome, "repaired")
        self.assertEqual(
            value,
            "<|channel>thought\nReasoning\n<channel|>Preamble. Final",
        )

    def test_ordinary_channel_word_is_not_markup(self):
        value, outcome = _normalize_gemma_reasoning_value("The side-channel leaked.")
        self.assertEqual(outcome, "added")
        self.assertEqual(value, _EMPTY_GEMMA_REASONING + "The side-channel leaked.")

    def test_updates_stats_for_gpt_only(self):
        conversations = [
            {"from": "human", "value": "Question"},
            {"from": "gpt", "value": "Answer"},
        ]
        stats = {}
        _normalize_gemma_reasoning_tags(conversations, stats)
        self.assertEqual(stats["reasoning_empty_tags_added_count"], 1)
        self.assertEqual(conversations[0]["value"], "Question")

    def test_filter_pipeline_normalizes_and_reports_counts(self):
        rows = [
            {"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]},
            {
                "conversations": [
                    {"from": "human", "value": "Q?"},
                    {
                        "from": "gpt",
                        "value": "< |channel>thought\nWhy\n< channel|>A.",
                    },
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary, output_path = filter_dataset(
                source,
                output,
                check_blank_turns=False,
                check_invalid_endings=False,
                check_null_gpt=False,
                check_duplicate_system=False,
                check_duplicate_turns=False,
                strip_think_tags=False,
                normalize_gemma_reasoning=True,
            )
            converted = [
                json.loads(line)
                for line in Path(output_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertIn("Gemma empty thoughts added      : 1", summary)
        self.assertIn("Gemma reasoning tags repaired   : 1", summary)
        self.assertTrue(converted[0]["conversations"][1]["value"].startswith(_EMPTY_GEMMA_REASONING))
        self.assertIn("<|channel>thought\nWhy\n<channel|>", converted[1]["conversations"][1]["value"])


if __name__ == "__main__":
    unittest.main()
