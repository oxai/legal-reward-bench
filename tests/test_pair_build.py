from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_STAGE = ROOT / "pipeline" / "04_pairs"
sys.path.insert(0, str(PAIR_STAGE))

_BUILD_SPEC = importlib.util.spec_from_file_location("pair_build", PAIR_STAGE / "build.py")
assert _BUILD_SPEC is not None and _BUILD_SPEC.loader is not None
pair_build = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(pair_build)


def label(response_id: str, labels: dict[str, str]) -> dict:
    return {
        "id": f"{response_id}__label",
        "response_id": response_id,
        "triple_id": response_id.rsplit("__", 1)[0],
        "labels": labels,
    }


def response(response_id: str, text: str) -> dict:
    return {
        "id": response_id,
        "triple_id": response_id.rsplit("__", 1)[0],
        "response": text,
    }


class PairBuildTests(unittest.TestCase):
    def test_hierarchy_materializes_answerable_and_unanswerable_pairs(self) -> None:
        triples = {
            "q1__gold_plus_bm25_distractors": {
                "id": "q1__gold_plus_bm25_distractors",
                "question": "Question 1?",
                "context": "Context 1",
                "metadata": {
                    "base_triple_id": "q1",
                    "context_variant": "gold_plus_bm25_distractors",
                    "answerability": "answerable",
                },
            },
            "q2__bm25_hard_negative": {
                "id": "q2__bm25_hard_negative",
                "question": "Question 2?",
                "context": "Context 2",
                "metadata": {
                    "base_triple_id": "q2",
                    "context_variant": "bm25_hard_negative",
                    "answerability": "unanswerable",
                },
            },
        }
        responses = {
            "q1__gold_plus_bm25_distractors__a": response(
                "q1__gold_plus_bm25_distractors__a", "supported answer"
            ),
            "q1__gold_plus_bm25_distractors__b": response(
                "q1__gold_plus_bm25_distractors__b", "wrong answer"
            ),
            "q1__gold_plus_bm25_distractors__c": response(
                "q1__gold_plus_bm25_distractors__c", "refusal"
            ),
            "q2__bm25_hard_negative__a": response("q2__bm25_hard_negative__a", "refusal"),
            "q2__bm25_hard_negative__b": response("q2__bm25_hard_negative__b", "attempt"),
        }
        labels = [
            label(
                "q1__gold_plus_bm25_distractors__a",
                {
                    "answer_behavior": "attempted",
                    "faithfulness": "fully_supported",
                    "correctness": "correct",
                    "completeness": "complete",
                },
            ),
            label(
                "q1__gold_plus_bm25_distractors__b",
                {
                    "answer_behavior": "attempted",
                    "faithfulness": "fully_supported",
                    "correctness": "incorrect",
                    "completeness": "complete",
                },
            ),
            label(
                "q1__gold_plus_bm25_distractors__c",
                {
                    "answer_behavior": "abstained",
                    "faithfulness": "not_applicable",
                    "correctness": "not_applicable",
                    "completeness": "not_applicable",
                },
            ),
            label(
                "q2__bm25_hard_negative__a",
                {
                    "answer_behavior": "abstained",
                    "faithfulness": "not_applicable",
                    "correctness": "not_applicable",
                    "completeness": "not_applicable",
                },
            ),
            label(
                "q2__bm25_hard_negative__b",
                {
                    "answer_behavior": "attempted",
                    "faithfulness": "fully_supported",
                    "correctness": "correct",
                    "completeness": "complete",
                },
            ),
        ]

        pairs = pair_build.build_pairs(
            triples=triples,
            labels=labels,
            responses=responses,
            prompt_template="{context}\nQuestion: {question}\nAnswer:",
        )

        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            [pair["metadata"]["decisive_dimension"] for pair in pairs],
            ["correctness", "answer_behavior", "answer_behavior", "answer_behavior"],
        )
        self.assertEqual(pairs[0]["chosen"], "supported answer")
        self.assertEqual(pairs[0]["rejected"], "wrong answer")
        self.assertEqual(pairs[-1]["chosen"], "refusal")
        self.assertEqual(pairs[-1]["rejected"], "attempt")
        self.assertIn("Question 1?", pairs[0]["prompt"])
        self.assertEqual(
            [pair["split"] for pair in pairs],
            ["correctness", "refusal_answerable", "refusal_answerable", "refusal_unanswerable"],
        )


if __name__ == "__main__":
    unittest.main()
