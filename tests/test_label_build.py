from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "03_labels"))

from build import LabelConfig, LabelResources, ResponseLabeler  # noqa: E402
from judge import JudgeConfig, JudgeResult, JsonJudge  # noqa: E402


def make_triple() -> dict[str, Any]:
    return {
        "id": "t1",
        "question": "Question?",
        "context": "Candidate context",
        "answer": "Authoritative answer",
        "metadata": {
            "answerability": "answerable",
            "context_variant": "base",
        },
    }


def make_response() -> dict[str, Any]:
    return {
        "id": "r1",
        "triple_id": "t1",
        "response": "Candidate answer",
    }


def make_config() -> LabelConfig:
    return LabelConfig(
        judge_model="judge-model",
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=30,
        retries=1,
        concurrency=1,
    )


def make_resources() -> LabelResources:
    return LabelResources(
        behavior_prompt_template="behavior {question} {candidate_response}",
        behavior_schema={"title": "behavior"},
        prompt_templates={
            "faithfulness": "faith {question} {candidate_context} {candidate_response}",
            "correctness": "correct {question} {reference_answer} {candidate_response}",
            "completeness": "complete {question} {reference_answer} {candidate_response}",
        },
        schemas={
            "faithfulness": {"title": "faithfulness"},
            "correctness": {"title": "correctness"},
            "completeness": {"title": "completeness"},
        },
    )


class FakeJudge:
    def __init__(self, results: list[JudgeResult]) -> None:
        self.results = results
        self.prompts: list[str] = []

    async def generate(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        validate: Callable[[dict[str, Any]], list[str]],
    ) -> JudgeResult:
        del schema
        self.prompts.append(prompt)
        result = self.results.pop(0)
        errors = validate(result.payload)
        if errors:
            raise AssertionError(errors)
        return result


class LabelBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_judge_retries_same_model_without_fallback(self) -> None:
        judge = JsonJudge(
            JudgeConfig(
                model="judge-model",
                temperature=0.0,
                max_tokens=256,
                timeout_seconds=30,
                retries=1,
            )
        )
        generate = AsyncMock(side_effect=[RuntimeError("temporary"), {"answer_behavior": "attempted"}])

        with patch("judge.generate_json", generate):
            result = await judge.generate(
                prompt="prompt",
                schema={"title": "behavior"},
                validate=lambda payload: [] if payload.get("answer_behavior") == "attempted" else ["bad"],
            )

        self.assertEqual(result.model, "judge-model")
        self.assertEqual(generate.await_count, 2)
        self.assertEqual([call.kwargs["model"] for call in generate.await_args_list], ["judge-model", "judge-model"])

    async def test_attempted_response_gets_three_independent_semantic_labels(self) -> None:
        judge = FakeJudge(
            [
                JudgeResult({"answer_behavior": "attempted"}, "behavior-model"),
                JudgeResult({"faithfulness": "fully_supported"}, "faith-model"),
                JudgeResult({"correctness": "correct"}, "correct-model"),
                JudgeResult({"completeness": "complete"}, "complete-model"),
            ]
        )
        labeler = ResponseLabeler(
            triples={"t1": make_triple()},
            resources=make_resources(),
            config=make_config(),
            judge=judge,
        )

        label = await labeler.label(make_response())

        self.assertEqual(
            label.labels,
            {
                "answer_behavior": "attempted",
                "faithfulness": "fully_supported",
                "correctness": "correct",
                "completeness": "complete",
            },
        )
        self.assertEqual(label.metadata["behavior_judge_model"], "behavior-model")
        self.assertEqual(label.metadata["faithfulness_judge_model"], "faith-model")
        self.assertEqual(label.metadata["correctness_judge_model"], "correct-model")
        self.assertEqual(label.metadata["completeness_judge_model"], "complete-model")
        self.assertEqual(len(judge.prompts), 4)
        self.assertIn("Candidate context", judge.prompts[1])
        self.assertNotIn("Authoritative answer", judge.prompts[1])

    async def test_abstained_response_skips_semantic_judges(self) -> None:
        judge = FakeJudge([JudgeResult({"answer_behavior": "abstained"}, "behavior-model")])
        labeler = ResponseLabeler(
            triples={"t1": make_triple()},
            resources=make_resources(),
            config=make_config(),
            judge=judge,
        )

        label = await labeler.label(make_response())

        self.assertEqual(
            label.labels,
            {
                "answer_behavior": "abstained",
                "faithfulness": "not_applicable",
                "correctness": "not_applicable",
                "completeness": "not_applicable",
            },
        )
        self.assertEqual(len(judge.prompts), 1)
        self.assertIsNone(label.metadata["faithfulness_judge_model"])
        self.assertIsNone(label.metadata["correctness_judge_model"])
        self.assertIsNone(label.metadata["completeness_judge_model"])


if __name__ == "__main__":
    unittest.main()
