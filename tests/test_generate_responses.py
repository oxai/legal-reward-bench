from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "02_responses"))

from common.llm import TextGeneration
from common.records import Triple
from generate import generate_responses


def make_triple(id_: str) -> Triple:
    return Triple(
        id=id_,
        question=f"question {id_}",
        context=f"context {id_}",
        answer="reference",
        metadata={"context_id": f"context-{id_}"},
    )


class GenerateResponsesTests(unittest.TestCase):
    def test_serial_generation_preserves_order_and_records_concurrency(self) -> None:
        triples = [make_triple("t1"), make_triple("t2")]

        async def fake_generate_text_async(**kwargs: object) -> TextGeneration:
            prompt = str(kwargs["prompt"])
            if prompt.endswith("question t1"):
                return TextGeneration(text="answer 1", finish_reason="stop")
            return TextGeneration(text="answer 2", finish_reason="length")

        with patch("generate.generate_text_async", side_effect=fake_generate_text_async) as mock_generate:
            responses = generate_responses(
                triples,
                model="test-model",
                prompt_template="{context}\n{question}",
                prompt_version="prompt_v1",
                temperature=0.0,
                max_tokens=128,
                think=False,
                concurrency=1,
            )

        self.assertEqual([response.triple_id for response in responses], ["t1", "t2"])
        self.assertEqual([response.response for response in responses], ["answer 1", "answer 2"])
        self.assertEqual([response.metadata["concurrency"] for response in responses], [1, 1])
        self.assertEqual([response.metadata["finish_reason"] for response in responses], ["stop", "length"])
        self.assertEqual(mock_generate.call_count, 2)

    def test_concurrent_generation_preserves_order_and_records_concurrency(self) -> None:
        triples = [make_triple("t1"), make_triple("t2"), make_triple("t3")]

        async def fake_generate_text_async(**kwargs: object) -> TextGeneration:
            prompt = str(kwargs["prompt"])
            return TextGeneration(
                text=f"answer for {prompt.splitlines()[-1]}",
                finish_reason="stop",
            )

        with patch(
            "generate.generate_text_async",
            side_effect=fake_generate_text_async,
        ) as mock_generate:
            responses = generate_responses(
                triples,
                model="test-model",
                prompt_template="{context}\n{question}",
                prompt_version="prompt_v1",
                temperature=0.0,
                max_tokens=128,
                think=False,
                concurrency=2,
            )

        self.assertEqual([response.triple_id for response in responses], ["t1", "t2", "t3"])
        self.assertEqual(
            [response.response for response in responses],
            ["answer for question t1", "answer for question t2", "answer for question t3"],
        )
        self.assertEqual([response.metadata["concurrency"] for response in responses], [2, 2, 2])
        self.assertEqual([response.metadata["finish_reason"] for response in responses], ["stop", "stop", "stop"])
        self.assertEqual(mock_generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
