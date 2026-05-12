from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "03_labels"))

from build import build_gold_context_map, resolve_gold_context  # noqa: E402


def make_triple(
    id_: str,
    *,
    base_id: str,
    variant: str,
    context: str,
) -> dict:
    return {
        "id": id_,
        "question": "question",
        "context": context,
        "answer": "answer",
        "metadata": {
            "base_triple_id": base_id,
            "context_variant": variant,
        },
    }


class GoldContextResolutionTests(unittest.TestCase):
    def test_uses_base_context_as_authoritative_gold_context(self) -> None:
        triples = {
            "q1": make_triple("q1", base_id="q1", variant="base", context="gold context with footnote"),
            "q1__random": make_triple(
                "q1__random",
                base_id="q1",
                variant="random_context",
                context="candidate context",
            ),
        }

        gold_contexts = build_gold_context_map(triples)

        self.assertEqual(gold_contexts, {"q1": "gold context with footnote"})
        self.assertEqual(resolve_gold_context(triples["q1__random"], gold_contexts), "gold context with footnote")

    def test_requires_base_context_in_triples_artifact(self) -> None:
        triples = {
            "q1__random": make_triple(
                "q1__random",
                base_id="q1",
                variant="random_context",
                context="candidate context",
            ),
        }

        with self.assertRaisesRegex(ValueError, "context_variant='base'"):
            build_gold_context_map(triples)


if __name__ == "__main__":
    unittest.main()
