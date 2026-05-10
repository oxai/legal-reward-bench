from __future__ import annotations

# ruff: noqa: E402

import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "01_triples"))

from common.records import Triple
from build import interleave_base_and_variants, make_base_triples
from sources.legal_rag_bench import build_context as source_build_context
from variants import build_variant_triples, parse_variant_selection
from variants.bm25 import build_bm25_variants
from variants.common.config import PASSAGES_PER_CONTEXT
from variants.common.context import build_context as variant_build_context
from variants.common.randomness import deterministic_index
from variants.common.vector_index import VectorIndex, load_cached_vectors, save_cached_vectors
from variants.contriever import build_contriever_variants
from variants.glove import build_glove_variants
from variants.nomic import build_nomic_variants
from variants.random import build_random_variants
from variants.surrounding import build_surrounding_variants, surrounding_gold_slot, surrounding_rows


def make_row(id_: str, title: str, text: str | None = None) -> dict[str, str]:
    return {
        "id": id_,
        "title": title,
        "text": text or f"{title} text",
    }


def make_triple(id_: str = "triple-1", context_id: str = "gold") -> Triple:
    return Triple(
        id=id_,
        question="What is the rule for juror excusal?",
        context="unused base context",
        answer="reference answer",
        metadata={
            "context_id": context_id,
            "context_title": "1.2 Gold",
            "dataset": "test",
        },
    )


def make_corpus() -> list[dict[str, str]]:
    return [
        make_row("pre0", "2.1 Before zero"),
        make_row("pre1", "2.2 Before one"),
        make_row("pre2", "2.3 Before two"),
        make_row("pre3", "2.4 Before three"),
        make_row("gold", "1.2 Gold", "UNIQUE_GOLD_PASSAGE juror excusal legal standard"),
        make_row("same-family", "1.2 Same family", "UNIQUE_SAME_FAMILY_PASSAGE"),
        make_row("d0", "3.1 Juror lexical", "juror excusal related distractor"),
        make_row("d1", "3.2 Juror lexical", "juror excusal related distractor"),
        make_row("d2", "3.3 Juror lexical", "juror excusal related distractor"),
        make_row("d3", "3.4 Juror lexical", "juror excusal related distractor"),
        make_row("d4", "3.5 Juror lexical", "juror excusal related distractor"),
        make_row("d5", "3.6 Juror lexical", "juror excusal related distractor"),
        make_row("d6", "3.7 Juror lexical", "juror excusal related distractor"),
        make_row("d7", "3.8 Juror lexical", "juror excusal related distractor"),
        make_row("d8", "3.9 Juror lexical", "juror excusal related distractor"),
        make_row("d9", "4.1 Juror lexical", "juror excusal related distractor"),
        make_row("d10", "4.2 Juror lexical", "juror excusal related distractor"),
        make_row("d11", "4.3 Juror lexical", "juror excusal related distractor"),
    ]


def topic_family(title: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
    if not match:
        return ""
    return ".".join(match.group(1).split(".")[:2])


class TripleVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triple = make_triple()
        self.corpus = make_corpus()
        self.corpus_by_id = {row["id"]: row for row in self.corpus}

    def assert_answerable_with_gold_slot(self, triple: Triple, variant: str) -> None:
        self.assertEqual(triple.metadata["context_variant"], variant)
        self.assertEqual(triple.metadata["answerability"], "answerable")
        self.assertEqual(len(triple.metadata["variant_context_ids"]), PASSAGES_PER_CONTEXT)
        self.assert_gold_slot_points_to(triple, "gold")
        self.assertEqual(triple.metadata["gold_context_ids"], ["gold"])

    def assert_unanswerable_negative(self, triple: Triple, variant: str) -> None:
        self.assertEqual(triple.metadata["context_variant"], variant)
        self.assertEqual(triple.metadata["answerability"], "unanswerable")
        self.assertEqual(len(triple.metadata["variant_context_ids"]), PASSAGES_PER_CONTEXT)
        self.assertNotIn("gold", triple.metadata["variant_context_ids"])
        self.assertNotIn("same-family", triple.metadata["variant_context_ids"])
        self.assertNotIn("UNIQUE_GOLD_PASSAGE", triple.context)
        self.assertNotIn("UNIQUE_SAME_FAMILY_PASSAGE", triple.context)
        self.assertEqual(triple.metadata["gold_context_ids"], ["gold"])

    def test_surrounding_variant_records_gold_slot(self) -> None:
        triples = build_surrounding_variants(
            [self.triple],
            corpus=self.corpus,
            selected_variants={"gold_surrounding"},
        )

        self.assertEqual(len(triples), 1)
        self.assert_answerable_with_gold_slot(triples[0], "gold_surrounding")

    def test_random_variants_are_deterministic_and_exclude_negatives(self) -> None:
        selected = {"gold_plus_random_distractors", "random_context"}
        first = build_random_variants([self.triple], corpus=self.corpus, selected_variants=selected)
        second = build_random_variants([self.triple], corpus=self.corpus, selected_variants=selected)

        self.assertEqual(
            [triple.metadata["variant_context_ids"] for triple in first],
            [triple.metadata["variant_context_ids"] for triple in second],
        )
        self.assert_answerable_with_gold_slot(first[0], "gold_plus_random_distractors")
        self.assertNotIn("same-family", first[0].metadata["variant_context_ids"])
        self.assert_unanswerable_negative(first[1], "random_context")

    def test_random_variants_are_independent_of_selected_random_methods(self) -> None:
        triples = [make_triple("triple-1"), make_triple("triple-2")]
        only_answerable = build_random_variants(
            triples,
            corpus=self.corpus,
            selected_variants={"gold_plus_random_distractors"},
        )
        both_random_variants = build_random_variants(
            triples,
            corpus=self.corpus,
            selected_variants={"gold_plus_random_distractors", "random_context"},
        )

        only_by_base_id = {
            triple.metadata["base_triple_id"]: triple.metadata["variant_context_ids"]
            for triple in only_answerable
        }
        both_answerable_by_base_id = {
            triple.metadata["base_triple_id"]: triple.metadata["variant_context_ids"]
            for triple in both_random_variants
            if triple.metadata["context_variant"] == "gold_plus_random_distractors"
        }
        self.assertEqual(only_by_base_id, both_answerable_by_base_id)

    def test_bm25_variants(self) -> None:
        triples = build_bm25_variants(
            [self.triple],
            corpus=self.corpus,
            selected_variants={"gold_plus_bm25_distractors", "bm25_hard_negative"},
        )

        self.assertEqual(len(triples), 2)
        self.assert_answerable_with_gold_slot(triples[0], "gold_plus_bm25_distractors")
        self.assertNotIn("same-family", triples[0].metadata["variant_context_ids"])
        self.assert_unanswerable_negative(triples[1], "bm25_hard_negative")

    def test_build_variant_triples_interleaves_by_base_triple_and_honors_empty_selection(self) -> None:
        triples = build_variant_triples(
            [make_triple("triple-1"), make_triple("triple-2")],
            corpus=self.corpus,
            variants={"gold_surrounding", "random_context"},
        )

        self.assertEqual(
            [(triple.metadata["base_triple_id"], triple.metadata["context_variant"]) for triple in triples],
            [
                ("triple-1", "gold_surrounding"),
                ("triple-1", "random_context"),
                ("triple-2", "gold_surrounding"),
                ("triple-2", "random_context"),
            ],
        )
        self.assertEqual(build_variant_triples([self.triple], corpus=self.corpus, variants=set()), [])
        with self.assertRaises(ValueError):
            build_variant_triples([self.triple], corpus=self.corpus, variants={"missing_variant"})

    def test_parse_variant_selection(self) -> None:
        self.assertEqual(
            parse_variant_selection(" gold_surrounding, random_context "),
            {"gold_surrounding", "random_context"},
        )
        self.assertIn("gold_surrounding", parse_variant_selection("all"))
        with self.assertRaises(ValueError):
            parse_variant_selection(" ")
        with self.assertRaises(ValueError):
            parse_variant_selection("missing_variant")

    def test_base_triples_use_passage_formatting(self) -> None:
        base = make_base_triples([self.triple])

        self.assertEqual(len(base), 1)
        self.assertTrue(base[0].context.startswith("[Passage 1]\n"))
        self.assertEqual(base[0].metadata["context_variant"], "base")
        self.assertEqual(base[0].metadata["gold_slot"], 1)
        self.assertEqual(base[0].metadata["variant_context_ids"], ["gold"])

    def test_interleave_base_and_variants_keeps_variants_with_base_triple(self) -> None:
        base = make_base_triples([make_triple("triple-1"), make_triple("triple-2")])
        variants = build_variant_triples(
            [make_triple("triple-1"), make_triple("triple-2")],
            corpus=self.corpus,
            variants={"random_context"},
        )

        self.assertEqual(
            [(triple.metadata["base_triple_id"], triple.metadata["context_variant"]) for triple in interleave_base_and_variants(base, variants)],
            [
                ("triple-1", "base"),
                ("triple-1", "random_context"),
                ("triple-2", "base"),
                ("triple-2", "random_context"),
            ],
        )

    def test_answerable_retrieval_gold_slot_is_deterministic_and_recorded(self) -> None:
        triples = build_bm25_variants(
            [self.triple],
            corpus=self.corpus,
            selected_variants={"gold_plus_bm25_distractors"},
        )
        expected_slot = deterministic_index(
            PASSAGES_PER_CONTEXT,
            self.triple.id,
            "gold_plus_bm25_distractors",
            "gold_slot",
        ) + 1

        self.assertEqual(triples[0].metadata["gold_slot"], expected_slot)
        self.assert_gold_slot_points_to(triples[0], "gold")

    def test_surrounding_handles_corpus_start_edge(self) -> None:
        slot = surrounding_gold_slot(corpus_size=len(self.corpus), gold_index=0, triple_id="edge")
        rows = surrounding_rows(self.corpus, 0, gold_slot=slot)

        self.assertEqual(slot, 0)
        self.assertEqual(len(rows), PASSAGES_PER_CONTEXT)
        self.assertEqual(rows[slot]["id"], "pre0")

    def test_context_formatting_keeps_existing_markdown_heading_and_includes_footnotes(self) -> None:
        row = make_row("with-footnote", "2.1 Views", "# 2.1 Views\n\nBody [^1].")
        row["footnotes"] = "[^1]: Footnote body."

        self.assertEqual(
            variant_build_context(row),
            "# 2.1 Views\n\nBody [^1].\n\n[^1]: Footnote body.",
        )
        self.assertEqual(
            source_build_context(
                title="2.1 Views",
                text="# 2.1 Views\n\nBody [^1].",
                footnotes="[^1]: Footnote body.",
            ),
            "# 2.1 Views\n\nBody [^1].\n\n[^1]: Footnote body.",
        )

    def test_context_formatting_keeps_existing_bold_markdown_heading(self) -> None:
        row = make_row(
            "bold-heading",
            "7.4.12 Stalking (From 7/6/11)",
            "# **7.4.12 Stalking (From 7/6/11)**\n\nBody.",
        )

        self.assertEqual(
            variant_build_context(row),
            "# **7.4.12 Stalking (From 7/6/11)**\n\nBody.",
        )
        self.assertEqual(
            source_build_context(
                title="7.4.12 Stalking (From 7/6/11)",
                text="# **7.4.12 Stalking (From 7/6/11)**\n\nBody.",
            ),
            "# **7.4.12 Stalking (From 7/6/11)**\n\nBody.",
        )

    def test_dense_method_variants_use_real_embeddings(self) -> None:
        corpus = self.corpus
        corpus_by_id = self.corpus_by_id
        integration_triple = make_triple(context_id="gold")
        dense_cases = [
            (
                build_nomic_variants,
                {"gold_plus_nomic_distractors", "nomic_hard_negative"},
                "gold_plus_nomic_distractors",
                "nomic_hard_negative",
            ),
            (
                build_glove_variants,
                {"gold_plus_glove_distractors", "glove_hard_negative"},
                "gold_plus_glove_distractors",
                "glove_hard_negative",
            ),
            (
                build_contriever_variants,
                {"gold_plus_contriever_distractors", "contriever_hard_negative"},
                "gold_plus_contriever_distractors",
                "contriever_hard_negative",
            ),
        ]

        for build_variants, selected, answerable_variant, negative_variant in dense_cases:
            with self.subTest(answerable_variant=answerable_variant):
                triples = build_variants(
                    [integration_triple],
                    corpus=corpus,
                    selected_variants=selected,
                )

                self.assertEqual(len(triples), 2)
                self.assert_answerable_with_gold_slot_for_gold(
                    triples[0],
                    answerable_variant,
                    gold_id="gold",
                    corpus_by_id=corpus_by_id,
                )
                self.assert_unanswerable_negative_for_gold_family(
                    triples[1],
                    negative_variant,
                    corpus_by_id=corpus_by_id,
                    gold_id="gold",
                )

    def test_unselected_methods_return_no_triples(self) -> None:
        self.assertEqual(
            build_random_variants([self.triple], corpus=self.corpus, selected_variants={"gold_surrounding"}),
            [],
        )
        self.assertEqual(
            build_bm25_variants([self.triple], corpus=self.corpus, selected_variants={"gold_surrounding"}),
            [],
        )

    def assert_answerable_with_gold_slot_for_gold(
        self,
        triple: Triple,
        variant: str,
        *,
        gold_id: str,
        corpus_by_id: dict[str, dict[str, str]],
    ) -> None:
        context_ids = triple.metadata["variant_context_ids"]
        gold_family = topic_family(corpus_by_id[gold_id]["title"])

        self.assertEqual(triple.metadata["context_variant"], variant)
        self.assertEqual(triple.metadata["answerability"], "answerable")
        self.assertEqual(len(context_ids), PASSAGES_PER_CONTEXT)
        self.assert_gold_slot_points_to(triple, gold_id)
        self.assertEqual(triple.metadata["gold_context_ids"], [gold_id])
        self.assertFalse(
            [
                context_id
                for context_id in context_ids
                if context_id != gold_id
                and topic_family(corpus_by_id[context_id]["title"]) == gold_family
            ]
        )

    def assert_gold_slot_points_to(self, triple: Triple, gold_id: str) -> None:
        gold_slot = triple.metadata["gold_slot"]
        self.assertGreaterEqual(gold_slot, 1)
        self.assertLessEqual(gold_slot, len(triple.metadata["variant_context_ids"]))
        self.assertEqual(triple.metadata["variant_context_ids"][gold_slot - 1], gold_id)

    def assert_unanswerable_negative_for_gold_family(
        self,
        triple: Triple,
        variant: str,
        *,
        corpus_by_id: dict[str, dict[str, str]],
        gold_id: str,
    ) -> None:
        context_ids = triple.metadata["variant_context_ids"]
        gold_family = topic_family(corpus_by_id[gold_id]["title"])
        gold_text = corpus_by_id[gold_id]["text"]

        self.assertEqual(triple.metadata["context_variant"], variant)
        self.assertEqual(triple.metadata["answerability"], "unanswerable")
        self.assertEqual(len(context_ids), PASSAGES_PER_CONTEXT)
        self.assertNotIn(gold_id, context_ids)
        self.assertNotIn(gold_text, triple.context)
        self.assertEqual(triple.metadata["gold_context_ids"], [gold_id])
        self.assertFalse(
            [context_id for context_id in context_ids if topic_family(corpus_by_id[context_id]["title"]) == gold_family]
        )


class RandomSamplerTests(unittest.TestCase):
    def test_sample_ids_excludes_ids_and_uses_supplied_rng(self) -> None:
        from variants.random import sample_ids

        ids = ["a", "b", "c", "d"]
        sampled = sample_ids(ids, rng=random.Random(13), count=2, exclude={"a"})

        self.assertEqual(len(sampled), 2)
        self.assertNotIn("a", sampled)
        self.assertEqual(sampled, sample_ids(ids, rng=random.Random(13), count=2, exclude={"a"}))


class VectorIndexTests(unittest.TestCase):
    def test_vector_index_tie_order_is_stable(self) -> None:
        index = VectorIndex(
            doc_ids=["a", "b", "c"],
            vectors=np.zeros((3, 2), dtype=np.float32),
            embed_query=lambda _: np.zeros(2, dtype=np.float32),
            label="test",
        )

        self.assertEqual(index.search("query", count=3, exclude=set()), ["a", "b", "c"])

    def test_cached_vectors_require_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "vectors.npz"
            doc_ids = ["a", "b"]
            vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            save_cached_vectors(
                cache_path,
                doc_ids=doc_ids,
                vectors=vectors,
                model="model",
                model_revision="revision",
                corpus_hash="corpus-hash",
                dataset_revision="dataset-revision",
            )

            self.assertIsNotNone(
                load_cached_vectors(
                    cache_path,
                    doc_ids,
                    model="model",
                    model_revision="revision",
                    corpus_hash="corpus-hash",
                    dataset_revision="dataset-revision",
                )
            )
            self.assertIsNone(
                load_cached_vectors(
                    cache_path,
                    doc_ids,
                    model="model",
                    model_revision="old-revision",
                    corpus_hash="corpus-hash",
                    dataset_revision="dataset-revision",
                )
            )
            self.assertIsNone(
                load_cached_vectors(
                    cache_path,
                    doc_ids,
                    model="model",
                    model_revision="revision",
                    corpus_hash="old-corpus-hash",
                    dataset_revision="dataset-revision",
                )
            )
            self.assertIsNone(
                load_cached_vectors(
                    cache_path,
                    doc_ids,
                    model="model",
                    model_revision="revision",
                    corpus_hash="corpus-hash",
                    dataset_revision="old-dataset-revision",
                )
            )
            self.assertIsNone(
                load_cached_vectors(
                    cache_path,
                    ["b", "a"],
                    model="model",
                    model_revision="revision",
                    corpus_hash="corpus-hash",
                    dataset_revision="dataset-revision",
                )
            )


if __name__ == "__main__":
    unittest.main()
