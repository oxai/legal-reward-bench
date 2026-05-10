from __future__ import annotations

from typing import Any

from common.records import Triple

from .common.config import PASSAGES_PER_CONTEXT
from .common.context import make_variant, with_gold_at_slot
from .common.randomness import deterministic_index


def build_gold_surrounding_variant(
    triple: Triple,
    *,
    corpus: list[dict[str, Any]],
    gold_index: int,
    gold_id: str,
) -> Triple:
    gold_slot = surrounding_gold_slot(
        corpus_size=len(corpus),
        gold_index=gold_index,
        triple_id=triple.id,
    )
    return make_variant(
        triple,
        variant="gold_surrounding",
        answerability="answerable",
        context_rows=surrounding_rows(corpus, gold_index, gold_slot=gold_slot),
        gold_context_ids=[gold_id],
    )


def build_surrounding_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if "gold_surrounding" not in selected_variants:
        return []

    corpus_index = {row["id"]: index for index, row in enumerate(corpus)}
    return [
        build_gold_surrounding_variant(
            triple,
            corpus=corpus,
            gold_index=corpus_index[triple.metadata["context_id"]],
            gold_id=triple.metadata["context_id"],
        )
        for triple in base_triples
    ]


def surrounding_rows(
    corpus: list[dict[str, Any]],
    gold_index: int,
    *,
    gold_slot: int,
) -> list[dict[str, Any]]:
    gold_row = corpus[gold_index]
    after_count = PASSAGES_PER_CONTEXT - gold_slot - 1
    start = gold_index - gold_slot
    end = gold_index + after_count + 1

    if start < 0 or end > len(corpus):
        raise ValueError(
            f"Cannot build {PASSAGES_PER_CONTEXT}-passage surrounding context for "
            f"gold index {gold_index}"
        )

    before = corpus[start:gold_index]
    after = corpus[gold_index + 1 : end]
    return with_gold_at_slot(gold_row, before + after, gold_index=gold_slot)


def surrounding_gold_slot(*, corpus_size: int, gold_index: int, triple_id: str) -> int:
    min_slot = max(0, PASSAGES_PER_CONTEXT - (corpus_size - gold_index))
    max_slot = min(PASSAGES_PER_CONTEXT - 1, gold_index)
    if min_slot > max_slot:
        raise ValueError(
            f"Cannot build {PASSAGES_PER_CONTEXT}-passage surrounding context for "
            f"gold index {gold_index}"
        )
    return min_slot + deterministic_index(
        max_slot - min_slot + 1,
        triple_id,
        "gold_surrounding",
        "gold_slot",
    )
