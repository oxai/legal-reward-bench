from __future__ import annotations

import random as py_random

from typing import Any

from common.records import Triple

from .common.config import PASSAGES_PER_CONTEXT, RANDOM_SEED
from .common.context import (
    ids_in_topic_family,
    make_answerable_retrieval_variant,
    make_unanswerable_retrieval_variant,
)
from .common.randomness import deterministic_seed


RANDOM_VARIANTS = {"gold_plus_random_distractors", "random_context"}


def build_random_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if not selected_variants.intersection(RANDOM_VARIANTS):
        return []

    corpus_by_id = {row["id"]: row for row in corpus}
    corpus_ids = [row["id"] for row in corpus]
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_row = corpus_by_id[gold_id]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_row["title"])

        if "gold_plus_random_distractors" in selected_variants:
            output.append(
                build_gold_plus_random_distractors_variant(
                    triple,
                    gold_row=gold_row,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    corpus_ids=corpus_ids,
                    excluded_distractor_ids=excluded_negative_ids,
                )
            )
        if "random_context" in selected_variants:
            output.append(
                build_random_context_variant(
                    triple,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    corpus_ids=corpus_ids,
                    excluded_negative_ids=excluded_negative_ids,
                )
            )

    return output


def build_gold_plus_random_distractors_variant(
    triple: Triple,
    *,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    corpus_ids: list[str],
    excluded_distractor_ids: set[str],
) -> Triple:
    variant = "gold_plus_random_distractors"
    return make_answerable_retrieval_variant(
        triple,
        variant=variant,
        gold_row=gold_row,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        distractor_ids=sample_ids(
            corpus_ids,
            rng=rng_for(triple.id, variant),
            count=PASSAGES_PER_CONTEXT - 1,
            exclude=excluded_distractor_ids,
        ),
    )


def build_random_context_variant(
    triple: Triple,
    *,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    corpus_ids: list[str],
    excluded_negative_ids: set[str],
) -> Triple:
    variant = "random_context"
    return make_unanswerable_retrieval_variant(
        triple,
        variant=variant,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        context_ids=sample_ids(
            corpus_ids,
            rng=rng_for(triple.id, variant),
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        ),
    )


def sample_ids(
    ids: list[str],
    *,
    rng: py_random.Random,
    count: int,
    exclude: set[str],
) -> list[str]:
    candidates = [id_ for id_ in ids if id_ not in exclude]
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)} after exclusions")
    return rng.sample(candidates, count)


def rng_for(triple_id: str, variant: str) -> py_random.Random:
    return py_random.Random(deterministic_seed(RANDOM_SEED, triple_id, variant))
