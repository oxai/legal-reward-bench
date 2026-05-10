from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from common.records import Triple

from .common.config import PASSAGES_PER_CONTEXT
from .common.context import (
    ids_in_topic_family,
    make_answerable_retrieval_variant,
    make_unanswerable_retrieval_variant,
)
from .common.tokenize import tokenize


BM25_VARIANTS = {"gold_plus_bm25_distractors", "bm25_hard_negative"}


class BM25:
    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.doc_ids = [row["id"] for row in corpus]
        self.tokens = [tokenize(row["title"] + " " + row["text"]) for row in corpus]
        self.index = BM25Okapi(self.tokens)

    def search(self, query: str, *, count: int, exclude: set[str]) -> list[str]:
        scores = self.index.get_scores(tokenize(query))
        scored = []
        for index, (score, doc_id) in enumerate(zip(scores, self.doc_ids, strict=True)):
            if doc_id in exclude:
                continue
            scored.append((float(score), -index, doc_id))
        scored.sort(reverse=True)
        results = [doc_id for _, _, doc_id in scored[:count]]
        if len(results) < count:
            raise ValueError(f"Need {count} BM25 results, found {len(results)} after exclusions")
        return results


def build_bm25_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if not selected_variants.intersection(BM25_VARIANTS):
        return []

    bm25 = BM25(corpus)
    corpus_by_id = {row["id"]: row for row in corpus}
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_row = corpus_by_id[gold_id]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_row["title"])

        if "gold_plus_bm25_distractors" in selected_variants:
            output.append(
                build_gold_plus_bm25_distractors_variant(
                    triple,
                    gold_row=gold_row,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_distractor_ids=excluded_negative_ids,
                    bm25=bm25,
                )
            )
        if "bm25_hard_negative" in selected_variants:
            output.append(
                build_bm25_hard_negative_variant(
                    triple,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_negative_ids=excluded_negative_ids,
                    bm25=bm25,
                )
            )

    return output


def build_gold_plus_bm25_distractors_variant(
    triple: Triple,
    *,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_distractor_ids: set[str],
    bm25: BM25,
) -> Triple:
    return make_answerable_retrieval_variant(
        triple,
        variant="gold_plus_bm25_distractors",
        gold_row=gold_row,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        distractor_ids=bm25.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude=excluded_distractor_ids,
        ),
    )


def build_bm25_hard_negative_variant(
    triple: Triple,
    *,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_negative_ids: set[str],
    bm25: BM25,
) -> Triple:
    return make_unanswerable_retrieval_variant(
        triple,
        variant="bm25_hard_negative",
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        context_ids=bm25.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        ),
    )
