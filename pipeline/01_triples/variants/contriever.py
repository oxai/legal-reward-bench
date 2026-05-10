from __future__ import annotations

from typing import Any

import numpy as np

from common.records import Triple

from .common.config import CACHE_DIR, PASSAGES_PER_CONTEXT
from .common.context import (
    build_context,
    ids_in_topic_family,
    make_answerable_retrieval_variant,
    make_unanswerable_retrieval_variant,
)
from .common.vector_index import (
    VectorIndex,
    corpus_revision,
    hash_corpus,
    load_cached_vectors,
    save_cached_vectors,
)

CONTRIEVER_MODEL = "facebook/contriever-msmarco"
CONTRIEVER_REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"
CONTRIEVER_BATCH_SIZE = 16
CONTRIEVER_VARIANTS = {"gold_plus_contriever_distractors", "contriever_hard_negative"}


def load_or_build_contriever_embeddings(corpus: list[dict[str, Any]]) -> VectorIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = [row["id"] for row in corpus]
    corpus_hash = hash_corpus(corpus)
    dataset_revision = corpus_revision(corpus)
    cache_path = CACHE_DIR / f"corpus_embeddings__facebook_contriever_msmarco__{corpus_hash[:12]}.npz"
    vectors = load_cached_vectors(
        cache_path,
        doc_ids,
        model=CONTRIEVER_MODEL,
        model_revision=CONTRIEVER_REVISION,
        corpus_hash=corpus_hash,
        dataset_revision=dataset_revision,
    )
    encoder = ContrieverEncoder()

    if vectors is None:
        texts = [build_context(row) for row in corpus]
        vectors = encoder.embed_texts(texts)
        save_cached_vectors(
            cache_path,
            doc_ids=doc_ids,
            vectors=vectors,
            model=CONTRIEVER_MODEL,
            model_revision=CONTRIEVER_REVISION,
            corpus_hash=corpus_hash,
            dataset_revision=dataset_revision,
        )

    return VectorIndex(
        doc_ids=doc_ids,
        vectors=vectors,
        embed_query=lambda query: encoder.embed_texts([query])[0],
        label=CONTRIEVER_MODEL,
    )


def build_contriever_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if not selected_variants.intersection(CONTRIEVER_VARIANTS):
        return []

    contriever_embeddings = load_or_build_contriever_embeddings(corpus)
    corpus_by_id = {row["id"]: row for row in corpus}
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_row = corpus_by_id[gold_id]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_row["title"])

        if "gold_plus_contriever_distractors" in selected_variants:
            output.append(
                build_gold_plus_contriever_distractors_variant(
                    triple,
                    gold_row=gold_row,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_distractor_ids=excluded_negative_ids,
                    contriever_embeddings=contriever_embeddings,
                )
            )
        if "contriever_hard_negative" in selected_variants:
            output.append(
                build_contriever_hard_negative_variant(
                    triple,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_negative_ids=excluded_negative_ids,
                    contriever_embeddings=contriever_embeddings,
                )
            )

    return output


def build_gold_plus_contriever_distractors_variant(
    triple: Triple,
    *,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_distractor_ids: set[str],
    contriever_embeddings: VectorIndex,
) -> Triple:
    return make_answerable_retrieval_variant(
        triple,
        variant="gold_plus_contriever_distractors",
        gold_row=gold_row,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        distractor_ids=contriever_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude=excluded_distractor_ids,
        ),
    )


def build_contriever_hard_negative_variant(
    triple: Triple,
    *,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_negative_ids: set[str],
    contriever_embeddings: VectorIndex,
) -> Triple:
    return make_unanswerable_retrieval_variant(
        triple,
        variant="contriever_hard_negative",
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        context_ids=contriever_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        ),
    )


class ContrieverEncoder:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONTRIEVER_MODEL,
            revision=CONTRIEVER_REVISION,
        )
        self.model = AutoModel.from_pretrained(
            CONTRIEVER_MODEL,
            revision=CONTRIEVER_REVISION,
        )
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), CONTRIEVER_BATCH_SIZE):
            batch = texts[start : start + CONTRIEVER_BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                outputs = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state * mask
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = token_embeddings.sum(dim=1) / lengths
            vectors.extend(pooled.cpu().numpy().astype(np.float32))
        return np.array(vectors, dtype=np.float32)
