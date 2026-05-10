from __future__ import annotations

import subprocess
from typing import Any

import numpy as np

from common.llm import embed_texts
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

NOMIC_MODEL = "nomic-embed-text"
NOMIC_MODEL_ID = "0a109f422b47"
NOMIC_PREFIX_VERSION = "search-prefix-v1"
NOMIC_MODEL_REVISION = f"{NOMIC_MODEL_ID}:{NOMIC_PREFIX_VERSION}"
NOMIC_VARIANTS = {"gold_plus_nomic_distractors", "nomic_hard_negative"}


def load_or_build_nomic_embeddings(corpus: list[dict[str, Any]]) -> VectorIndex:
    assert_nomic_model_id()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = [row["id"] for row in corpus]
    corpus_hash = hash_corpus(corpus)
    dataset_revision = corpus_revision(corpus)
    cache_path = CACHE_DIR / f"corpus_embeddings__{NOMIC_MODEL.replace('-', '_')}__{corpus_hash[:12]}.npz"
    vectors = load_cached_vectors(
        cache_path,
        doc_ids,
        model=NOMIC_MODEL,
        model_revision=NOMIC_MODEL_REVISION,
        corpus_hash=corpus_hash,
        dataset_revision=dataset_revision,
    )

    if vectors is None:
        texts = [f"search_document: {build_context(row)}" for row in corpus]
        vectors = np.array(embed_texts(model=NOMIC_MODEL, texts=texts), dtype=np.float32)
        save_cached_vectors(
            cache_path,
            doc_ids=doc_ids,
            vectors=vectors,
            model=NOMIC_MODEL,
            model_revision=NOMIC_MODEL_REVISION,
            corpus_hash=corpus_hash,
            dataset_revision=dataset_revision,
        )

    return VectorIndex(
        doc_ids=doc_ids,
        vectors=vectors,
        embed_query=lambda query: np.array(
            embed_texts(model=NOMIC_MODEL, texts=[f"search_query: {query}"])[0],
            dtype=np.float32,
        ),
        label=NOMIC_MODEL,
    )


def assert_nomic_model_id() -> None:
    installed_id = installed_ollama_model_id(NOMIC_MODEL)
    if installed_id != NOMIC_MODEL_ID:
        raise RuntimeError(
            f"Expected Ollama model {NOMIC_MODEL} to have ID {NOMIC_MODEL_ID}, "
            f"found {installed_id or 'not installed'}."
        )


def installed_ollama_model_id(model: str) -> str | None:
    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].split(":", 1)[0] == model:
            return parts[1]
    return None


def build_nomic_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if not selected_variants.intersection(NOMIC_VARIANTS):
        return []

    nomic_embeddings = load_or_build_nomic_embeddings(corpus)
    corpus_by_id = {row["id"]: row for row in corpus}
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_row = corpus_by_id[gold_id]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_row["title"])

        if "gold_plus_nomic_distractors" in selected_variants:
            output.append(
                build_gold_plus_nomic_distractors_variant(
                    triple,
                    gold_row=gold_row,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_distractor_ids=excluded_negative_ids,
                    nomic_embeddings=nomic_embeddings,
                )
            )
        if "nomic_hard_negative" in selected_variants:
            output.append(
                build_nomic_hard_negative_variant(
                    triple,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_negative_ids=excluded_negative_ids,
                    nomic_embeddings=nomic_embeddings,
                )
            )

    return output


def build_gold_plus_nomic_distractors_variant(
    triple: Triple,
    *,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_distractor_ids: set[str],
    nomic_embeddings: VectorIndex,
) -> Triple:
    return make_answerable_retrieval_variant(
        triple,
        variant="gold_plus_nomic_distractors",
        gold_row=gold_row,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        distractor_ids=nomic_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude=excluded_distractor_ids,
        ),
    )


def build_nomic_hard_negative_variant(
    triple: Triple,
    *,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_negative_ids: set[str],
    nomic_embeddings: VectorIndex,
) -> Triple:
    return make_unanswerable_retrieval_variant(
        triple,
        variant="nomic_hard_negative",
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        context_ids=nomic_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        ),
    )
