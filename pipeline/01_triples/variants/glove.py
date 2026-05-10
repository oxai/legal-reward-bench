from __future__ import annotations

import gzip
import hashlib
import shutil
import urllib.request
from collections.abc import Iterable
from pathlib import Path
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
from .common.tokenize import tokenize
from .common.vector_index import (
    VectorIndex,
    corpus_revision,
    hash_corpus,
    load_cached_vectors,
    save_cached_vectors,
)

GLOVE_MODEL = "glove-wiki-gigaword-50"
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
GLOVE_MD5 = "c289bc5d7f2f02c6dc9f2f9b67641813"
GLOVE_DIMENSIONS = 50
GLOVE_VARIANTS = {"gold_plus_glove_distractors", "glove_hard_negative"}


class GloveVectors:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        token_vectors = [self.vectors[token] for token in tokenize(text) if token in self.vectors]
        if not token_vectors:
            return np.zeros(GLOVE_DIMENSIONS, dtype=np.float32)
        return np.mean(token_vectors, axis=0, dtype=np.float32)


def load_or_build_glove_embeddings(corpus: list[dict[str, Any]], triples: list[Triple]) -> VectorIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = [row["id"] for row in corpus]
    query_vocabulary = vocabulary_from_texts(triple.question for triple in triples)
    corpus_hash = hash_corpus(corpus)
    dataset_revision = corpus_revision(corpus)
    cache_path = CACHE_DIR / f"corpus_embeddings__{GLOVE_MODEL.replace('-', '_')}__{corpus_hash[:12]}.npz"
    vectors = load_cached_vectors(
        cache_path,
        doc_ids,
        model=GLOVE_MODEL,
        model_revision=GLOVE_MD5,
        corpus_hash=corpus_hash,
        dataset_revision=dataset_revision,
    )

    if vectors is not None:
        glove = load_glove_vectors(query_vocabulary)
        return VectorIndex(
            doc_ids=doc_ids,
            vectors=vectors,
            embed_query=glove.embed,
            label=GLOVE_MODEL,
        )

    texts = [build_context(row) for row in corpus]
    glove = load_glove_vectors(query_vocabulary | vocabulary_from_texts(texts))
    vectors = np.array([glove.embed(text) for text in texts], dtype=np.float32)
    save_cached_vectors(
        cache_path,
        doc_ids=doc_ids,
        vectors=vectors,
        model=GLOVE_MODEL,
        model_revision=GLOVE_MD5,
        corpus_hash=corpus_hash,
        dataset_revision=dataset_revision,
    )
    return VectorIndex(
        doc_ids=doc_ids,
        vectors=vectors,
        embed_query=glove.embed,
        label=GLOVE_MODEL,
    )


def build_glove_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    selected_variants: set[str],
) -> list[Triple]:
    if not selected_variants.intersection(GLOVE_VARIANTS):
        return []

    glove_embeddings = load_or_build_glove_embeddings(corpus, base_triples)
    corpus_by_id = {row["id"]: row for row in corpus}
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_row = corpus_by_id[gold_id]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_row["title"])

        if "gold_plus_glove_distractors" in selected_variants:
            output.append(
                build_gold_plus_glove_distractors_variant(
                    triple,
                    gold_row=gold_row,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_distractor_ids=excluded_negative_ids,
                    glove_embeddings=glove_embeddings,
                )
            )
        if "glove_hard_negative" in selected_variants:
            output.append(
                build_glove_hard_negative_variant(
                    triple,
                    gold_id=gold_id,
                    corpus_by_id=corpus_by_id,
                    excluded_negative_ids=excluded_negative_ids,
                    glove_embeddings=glove_embeddings,
                )
            )

    return output


def build_gold_plus_glove_distractors_variant(
    triple: Triple,
    *,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_distractor_ids: set[str],
    glove_embeddings: VectorIndex,
) -> Triple:
    return make_answerable_retrieval_variant(
        triple,
        variant="gold_plus_glove_distractors",
        gold_row=gold_row,
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        distractor_ids=glove_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude=excluded_distractor_ids,
        ),
    )


def build_glove_hard_negative_variant(
    triple: Triple,
    *,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    excluded_negative_ids: set[str],
    glove_embeddings: VectorIndex,
) -> Triple:
    return make_unanswerable_retrieval_variant(
        triple,
        variant="glove_hard_negative",
        gold_id=gold_id,
        corpus_by_id=corpus_by_id,
        context_ids=glove_embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        ),
    )


def vocabulary_from_texts(texts: Iterable[str]) -> set[str]:
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(tokenize(text))
    return vocabulary


def load_glove_vectors(vocabulary: set[str]) -> GloveVectors:
    glove_path = ensure_glove_file()
    vectors: dict[str, np.ndarray] = {}

    with gzip.open(glove_path, "rt", encoding="utf-8") as file:
        header = next(file).strip().split()
        if header != ["400000", str(GLOVE_DIMENSIONS)]:
            raise RuntimeError(f"Unexpected {GLOVE_MODEL} header: {' '.join(header)}")

        for line in file:
            parts = line.rstrip().split(" ")
            token = parts[0]
            if token not in vocabulary:
                continue
            vector = np.array(parts[1:], dtype=np.float32)
            if vector.shape != (GLOVE_DIMENSIONS,):
                raise RuntimeError(f"Unexpected vector size for GloVe token {token!r}")
            vectors[token] = vector

    return GloveVectors(vectors)


def ensure_glove_file() -> Path:
    glove_dir = CACHE_DIR / "glove"
    glove_dir.mkdir(parents=True, exist_ok=True)
    glove_path = glove_dir / f"{GLOVE_MODEL}.gz"

    if glove_path.exists() and file_md5(glove_path) == GLOVE_MD5:
        return glove_path

    temp_path = glove_path.with_suffix(".gz.tmp")
    print(f"Downloading {GLOVE_MODEL} vectors to {glove_path}")
    with urllib.request.urlopen(GLOVE_URL, timeout=60) as response:
        with temp_path.open("wb") as file:
            shutil.copyfileobj(response, file)

    actual_md5 = file_md5(temp_path)
    if actual_md5 != GLOVE_MD5:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded {GLOVE_MODEL} checksum mismatch: {actual_md5}")

    temp_path.replace(glove_path)
    return glove_path


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
