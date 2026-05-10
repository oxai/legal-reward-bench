from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class VectorIndex:
    def __init__(
        self,
        *,
        doc_ids: list[str],
        vectors: np.ndarray,
        embed_query: Callable[[str], np.ndarray],
        label: str,
    ) -> None:
        self.doc_ids = doc_ids
        self.embed_query = embed_query
        self.label = label
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)

    def search(self, query: str, *, count: int, exclude: set[str]) -> list[str]:
        query_vector = self.embed_query(query)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = self.vectors @ query_vector
        ranked = np.argsort(-scores, kind="stable")

        results: list[str] = []
        for index in ranked:
            doc_id = self.doc_ids[int(index)]
            if doc_id in exclude:
                continue
            results.append(doc_id)
            if len(results) == count:
                break
        if len(results) < count:
            raise ValueError(
                f"Need {count} {self.label} results, found {len(results)} after exclusions"
            )
        return results


def load_cached_vectors(
    cache_path: Path,
    doc_ids: list[str],
    *,
    model: str,
    model_revision: str,
    corpus_hash: str,
    dataset_revision: str,
) -> np.ndarray | None:
    # Embedding caches are only reproducible with the pinned dependency versions in pyproject.toml.
    if not cache_path.exists():
        return None

    cached = np.load(cache_path, allow_pickle=False)
    required_keys = {
        "doc_ids",
        "vectors",
        "model",
        "model_revision",
        "corpus_hash",
        "dataset_revision",
        "vector_dimension",
    }
    if not required_keys.issubset(cached.files):
        return None

    cached_ids = [str(id_) for id_ in cached["doc_ids"]]
    if cached_ids != doc_ids:
        return None

    vectors = np.array(cached["vectors"], dtype=np.float32)
    if cached_string(cached, "model") != model:
        return None
    if cached_string(cached, "model_revision") != model_revision:
        return None
    if cached_string(cached, "corpus_hash") != corpus_hash:
        return None
    if cached_string(cached, "dataset_revision") != dataset_revision:
        return None
    if cached_int(cached, "vector_dimension") != vectors.shape[1]:
        return None
    return vectors


def save_cached_vectors(
    cache_path: Path,
    *,
    doc_ids: list[str],
    vectors: np.ndarray,
    model: str,
    model_revision: str,
    corpus_hash: str,
    dataset_revision: str,
) -> None:
    np.savez_compressed(
        cache_path,
        doc_ids=np.array(doc_ids),
        vectors=vectors,
        model=np.array(model),
        model_revision=np.array(model_revision),
        corpus_hash=np.array(corpus_hash),
        dataset_revision=np.array(dataset_revision),
        vector_dimension=np.array(vectors.shape[1], dtype=np.int32),
    )


def hash_corpus(corpus: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in corpus:
        for key in ("id", "title", "text", "footnotes", "_dataset_revision"):
            digest.update(str(row.get(key, "")).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def corpus_revision(corpus: list[dict[str, Any]]) -> str:
    revisions = {str(row.get("_dataset_revision", "")) for row in corpus}
    if len(revisions) != 1:
        raise ValueError(f"Expected one dataset revision in corpus, found {sorted(revisions)}")
    return revisions.pop()


def cached_string(cached: np.lib.npyio.NpzFile, key: str) -> str:
    return str(cached[key].item())


def cached_int(cached: np.lib.npyio.NpzFile, key: str) -> int:
    return int(cached[key].item())
