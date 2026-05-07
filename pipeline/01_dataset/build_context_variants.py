from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from common.llm import embed_texts
from common.records import Triple
from common.storage import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "isaacus/legal-rag-bench"
DEFAULT_INPUT = ROOT / "pipeline" / "01_dataset" / "outputs" / "triples" / "legal_rag_bench.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "pipeline" / "01_dataset" / "outputs" / "triples" / "legal_rag_bench_context_variants.jsonl"
)
CACHE_DIR = ROOT / "pipeline" / "01_dataset" / "outputs" / "cache"
EMBEDDING_MODEL = "nomic-embed-text"
RANDOM_SEED = 13
PASSAGES_PER_CONTEXT = 10
VARIANTS = (
    "gold_exact",
    "gold_surrounding",
    "gold_plus_random_distractors",
    "gold_plus_bm25_distractors",
    "random_context",
    "bm25_hard_negative",
    "embedding_hard_negative",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Legal RAG Bench context variants.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_triples = [Triple(**record) for record in read_jsonl(args.input)]
    if args.limit is not None:
        base_triples = base_triples[: args.limit]

    corpus = load_corpus()
    bm25 = BM25(corpus)
    embeddings = load_or_build_embeddings(corpus)
    variants = build_context_variants(base_triples, corpus=corpus, bm25=bm25, embeddings=embeddings)
    count = write_jsonl(args.output, (triple.to_dict() for triple in variants))
    print(f"Wrote {count} context-variant triples to {args.output}")


def load_corpus() -> list[dict[str, Any]]:
    rows = load_dataset(DATASET_NAME, name="corpus", split="test")
    return [dict(row) for row in rows]


def build_context_variants(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    bm25: "BM25",
    embeddings: "EmbeddingIndex",
) -> list[Triple]:
    rng = random.Random(RANDOM_SEED)
    corpus_by_id = {row["id"]: row for row in corpus}
    corpus_ids = [row["id"] for row in corpus]
    corpus_index = {row["id"]: index for index, row in enumerate(corpus)}
    output: list[Triple] = []

    for triple in base_triples:
        gold_id = triple.metadata["context_id"]
        gold_title = corpus_by_id[gold_id]["title"]
        excluded_negative_ids = ids_in_topic_family(corpus, gold_title)

        output.append(
            make_variant(
                triple,
                variant="gold_exact",
                answerability="answerable",
                context_rows=[corpus_by_id[gold_id]],
                gold_context_ids=[gold_id],
            )
        )
        output.append(
            make_variant(
                triple,
                variant="gold_surrounding",
                answerability="answerable",
                context_rows=surrounding_rows(corpus, corpus_index[gold_id]),
                gold_context_ids=[gold_id],
            )
        )
        random_distractors = sample_ids(
            corpus_ids,
            rng=rng,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude={gold_id},
        )
        output.append(
            make_variant(
                triple,
                variant="gold_plus_random_distractors",
                answerability="answerable",
                context_rows=[corpus_by_id[gold_id], *rows_for_ids(corpus_by_id, random_distractors)],
                gold_context_ids=[gold_id],
            )
        )
        bm25_distractors = bm25.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT - 1,
            exclude={gold_id},
        )
        output.append(
            make_variant(
                triple,
                variant="gold_plus_bm25_distractors",
                answerability="answerable",
                context_rows=[corpus_by_id[gold_id], *rows_for_ids(corpus_by_id, bm25_distractors)],
                gold_context_ids=[gold_id],
            )
        )
        random_negatives = sample_ids(
            corpus_ids,
            rng=rng,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        )
        output.append(
            make_variant(
                triple,
                variant="random_context",
                answerability="unanswerable",
                context_rows=rows_for_ids(corpus_by_id, random_negatives),
                gold_context_ids=[gold_id],
            )
        )
        bm25_negatives = bm25.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        )
        output.append(
            make_variant(
                triple,
                variant="bm25_hard_negative",
                answerability="unanswerable",
                context_rows=rows_for_ids(corpus_by_id, bm25_negatives),
                gold_context_ids=[gold_id],
            )
        )
        embedding_negatives = embeddings.search(
            triple.question,
            count=PASSAGES_PER_CONTEXT,
            exclude=excluded_negative_ids,
        )
        output.append(
            make_variant(
                triple,
                variant="embedding_hard_negative",
                answerability="unanswerable",
                context_rows=rows_for_ids(corpus_by_id, embedding_negatives),
                gold_context_ids=[gold_id],
            )
        )

    return output


def make_variant(
    triple: Triple,
    *,
    variant: str,
    answerability: str,
    context_rows: list[dict[str, Any]],
    gold_context_ids: list[str],
) -> Triple:
    return Triple(
        id=f"{triple.id}__{variant}",
        question=triple.question,
        context=format_passages(context_rows),
        answer=triple.answer,
        metadata={
            **triple.metadata,
            "base_triple_id": triple.id,
            "context_variant": variant,
            "answerability": answerability,
            "variant_context_ids": [row["id"] for row in context_rows],
            "gold_context_ids": gold_context_ids,
        },
    )


def format_passages(rows: list[dict[str, Any]]) -> str:
    if len(rows) == 1:
        return build_context(rows[0])
    return "\n\n".join(f"[Passage {index}]\n{build_context(row)}" for index, row in enumerate(rows, 1))


def build_context(row: dict[str, Any]) -> str:
    title = row["title"].strip()
    text = row["text"].strip()
    if not title:
        return text
    if text.startswith(title):
        return text
    return f"{title}\n\n{text}"


def surrounding_rows(corpus: list[dict[str, Any]], gold_index: int) -> list[dict[str, Any]]:
    start = max(0, gold_index - 4)
    end = min(len(corpus), gold_index + 6)

    if end - start < PASSAGES_PER_CONTEXT:
        missing = PASSAGES_PER_CONTEXT - (end - start)
        start = max(0, start - missing)
        end = min(len(corpus), start + PASSAGES_PER_CONTEXT)

    return corpus[start:end]


def ids_in_topic_family(corpus: list[dict[str, Any]], title: str) -> set[str]:
    family = topic_family(title)
    if not family:
        return {row["id"] for row in corpus if row["title"] == title}
    return {row["id"] for row in corpus if topic_family(row["title"]) == family}


def topic_family(title: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", title.strip())
    if not match:
        return ""
    parts = match.group(1).split(".")
    return ".".join(parts[:2])


def rows_for_ids(corpus_by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    return [corpus_by_id[id_] for id_ in ids]


def sample_ids(
    ids: list[str],
    *,
    rng: random.Random,
    count: int,
    exclude: set[str],
) -> list[str]:
    candidates = [id_ for id_ in ids if id_ not in exclude]
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, found {len(candidates)} after exclusions")
    return rng.sample(candidates, count)


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


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


class EmbeddingIndex:
    def __init__(self, *, doc_ids: list[str], vectors: np.ndarray) -> None:
        self.doc_ids = doc_ids
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)

    def search(self, query: str, *, count: int, exclude: set[str]) -> list[str]:
        query_vector = np.array(embed_texts(model=EMBEDDING_MODEL, texts=[query])[0], dtype=np.float32)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = self.vectors @ query_vector
        ranked = np.argsort(-scores)

        results: list[str] = []
        for index in ranked:
            doc_id = self.doc_ids[int(index)]
            if doc_id in exclude:
                continue
            results.append(doc_id)
            if len(results) == count:
                break
        if len(results) < count:
            raise ValueError(f"Need {count} embedding results, found {len(results)} after exclusions")
        return results


def load_or_build_embeddings(corpus: list[dict[str, Any]]) -> EmbeddingIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"corpus_embeddings__{EMBEDDING_MODEL.replace('-', '_')}.npz"
    doc_ids = [row["id"] for row in corpus]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(id_) for id_ in cached["doc_ids"]]
        if cached_ids == doc_ids:
            return EmbeddingIndex(doc_ids=doc_ids, vectors=cached["vectors"])

    texts = [row["title"] + "\n\n" + row["text"] for row in corpus]
    vectors = np.array(embed_texts(model=EMBEDDING_MODEL, texts=texts), dtype=np.float32)
    np.savez_compressed(cache_path, doc_ids=np.array(doc_ids), vectors=vectors)
    return EmbeddingIndex(doc_ids=doc_ids, vectors=vectors)


if __name__ == "__main__":
    main()
