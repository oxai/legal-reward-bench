from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

from datasets import load_dataset

from common.records import Triple

DATASET_NAME = "isaacus/legal-rag-bench"
# Hugging Face dataset commit SHA; pinning keeps triples and embedding caches reproducible.
DATASET_REVISION = "db0b31dc6d195ce9916897e1ac5e4e6209736c8a"


@dataclass(frozen=True)
class SourceData:
    triples: list[Triple]
    corpus: list[dict[str, Any]]


def load_source(limit: int | None = None) -> SourceData:
    corpus = [
        {**dict(row), "_dataset_revision": DATASET_REVISION}
        for row in load_dataset(DATASET_NAME, name="corpus", split="test", revision=DATASET_REVISION)
    ]
    qa = load_dataset(DATASET_NAME, name="qa", split="test", revision=DATASET_REVISION)

    corpus_by_id = {row["id"]: row for row in corpus}
    if len(corpus_by_id) != len(corpus):
        raise ValueError("Legal RAG Bench corpus contains duplicate ids.")
    rows = qa if limit is None else islice(qa, limit)

    triples: list[Triple] = []
    for row in rows:
        passage = corpus_by_id[row["relevant_passage_id"]]
        triples.append(
            Triple(
                id=f"legal-rag-bench-{int(row['id']):04d}",
                question=row["question"],
                context=build_context(
                    title=passage["title"],
                    text=passage["text"],
                    footnotes=passage.get("footnotes", ""),
                ),
                answer=row["answer"],
                metadata={
                    "dataset": DATASET_NAME,
                    "dataset_revision": DATASET_REVISION,
                    "qa_id": row["id"],
                    "context_id": row["relevant_passage_id"],
                    "context_title": passage["title"],
                    "source_subset": "qa",
                    "corpus_subset": "corpus",
                },
            )
        )
    return SourceData(triples=triples, corpus=corpus)


def build_context(*, title: str, text: str, footnotes: str = "") -> str:
    title = title.strip()
    text = text.strip()
    footnotes = (footnotes or "").strip()
    if not title:
        context = text
    elif text_has_title(text, title):
        context = text
    else:
        context = f"{title}\n\n{text}"
    if footnotes:
        return f"{context}\n\n{footnotes}"
    return context


def text_has_title(text: str, title: str) -> bool:
    first_line = text.lstrip().splitlines()[0].strip() if text.strip() else ""
    return normalize_heading(first_line) == normalize_heading(title)


def normalize_heading(value: str) -> str:
    value = value.strip().lstrip("#").strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "*_":
        value = value[1:-1].strip()
    return value
