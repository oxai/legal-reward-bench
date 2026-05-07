from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from common.records import Triple
from common.storage import write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "isaacus/legal-rag-bench"
DEFAULT_OUTPUT = ROOT / "pipeline" / "01_dataset" / "outputs" / "triples" / "legal_rag_bench.jsonl"

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Legal RAG Bench triples.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    triples = prepare_triples(limit=args.limit)
    count = write_jsonl(args.output, (triple.to_dict() for triple in triples))
    print(f"Wrote {count} triples to {args.output}")


def prepare_triples(limit: int | None = None) -> list[Triple]:
    corpus = load_dataset(DATASET_NAME, name="corpus", split="test")
    qa = load_dataset(DATASET_NAME, name="qa", split="test")

    corpus_by_id = {row["id"]: row for row in corpus}
    rows = qa if limit is None else islice(qa, limit)

    triples: list[Triple] = []
    for row in rows:
        passage = corpus_by_id[row["relevant_passage_id"]]
        triples.append(
            Triple(
                id=f"legal-rag-bench-{int(row['id']):04d}",
                question=row["question"],
                context=build_context(title=passage["title"], text=passage["text"]),
                answer=row["answer"],
                metadata={
                    "dataset": DATASET_NAME,
                    "qa_id": row["id"],
                    "context_id": row["relevant_passage_id"],
                    "context_title": passage["title"],
                    "source_subset": "qa",
                    "corpus_subset": "corpus",
                },
            )
        )
    return triples


def build_context(*, title: str, text: str) -> str:
    title = title.strip()
    text = text.strip()
    if not title:
        return text
    return f"{title}\n\n{text}"


if __name__ == "__main__":
    main()
