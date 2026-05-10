from __future__ import annotations

import re
from typing import Any

from common.records import Triple

from .config import PASSAGES_PER_CONTEXT
from .randomness import deterministic_index


def make_variant(
    triple: Triple,
    *,
    variant: str,
    answerability: str,
    context_rows: list[dict[str, Any]],
    gold_context_ids: list[str],
) -> Triple:
    metadata = {
        **triple.metadata,
        "base_triple_id": triple.id,
        "context_variant": variant,
        "answerability": answerability,
        "variant_context_ids": [row["id"] for row in context_rows],
        "gold_context_ids": gold_context_ids,
    }
    gold_slots = [
        index
        for index, row in enumerate(context_rows, start=1)
        if row["id"] in set(gold_context_ids)
    ]
    if gold_slots:
        metadata["gold_slot"] = gold_slots[0]

    return Triple(
        id=f"{triple.id}__{variant}",
        question=triple.question,
        context=format_passages(context_rows),
        answer=triple.answer,
        metadata=metadata,
    )


def make_answerable_retrieval_variant(
    triple: Triple,
    *,
    variant: str,
    gold_row: dict[str, Any],
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    distractor_ids: list[str],
) -> Triple:
    gold_index = deterministic_index(PASSAGES_PER_CONTEXT, triple.id, variant, "gold_slot")
    return make_variant(
        triple,
        variant=variant,
        answerability="answerable",
        context_rows=with_gold_at_slot(
            gold_row,
            rows_for_ids(corpus_by_id, distractor_ids),
            gold_index=gold_index,
        ),
        gold_context_ids=[gold_id],
    )


def make_unanswerable_retrieval_variant(
    triple: Triple,
    *,
    variant: str,
    gold_id: str,
    corpus_by_id: dict[str, dict[str, Any]],
    context_ids: list[str],
) -> Triple:
    return make_variant(
        triple,
        variant=variant,
        answerability="unanswerable",
        context_rows=rows_for_ids(corpus_by_id, context_ids),
        gold_context_ids=[gold_id],
    )


def format_passages(rows: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[Passage {index}]\n{build_context(row)}" for index, row in enumerate(rows, 1))


def build_context(row: dict[str, Any]) -> str:
    title = row["title"].strip()
    text = row["text"].strip()
    footnotes = (row.get("footnotes") or "").strip()
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


def with_gold_at_slot(
    gold_row: dict[str, Any],
    distractor_rows: list[dict[str, Any]],
    *,
    gold_index: int,
) -> list[dict[str, Any]]:
    if len(distractor_rows) != PASSAGES_PER_CONTEXT - 1:
        raise ValueError(
            f"Need {PASSAGES_PER_CONTEXT - 1} distractors, found {len(distractor_rows)}"
        )
    if not 0 <= gold_index < PASSAGES_PER_CONTEXT:
        raise ValueError(f"gold_index must be between 0 and {PASSAGES_PER_CONTEXT - 1}")
    if any(row["id"] == gold_row["id"] for row in distractor_rows):
        raise ValueError(f"Gold context {gold_row['id']} appeared among distractors")

    return [
        *distractor_rows[:gold_index],
        gold_row,
        *distractor_rows[gold_index:],
    ]
