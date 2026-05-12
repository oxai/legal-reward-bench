from __future__ import annotations

from typing import Any

from datasets import load_dataset

DATASET_NAME = "Salesforce/ContextualJudgeBench"
DATASET_REVISION = None

ALL_SPLITS = (
    "refusal_answerable",
    "refusal_unanswerable",
    "faithfulness_qa",
    "completeness_qa",
    "conciseness_qa",
    "faithfulness_summ",
    "completeness_summ",
    "conciseness_summ",
)

QA_SPLITS = (
    "refusal_answerable",
    "refusal_unanswerable",
    "faithfulness_qa",
    "completeness_qa",
    "conciseness_qa",
)


def load_pairs(
    *,
    prompt_template: str,
    splits: tuple[str, ...] = QA_SPLITS,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        ds = load_dataset(DATASET_NAME, split=split, revision=DATASET_REVISION)
        for row in ds:
            records.append(_to_dpo_record(row, split=split, prompt_template=prompt_template))
        print(f"  Loaded {len(ds)} pairs from split '{split}'")
    if limit is not None:
        records = records[:limit]
    return records


def _to_dpo_record(row: dict[str, Any], *, split: str, prompt_template: str) -> dict[str, Any]:
    return {
        "prompt": prompt_template.format(context=row["context"], question=row["question"]),
        "chosen": row["positive_response"],
        "rejected": row["negative_response"],
        "split": split,
        "source": row.get("source", ""),
    }
