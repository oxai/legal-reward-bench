from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from common.storage import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "lrb"
DEFAULT_SEED = 42


def base_question_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    if metadata.get("base_triple_id"):
        return str(metadata["base_triple_id"])
    triple_id = str(record["triple_id"])
    return triple_id.split("__", 1)[0]


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(base_question_id(record), []).append(record)

    split_by_question = split_from_ratios(
        question_ids=sorted(grouped),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    out = {"train": [], "dev": [], "test": []}
    for question_id in sorted(grouped):
        split = split_by_question[question_id]
        for record in grouped[question_id]:
            out[split].append(mark_split(record, split=split))
    return out

def split_from_ratios(
    *,
    question_ids: list[str],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio must be between 0 and 1")
    test_ratio = 1.0 - train_ratio - dev_ratio
    if test_ratio <= 0:
        raise ValueError("--train-ratio + --dev-ratio must be less than 1")

    shuffled = list(question_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_dev = round(n * dev_ratio)
    train_ids = set(shuffled[:n_train])
    dev_ids = set(shuffled[n_train : n_train + n_dev])

    return {
        question_id: (
            "train" if question_id in train_ids else "dev" if question_id in dev_ids else "test"
        )
        for question_id in question_ids
    }


def mark_split(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    new_record = dict(record)
    metadata = dict(new_record.get("metadata") or {})
    metadata["dataset_split"] = split
    metadata.setdefault("preference_type", new_record.get("split"))
    new_record["metadata"] = metadata
    return new_record

def write_outputs(
    *,
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    write_valtest: bool,
) -> None:
    for split in ("train", "dev", "test"):
        path = output_dir / f"pairs_{split}.jsonl"
        count = write_jsonl(path, splits[split])
        print(f"Wrote {count} {split} pairs to {path}")

    if write_valtest:
        valtest = [*splits["dev"], *splits["test"]]
        path = output_dir / "pairs_valtest.jsonl"
        count = write_jsonl(path, valtest)
        print(f"Wrote {count} dev+test pairs to {path}")


def print_summary(splits: dict[str, list[dict[str, Any]]]) -> None:
    print("\nSplit summary:")
    for split in ("train", "dev", "test"):
        records = splits[split]
        questions = {base_question_id(record) for record in records}
        by_type = Counter(record.get("split", "unknown") for record in records)
        detail = ", ".join(f"{key}={by_type[key]}" for key in sorted(by_type))
        print(f"  {split}: {len(records)} pairs, {len(questions)} questions ({detail})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic local train/dev/test splits at question level."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write-valtest", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    splits = split_records(
        records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    write_outputs(output_dir=args.output_dir, splits=splits, write_valtest=args.write_valtest)
    print_summary(splits)


if __name__ == "__main__":
    main()
