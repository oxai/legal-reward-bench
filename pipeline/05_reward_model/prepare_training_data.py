from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.storage import read_jsonl, read_text, write_jsonl
from sources.contextual_judge_bench import ALL_SPLITS, load_pairs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LRB_TRAIN = ROOT / "data" / "lrb_v2" / "pairs_train.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "training" / "cjb_lrb_v2_train_dpo.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
DEFAULT_LRB_MAX_CHARS = 3000


def truncate_prompt(prompt: str, max_chars: int = DEFAULT_LRB_MAX_CHARS) -> str:
    if len(prompt) <= max_chars:
        return prompt
    tail = prompt[-max_chars:]
    newline = tail.find("\n")
    if 0 < newline < 200:
        tail = tail[newline + 1 :]
    header = "[Legal context truncated for length; question follows]\n\n"
    return header + tail


def truncate_lrb_prompts(records: list[dict[str, Any]], *, max_chars: int) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    truncated = 0
    for record in records:
        new_record = dict(record)
        old_prompt = str(new_record["prompt"])
        new_prompt = truncate_prompt(old_prompt, max_chars=max_chars)
        new_record["prompt"] = new_prompt
        if len(new_prompt) < len(old_prompt):
            truncated += 1
        out.append(new_record)
    return out, truncated


def load_cjb_training_records(*, prompt_template: str, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    return load_pairs(prompt_template=prompt_template, splits=splits)


def build_combined_training_records(
    *,
    lrb_train: Path,
    cjb_prompt_template: str,
    cjb_splits: tuple[str, ...],
    lrb_max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    lrb_records = read_jsonl(lrb_train)
    lrb_records, lrb_truncated = truncate_lrb_prompts(lrb_records, max_chars=lrb_max_chars)
    cjb_records = load_cjb_training_records(prompt_template=cjb_prompt_template, splits=cjb_splits)
    records = [*lrb_records, *cjb_records]
    stats = {
        "lrb_records": len(lrb_records),
        "lrb_truncated": lrb_truncated,
        "cjb_records": len(cjb_records),
        "total_records": len(records),
    }
    return records, stats


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(split.strip() for split in value.split(",") if split.strip())
    if not splits:
        raise ValueError("--cjb-splits must contain at least one split")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the combined CJB+LRB-v2 DPO training artifact."
    )
    parser.add_argument("--lrb-train", type=Path, default=DEFAULT_LRB_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cjb-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--cjb-splits", default=",".join(ALL_SPLITS))
    parser.add_argument("--lrb-max-chars", type=int, default=DEFAULT_LRB_MAX_CHARS)
    args = parser.parse_args()

    records, stats = build_combined_training_records(
        lrb_train=args.lrb_train,
        cjb_prompt_template=read_text(args.cjb_prompt),
        cjb_splits=parse_splits(args.cjb_splits),
        lrb_max_chars=args.lrb_max_chars,
    )
    count = write_jsonl(args.output, records)
    print(f"Wrote {count} combined training records to {args.output}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
