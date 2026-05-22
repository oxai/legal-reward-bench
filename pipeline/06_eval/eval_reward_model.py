"""
Evaluate a scalar reward model (AutoModelForSequenceClassification) on CJB.

reward_accuracy = fraction of pairs where reward(chosen) > reward(rejected).

Usage:
  python pipeline/06_eval/eval_reward_model.py \
      --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
      --source contextual_judge_bench \
      --output pipeline/06_eval/outputs/model_sweep/skywork-8b/judge.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "06_eval"
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pairs-output", type=Path, default=None, metavar="PATH",
                        help="JSONL file to write per-pair results {idx, split, correct}.")
    parser.add_argument("--qlora", action="store_true", help="Load in 4-bit NF4 for consistency with causal LM evals.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM: {(total-free)/1024**3:.1f} GB used / {total/1024**3:.1f} GB total", flush=True)

    print(f"Loading reward model: {args.model}  (qlora={args.qlora})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()

    records = load_records(args)
    if args.limit is not None:
        records = records[:args.limit]

    print(f"Evaluating {len(records)} pairs...", flush=True)
    results, pair_records = evaluate(model, tokenizer, records, max_length=args.max_length)

    print_table(results)

    output = args.output or (
        STAGE_DIR / "outputs" / f"reward_model__{args.model.replace('/', '_')}.csv"
    )
    write_csv(output, results)
    print(f"Wrote results to {output}")

    if args.pairs_output is not None:
        write_pairs_jsonl(args.pairs_output, pair_records)
        print(f"Wrote per-pair results to {args.pairs_output}")


def score_response(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> float:
    inputs = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(model.device)
    with torch.no_grad():
        return model(**inputs).logits[0, 0].item()


def evaluate(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    max_length: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    pair_records: list[dict[str, Any]] = []
    total = len(records)

    for i, record in enumerate(records, 1):
        split = record.get("split", "all")
        score_chosen = score_response(model, tokenizer, record["prompt"], record["chosen"], max_length)
        score_rejected = score_response(model, tokenizer, record["prompt"], record["rejected"], max_length)
        correct = int(score_chosen > score_rejected)

        split_correct[split] += correct
        split_correct["overall"] += correct
        split_total[split] += 1
        split_total["overall"] += 1
        pair_records.append({"idx": i - 1, "split": split, "correct": correct})

        if i % 50 == 0 or i == total:
            acc = split_correct["overall"] / split_total["overall"]
            print(f"[{i}/{total}] overall reward_accuracy={acc:.3f}", flush=True)

    agg = {
        split: {
            "n": split_total[split],
            "correct": split_correct[split],
            "reward_accuracy": round(split_correct[split] / split_total[split], 4),
        }
        for split in split_total
    }
    return agg, pair_records


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
        sys.path.insert(0, str(CJB_STAGE))
        sys.path.insert(0, str(ROOT))
        from sources.contextual_judge_bench import QA_SPLITS, load_pairs
        splits = (
            tuple(s.strip() for s in args.splits.split(",") if s.strip())
            if args.splits else QA_SPLITS
        )
        prompt_template = DEFAULT_PROMPT.read_text(encoding="utf-8")
        return load_pairs(prompt_template=prompt_template, splits=splits)

    sys.path.insert(0, str(ROOT))
    from common.storage import read_jsonl
    pairs_path = args.pairs or DEFAULT_PAIRS
    return [
        {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "split": r.get("split") or r.get("metadata", {}).get("context_variant", "pipeline"),
        }
        for r in read_jsonl(pairs_path)
        if "prompt" in r
    ]


def print_table(results: dict[str, dict[str, Any]]) -> None:
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    col_w = max(len(s) for s, _ in rows)
    print(f"\n{'split':<{col_w}}  {'n':>6}  {'reward_accuracy':>16}")
    print("-" * (col_w + 28))
    for split, stats in rows:
        print(f"{split:<{col_w}}  {stats['n']:>6}  {stats['reward_accuracy']:>16.4f}")


def write_pairs_jsonl(path: Path, pair_records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in pair_records:
            f.write(json.dumps(record) + "\n")


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "n", "correct", "reward_accuracy"])
        writer.writeheader()
        for split, stats in rows:
            writer.writerow({"split": split, **stats})


if __name__ == "__main__":
    main()
