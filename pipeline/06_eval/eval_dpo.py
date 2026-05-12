from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.storage import read_jsonl, read_text

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "06_eval"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPO alignment via reward accuracy on preference pairs.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or path to saved model/adapter.")
    parser.add_argument("--base-model", default=None, help="Base model id (required when --model is a LoRA adapter).")
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated ContextualJudgeBench splits.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.base_model)
    records = load_records(args)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Evaluating {len(records)} pairs...")
    results = evaluate(model, tokenizer, records, max_length=args.max_length)

    print_table(results)

    output = args.output or default_output_path(args.model)
    write_csv(output, results)
    print(f"Wrote results to {output}")


def load_model(
    model_path: str, base_model: str | None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    path = Path(model_path)
    is_adapter = path.exists() and (path / "adapter_config.json").exists()

    if is_adapter:
        if base_model is None:
            raise ValueError("--base-model is required when --model is a LoRA adapter directory.")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
        sys.path.insert(0, str(CJB_STAGE))
        from sources.contextual_judge_bench import QA_SPLITS, load_pairs

        splits = (
            tuple(s.strip() for s in args.splits.split(",") if s.strip())
            if args.splits
            else QA_SPLITS
        )
        prompt_template = read_text(DEFAULT_PROMPT)
        return load_pairs(prompt_template=prompt_template, splits=splits)

    pairs_path = args.pairs or DEFAULT_PAIRS
    return [
        {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "split": r.get("metadata", {}).get("context_variant", "pipeline"),
        }
        for r in read_jsonl(pairs_path)
    ]


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    max_length: int,
) -> dict[str, dict[str, Any]]:
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)

    total = len(records)
    for index, record in enumerate(records, start=1):
        split = record.get("split", "all")
        log_p_chosen = sequence_log_prob(model, tokenizer, record["prompt"], record["chosen"], max_length)
        log_p_rejected = sequence_log_prob(model, tokenizer, record["prompt"], record["rejected"], max_length)
        correct = log_p_chosen > log_p_rejected
        split_correct[split] += int(correct)
        split_correct["overall"] += int(correct)
        split_total[split] += 1
        split_total["overall"] += 1
        if index % 50 == 0 or index == total:
            overall_acc = split_correct["overall"] / split_total["overall"]
            print(f"[{index}/{total}] overall reward_accuracy={overall_acc:.3f}", flush=True)

    return {
        split: {
            "n": split_total[split],
            "correct": split_correct[split],
            "reward_accuracy": round(split_correct[split] / split_total[split], 4),
        }
        for split in split_total
    }


def sequence_log_prob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]

    max_completion = max_length - prompt_ids.shape[1]
    if max_completion <= 0:
        return 0.0
    completion_ids = completion_ids[:, :max_completion]

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    prompt_len = prompt_ids.shape[1]
    response_logits = logits[0, prompt_len - 1 : -1, :]
    response_labels = input_ids[0, prompt_len:]

    if response_labels.shape[0] == 0:
        return 0.0

    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    return log_probs[torch.arange(len(response_labels)), response_labels].sum().item()


def print_table(results: dict[str, dict[str, Any]]) -> None:
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    col_w = max(len(split) for split, _ in rows)
    print(f"\n{'split':<{col_w}}  {'n':>6}  {'reward_accuracy':>16}")
    print("-" * (col_w + 28))
    for split, stats in rows:
        print(f"{split:<{col_w}}  {stats['n']:>6}  {stats['reward_accuracy']:>16.4f}")
    print()


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "n", "correct", "reward_accuracy"])
        writer.writeheader()
        for split, stats in rows:
            writer.writerow({"split": split, **stats})


def default_output_path(model: str) -> Path:
    slug = model.replace("/", "_").replace("\\", "_").strip("_")
    return STAGE_DIR / "outputs" / f"reward_accuracy__{slug}.csv"


if __name__ == "__main__":
    main()
