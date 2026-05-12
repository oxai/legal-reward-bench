from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common.storage import read_jsonl, read_text

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "05_reward_model"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tuning on legal response preference pairs.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=STAGE_DIR / "outputs" / "dpo_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated ContextualJudgeBench splits (default: all QA splits).",
    )
    args = parser.parse_args()

    records = _load_records(args)
    if args.limit is not None:
        records = records[: args.limit]

    split = max(1, int(len(records) * (1 - args.eval_split)))
    train_dataset = Dataset.from_list(records[:split])
    eval_records = records[split:]
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved DPO model to {args.output_dir}")


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
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
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in read_jsonl(pairs_path)
    ]


if __name__ == "__main__":
    main()
