from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from model_utils import (
    add_shared_args,
    load_model_and_tokenizer,
    load_records,
    make_lora_config,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "05_reward_model"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT warm-up on chosen responses before DPO."
    )
    add_shared_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STAGE_DIR / "outputs" / "sft_model",
    )
    args = parser.parse_args()

    records = load_records(args, default_pairs=DEFAULT_PAIRS, default_prompt=DEFAULT_PROMPT)
    if args.limit is not None:
        records = records[: args.limit]

    sft_records = [{"text": r["prompt"] + r["chosen"]} for r in records]

    n_train = len(sft_records) if args.eval_split == 0.0 else max(1, int(len(sft_records) * (1 - args.eval_split)))
    train_dataset = Dataset.from_list(sft_records[:n_train])
    eval_records = sft_records[n_train:]
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_records)}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, use_qlora=args.qlora)
    lora_config = make_lora_config(r=args.lora_r, alpha=args.lora_alpha)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
