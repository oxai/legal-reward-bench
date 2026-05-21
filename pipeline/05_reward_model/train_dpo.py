from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from model_utils import (
    add_shared_args,
    apply_spectral_surgery_from_args,
    load_model_and_tokenizer,
    load_records,
    make_lora_config,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "05_reward_model"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tuning on legal response preference pairs.")
    add_shared_args(parser)
    parser.add_argument(
        "--sft-model",
        type=Path,
        default=None,
        help="Path to a saved SFT adapter to warm-start DPO from (merged into base before training).",
    )
    parser.add_argument("--output-dir", type=Path, default=STAGE_DIR / "outputs" / "dpo_model")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    records = load_records(args, default_pairs=DEFAULT_PAIRS, default_prompt=DEFAULT_PROMPT)
    if args.limit is not None:
        records = records[: args.limit]

    n_train = len(records) if args.eval_split == 0.0 else max(1, int(len(records) * (1 - args.eval_split)))
    train_dataset = Dataset.from_list(records[:n_train])
    eval_records = records[n_train:]
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_records)}")

    model, tokenizer = _load_model(args)
    apply_spectral_surgery_from_args(model, args)
    lora_config = make_lora_config(r=args.lora_r, alpha=args.lora_alpha)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved DPO model to {args.output_dir}")


def _load_model(args: argparse.Namespace):
    fa2 = getattr(args, "flash_attn2", False)
    if args.sft_model is not None:
        if args.qlora:
            # Quantized models can't be merged — DPO starts from base with a fresh LoRA
            print("Note: --sft-model is skipped with --qlora (adapters cannot be merged into quantized weights).")
            return load_model_and_tokenizer(args.base_model, use_qlora=True, use_flash_attn2=fa2)

        sft_path = args.sft_model
        is_adapter = (sft_path / "adapter_config.json").exists()
        if is_adapter:
            from peft import PeftModel
            base, tokenizer = load_model_and_tokenizer(args.base_model, use_flash_attn2=fa2)
            print(f"Loading SFT adapter from {sft_path} and merging into base...")
            model = PeftModel.from_pretrained(base, str(sft_path))
            model = model.merge_and_unload()
        else:
            model, tokenizer = load_model_and_tokenizer(str(sft_path), use_flash_attn2=fa2)
        return model, tokenizer

    return load_model_and_tokenizer(args.base_model, use_qlora=args.qlora, use_flash_attn2=fa2)


if __name__ == "__main__":
    main()
