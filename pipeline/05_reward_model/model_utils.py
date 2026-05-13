from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated ContextualJudgeBench splits (default: all QA splits).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Load base model in 4-bit NF4 (QLoRA). Reduces VRAM from ~2GB/B to ~0.5GB/B.",
    )


def load_model_and_tokenizer(
    model_path: str,
    *,
    use_qlora: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    return model, tokenizer


def make_lora_config(*, r: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_records(args: argparse.Namespace, *, default_pairs: Path, default_prompt: Path) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
        from sources.contextual_judge_bench import QA_SPLITS, load_pairs
        from common.storage import read_text

        splits = (
            tuple(s.strip() for s in args.splits.split(",") if s.strip())
            if args.splits
            else QA_SPLITS
        )
        prompt_template = read_text(default_prompt)
        return load_pairs(prompt_template=prompt_template, splits=splits)

    pairs_path = args.pairs or default_pairs
    from common.storage import read_jsonl
    return [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in read_jsonl(pairs_path)
    ]
