"""
Judge-specific SFT: train the model to predict 'A' or 'B' given a pairwise judge prompt.

Loss is computed ONLY on the single label token ('A' or 'B'), so the model directly
learns the discriminative judge task rather than learning to generate better responses.

This is what should have been trained instead of response-generation DPO:
  - Response-gen DPO: shifts P(chosen response | prompt) -- wrong objective for judging
  - Judge SFT       : shifts P("A" | judge_prompt) vs P("B" | ...) -- exact objective

A/B assignment is randomised 50/50 per example so position bias never enters training.

Usage:
  python pipeline/05_reward_model/train_judge_sft.py \
      --base-model meta-llama/Llama-3.1-8B-Instruct --qlora \
      --pairs pipeline/06_eval/outputs/model_sweep/cjb_train.jsonl \
      --output-dir pipeline/06_eval/outputs/model_sweep/llama3.1-8b/judge_sft_adapter

  # With spectral surgery before LoRA attachment:
  python pipeline/05_reward_model/train_judge_sft.py \
      --base-model meta-llama/Llama-3.1-8B-Instruct --qlora \
      --spectral-layer 4,15 --spectral-alpha 0.2,0.1 \
      --pairs pipeline/06_eval/outputs/model_sweep/cjb_train.jsonl \
      --output-dir pipeline/06_eval/outputs/model_sweep/llama3.1-8b/judge_sft_surgery_adapter
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

ROOT      = Path(__file__).resolve().parents[2]
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CJB_STAGE))

from model_utils import apply_spectral_surgery_from_args, load_model_and_tokenizer, make_lora_config

# Reuse the exact same prompt format as eval so training and eval are aligned.
JUDGE_SYSTEM = (
    "You are a strict, objective evaluator. "
    "You will be given an instruction and two responses. "
    "Your sole task is to decide which response better satisfies the given criterion. "
    "Reply with exactly one word: A or B."
)

JUDGE_TEMPLATE = """\
Criterion: {criterion}

Instruction / Context:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better according to the criterion above? Reply with only "A" or "B"."""

SPLIT_CRITERIA: dict[str, str] = {
    "completeness_qa":      "completeness -- the answer covers all key points from the context",
    "conciseness_qa":       "conciseness -- the answer is direct and avoids unnecessary verbosity",
    "faithfulness_qa":      "faithfulness -- the answer is factually grounded in the context without hallucination",
    "refusal_answerable":   "appropriateness -- answering when the question is answerable (not refusing unnecessarily)",
    "refusal_unanswerable": "appropriateness -- refusing to speculate when the question cannot be answered from context",
    "completeness_summ":    "completeness -- the summary covers all key points from the source",
    "conciseness_summ":     "conciseness -- the summary is brief and avoids unnecessary detail",
    "faithfulness_summ":    "faithfulness -- the summary is factually grounded in the source without hallucination",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JudgeTokenDataset(TorchDataset):
    """Each example: full judge prompt tokenised, label = single 'A' or 'B' token."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_length: int = 3072,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.examples: list[dict] = []
        rng = random.Random(seed)

        id_a = tokenizer.encode("A", add_special_tokens=False)
        id_b = tokenizer.encode("B", add_special_tokens=False)
        if not id_a or not id_b:
            raise RuntimeError("Tokenizer can't encode 'A' or 'B' as a single token.")
        self.tok_a, self.tok_b = id_a[0], id_b[0]

        skipped = 0
        for rec in records:
            split     = rec.get("split", "overall")
            chosen    = rec["chosen"]
            rejected  = rec["rejected"]
            prompt    = rec["prompt"]
            criterion = SPLIT_CRITERIA.get(split, "overall quality")

            # With augment=True, add both orderings (chosen=A and chosen=B).
            # This doubles training data and eliminates any position-bias leakage
            # from the training objective.
            orderings = [(True,), (False,)] if augment else [(rng.random() < 0.5,)]
            for (a_is_chosen,) in orderings:
                resp_a = chosen   if a_is_chosen else rejected
                resp_b = rejected if a_is_chosen else chosen
                label_tok = self.tok_a if a_is_chosen else self.tok_b

                text = JUDGE_TEMPLATE.format(
                    criterion=criterion,
                    prompt=prompt[:2000],
                    response_a=resp_a[:1500],
                    response_b=resp_b[:1500],
                )
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": text},
                ]
                try:
                    result = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                    )
                    if isinstance(result, list):
                        prefix_ids = result
                    else:
                        ids = result["input_ids"]
                        prefix_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                except Exception:
                    flat = JUDGE_SYSTEM + "\n\n" + text
                    prefix_ids = tokenizer(flat, add_special_tokens=True)["input_ids"]
                    if hasattr(prefix_ids, "tolist"):
                        prefix_ids = prefix_ids.tolist()

                if len(prefix_ids) >= max_length:
                    prefix_ids = prefix_ids[-(max_length - 1):]

                input_ids = prefix_ids + [label_tok]
                if len(input_ids) > max_length:
                    skipped += 1
                    continue

                # Loss only on the label token.
                labels = [-100] * len(prefix_ids) + [label_tok]
                self.examples.append({
                    "input_ids":      input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels":         labels,
                })

        print(f"Dataset: {len(self.examples)} examples  ({skipped} skipped / too long)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class JudgeCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, masks = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            # Left-pad so the label token stays at the right edge.
            input_ids.append([self.pad_token_id] * pad + f["input_ids"])
            labels.append(   [-100]               * pad + f["labels"])
            masks.append(    [0]                  * pad + f["attention_mask"])
        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
            "attention_mask": torch.tensor(masks,      dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",   default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pairs",        type=Path, required=True)
    parser.add_argument("--output-dir",   type=Path, required=True)
    parser.add_argument("--qlora",        action="store_true")
    parser.add_argument("--spectral-layer",  default=None)
    parser.add_argument("--spectral-alpha",  default=None)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=2)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--max-length",   type=int,   default=3072)
    parser.add_argument("--limit",        type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-augment",   action="store_true",
                        help="Disable both-ordering augmentation (default: augment).")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    records = _load_jsonl(args.pairs)
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} training pairs from {args.pairs}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, use_qlora=args.qlora)
    apply_spectral_surgery_from_args(model, args)

    from peft import get_peft_model
    lora_cfg = make_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = JudgeTokenDataset(records, tokenizer, max_length=args.max_length,
                                seed=args.seed, augment=not args.no_augment)
    collator = JudgeCollator(pad_token_id=tokenizer.pad_token_id)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved judge-SFT adapter to {args.output_dir}")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
