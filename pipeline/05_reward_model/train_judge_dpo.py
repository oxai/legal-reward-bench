"""
Judge-DPO: directly optimize the log-ratio P("A")/P("B") with the DPO objective.

This is strictly more principled than judge-SFT because:
  - judge-SFT pushes P(correct_label) up via cross-entropy, no constraint on P(wrong_label)
  - judge-DPO optimizes log P_theta("A")/P_ref("A") - log P_theta("B")/P_ref("B"),
    which is EXACTLY the quantity measured at eval time, with a KL penalty via beta
    that prevents the model from collapsing to P("A")=1 always.

Reference model: base model without LoRA (via PEFT disable_adapter context manager).
No second model load needed — works within QLoRA memory budget.

Usage:
  python pipeline/05_reward_model/train_judge_dpo.py \
      --base-model meta-llama/Llama-3.1-8B-Instruct --qlora \
      --pairs pipeline/06_eval/outputs/model_sweep/cjb_train.jsonl \
      --output-dir pipeline/06_eval/outputs/model_sweep/llama3.1-8b/judge_dpo_adapter
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

ROOT      = Path(__file__).resolve().parents[2]
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CJB_STAGE))

from model_utils import apply_spectral_surgery_from_args, load_model_and_tokenizer, make_lora_config

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

class JudgeDPODataset(TorchDataset):
    def __init__(self, records, tokenizer, max_length: int = 3072, seed: int = 42):
        self.examples = []
        rng = random.Random(seed)

        id_a = tokenizer.encode("A", add_special_tokens=False)
        id_b = tokenizer.encode("B", add_special_tokens=False)
        if not id_a or not id_b:
            raise RuntimeError("Tokenizer cannot encode 'A' or 'B' as a single token.")
        self.tok_a, self.tok_b = id_a[0], id_b[0]

        skipped = 0
        for rec in records:
            split     = rec.get("split", "overall")
            chosen    = rec["chosen"]
            rejected  = rec["rejected"]
            prompt    = rec["prompt"]
            criterion = SPLIT_CRITERIA.get(split, "overall quality")

            a_is_chosen = rng.random() < 0.5
            resp_a = chosen   if a_is_chosen else rejected
            resp_b = rejected if a_is_chosen else chosen
            correct_tok   = self.tok_a if a_is_chosen else self.tok_b
            incorrect_tok = self.tok_b if a_is_chosen else self.tok_a

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
                raw = tokenizer(flat, add_special_tokens=True)["input_ids"]
                prefix_ids = raw.tolist() if hasattr(raw, "tolist") else list(raw)

            if len(prefix_ids) >= max_length:
                prefix_ids = prefix_ids[-(max_length - 1):]

            if len(prefix_ids) + 1 > max_length:
                skipped += 1
                continue

            self.examples.append({
                "input_ids":    prefix_ids,
                "correct_tok":  correct_tok,
                "incorrect_tok": incorrect_tok,
            })

        print(f"Dataset: {len(self.examples)} examples  ({skipped} skipped)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class JudgeDPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, masks, correct_toks, incorrect_toks = [], [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append([self.pad_id] * pad + f["input_ids"])
            masks.append(    [0]          * pad + [1] * len(f["input_ids"]))
            correct_toks.append(f["correct_tok"])
            incorrect_toks.append(f["incorrect_tok"])
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(masks,          dtype=torch.long),
            "correct_tok":    torch.tensor(correct_toks,   dtype=torch.long),
            "incorrect_tok":  torch.tensor(incorrect_toks, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    correct_tok: torch.Tensor,
    incorrect_tok: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    device = input_ids.device

    # Policy forward pass
    policy_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
    policy_lp = F.log_softmax(policy_logits, dim=-1)
    lp_correct_policy   = policy_lp.gather(1, correct_tok.unsqueeze(1)).squeeze(1)
    lp_incorrect_policy = policy_lp.gather(1, incorrect_tok.unsqueeze(1)).squeeze(1)

    # Reference forward pass (base model = disable LoRA adapter)
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        else:
            ref_logits = policy_logits.detach()
    ref_lp = F.log_softmax(ref_logits, dim=-1)
    lp_correct_ref   = ref_lp.gather(1, correct_tok.unsqueeze(1)).squeeze(1)
    lp_incorrect_ref = ref_lp.gather(1, incorrect_tok.unsqueeze(1)).squeeze(1)

    # DPO loss
    chosen_ratio   = lp_correct_policy   - lp_correct_ref
    rejected_ratio = lp_incorrect_policy - lp_incorrect_ref
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()

    # Accuracy: fraction where policy prefers correct token over incorrect
    with torch.no_grad():
        correct_lp   = policy_lp.gather(1, correct_tok.unsqueeze(1)).squeeze(1)
        incorrect_lp = policy_lp.gather(1, incorrect_tok.unsqueeze(1)).squeeze(1)
        acc = (correct_lp > incorrect_lp).float().mean().item()

    return loss, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",    default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pairs",         type=Path, required=True)
    parser.add_argument("--output-dir",    type=Path, required=True)
    parser.add_argument("--qlora",         action="store_true")
    parser.add_argument("--spectral-layer",   default=None)
    parser.add_argument("--spectral-alpha",   default=None)
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--grad-accum",    type=int,   default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1)
    parser.add_argument("--beta",          type=float, default=0.1)
    parser.add_argument("--lora-r",        type=int,   default=16)
    parser.add_argument("--lora-alpha",    type=int,   default=32)
    parser.add_argument("--max-length",    type=int,   default=3072)
    parser.add_argument("--limit",         type=int,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
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

    dataset  = JudgeDPODataset(records, tokenizer, max_length=args.max_length, seed=args.seed)
    collator = JudgeDPOCollator(pad_token_id=tokenizer.pad_token_id)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collator, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    total_steps     = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
            loss, acc = dpo_loss(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["correct_tok"],
                batch["incorrect_tok"],
                beta=args.beta,
            )
            (loss / args.grad_accum).backward()
            epoch_loss += loss.item()
            epoch_acc  += acc
            n += 1

            if step % args.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    print(f"  step {global_step}  loss={epoch_loss/n:.4f}  train_acc={epoch_acc/n:.4f}", flush=True)

        print(f"Epoch {epoch}: loss={epoch_loss/n:.4f}  train_acc={epoch_acc/n:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved judge-DPO adapter to {args.output_dir}")


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
