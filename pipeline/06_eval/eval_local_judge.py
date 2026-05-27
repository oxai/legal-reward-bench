"""
Evaluate a local HuggingFace model as a pairwise judge on CJB test pairs.

Uses the exact same prompt format as eval_nim_judge.py so results are
directly comparable. For each pair the model sees:

  [System] You are a strict, objective evaluator... Reply A or B.
  [User]   Criterion: ...  Response A: ...  Response B: ...

Preference is determined by comparing log P("A") vs log P("B") as the
next token — no generation needed, fast and deterministic.

By default runs a single pass (no position-bias correction) to match
the --no-swap setting used for the NIM judges.

Usage:
  # Raw base model:
  python pipeline/06_eval/eval_local_judge.py \
      --model meta-llama/Llama-3.1-8B-Instruct --qlora \
      --pairs pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl \
      --output pipeline/06_eval/outputs/sota_eval/ours/llama8b-raw-judge.csv

  # DPO adapter + surgery:
  python pipeline/06_eval/eval_local_judge.py \
      --model pipeline/06_eval/outputs/model_sweep/llama3.1-8b/dpo_adapter \
      --base-model meta-llama/Llama-3.1-8B-Instruct --qlora \
      --spectral-layer 4,15 --spectral-alpha 0.2,0.1 \
      --pairs pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl \
      --output pipeline/06_eval/outputs/sota_eval/ours/llama8b-dpo-l04-l15-judge.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT     = Path(__file__).resolve().parents[2]
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CJB_STAGE))

from eval_nim_judge import JUDGE_SYSTEM, JUDGE_TEMPLATE, SPLIT_CRITERIA  # reuse same prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        required=True, help="HF model id or path to LoRA adapter dir.")
    parser.add_argument("--base-model",   default=None,  help="Base model id when --model is a LoRA adapter.")
    parser.add_argument("--pairs",        type=Path, required=True)
    parser.add_argument("--output",       type=Path, required=True)
    parser.add_argument("--qlora",        action="store_true")
    parser.add_argument("--spectral-layer",  default=None)
    parser.add_argument("--spectral-alpha",  default=None)
    parser.add_argument("--swap",         action="store_true",
                        help="Also run reversed order and average (position-bias correction).")
    parser.add_argument("--no-shuffle",  action="store_true",
                        help="Disable A/B randomisation (chosen always = A). Biased — for debugging only.")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--max-length",   type=int, default=3072)
    args = parser.parse_args()

    model, tokenizer = _load(args)

    records = _load_pairs(args.pairs)
    if args.limit:
        records = records[: args.limit]

    shuffle = not args.no_shuffle
    print(f"Pairs: {len(records)}  swap={args.swap}  shuffle_ab={shuffle}", flush=True)

    split_correct: dict[str, int] = defaultdict(int)
    split_total:   dict[str, int] = defaultdict(int)

    for i, rec in enumerate(records, 1):
        split    = rec.get("split", "overall")
        chosen   = rec["chosen"]
        rejected = rec["rejected"]
        prompt   = rec["prompt"]

        # Alternate A/B assignment so position bias cancels across the dataset.
        # Pair i (1-indexed): even → chosen=A, odd → chosen=B.
        ab_swapped = shuffle and (i % 2 == 0)
        if ab_swapped:
            chosen_for_judge, rejected_for_judge = rejected, chosen
        else:
            chosen_for_judge, rejected_for_judge = chosen, rejected

        pref = _judge(model, tokenizer, prompt, chosen_for_judge, rejected_for_judge, split, args)
        # If ab_swapped, the logprob favours A=rejected; flip the label.
        if pref is not None and ab_swapped:
            pref = not pref

        if pref is None:
            continue

        split_correct[split]     += int(pref)
        split_correct["overall"] += int(pref)
        split_total[split]       += 1
        split_total["overall"]   += 1

        if i % 50 == 0 or i == len(records):
            acc = split_correct["overall"] / split_total["overall"]
            print(f"  [{i}/{len(records)}] overall={acc:.4f}", flush=True)

    results = {
        split: {
            "n": split_total[split],
            "correct": split_correct[split],
            "reward_accuracy": round(split_correct[split] / split_total[split], 4),
        }
        for split in split_total
    }
    _write_csv(args.output, results)
    _print_table(results)
    print(f"\nWrote {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Judge logic
# ---------------------------------------------------------------------------

def _judge(
    model, tokenizer,
    prompt: str, chosen: str, rejected: str,
    split: str, args: argparse.Namespace,
) -> bool | None:
    criterion = SPLIT_CRITERIA.get(split, "overall quality")

    def _logprob_ab(resp_a: str, resp_b: str) -> float | None:
        """Return log P("A") - log P("B") given the judge prompt."""
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
            tokenized = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(tokenized, torch.Tensor):
                input_ids = tokenized.to(model.device)
            else:
                input_ids = tokenized["input_ids"].to(model.device)
        except Exception:
            # Fallback for tokenizers that don't support chat templates
            flat = JUDGE_SYSTEM + "\n\n" + text
            input_ids = tokenizer(flat, return_tensors="pt", truncation=True,
                                  max_length=args.max_length).input_ids.to(model.device)

        if input_ids.shape[1] > args.max_length:
            input_ids = input_ids[:, -args.max_length:]

        tok_a = tokenizer.encode("A", add_special_tokens=False)
        tok_b = tokenizer.encode("B", add_special_tokens=False)
        if not tok_a or not tok_b:
            return None
        id_a, id_b = tok_a[0], tok_b[0]

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]  # last token position
        lp_a = logits[id_a].item()
        lp_b = logits[id_b].item()
        return lp_a - lp_b  # positive → model prefers A

    # Pass 1: chosen=A, rejected=B → positive means chosen is preferred
    delta1 = _logprob_ab(chosen, rejected)
    if delta1 is None:
        return None

    if not args.swap:
        return delta1 > 0

    # Pass 2: rejected=A, chosen=B → negative means chosen is preferred
    delta2 = _logprob_ab(rejected, chosen)
    if delta2 is None:
        return delta1 > 0
    return (delta1 - delta2) > 0  # aggregate signal


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path     = Path(args.model)
    is_adapter = path.exists() and (path / "adapter_config.json").exists()
    base_id  = args.base_model if is_adapter else args.model

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map={"": 0})
    else:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Apply spectral surgery to base weights before attaching LoRA
    if args.spectral_layer:
        sys.path.insert(0, str(ROOT / "pipeline" / "05_reward_model"))
        from model_utils import apply_spectral_surgery_from_args
        apply_spectral_surgery_from_args(base, args)

    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(path))
        print(f"Loaded LoRA adapter from {path}", flush=True)
    else:
        model = base

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pairs(path: Path) -> list[dict[str, Any]]:
    import json
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "n", "correct", "reward_accuracy"])
        writer.writeheader()
        for split, stats in rows:
            writer.writerow({"split": split, **stats})


def _print_table(results: dict) -> None:
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    col_w = max(len(s) for s, _ in rows)
    print(f"\n{'split':<{col_w}}  {'n':>6}  {'reward_accuracy':>16}")
    print("-" * (col_w + 28))
    for split, stats in rows:
        print(f"{split:<{col_w}}  {stats['n']:>6}  {stats['reward_accuracy']:>16.4f}")


if __name__ == "__main__":
    main()
