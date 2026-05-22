"""
Evaluate reward accuracy on Bar Exam QA (reglab/barexam_qa).

Each of the 1,195 Multistate Bar Exam questions includes a gold retrieved passage
(caselaw / Cornell Law encyclopedia).  We construct preference pairs:
  prompt   = "Legal context:\n{gold_passage}\n\nQuestion:\n{question}\n\nAnswer: "
  chosen   = correct answer text (from choice_a/b/c/d matching field `answer`)
  rejected = randomly sampled wrong answer text (seed=42)

Reward accuracy = fraction of pairs where logp(chosen | prompt) > logp(rejected | prompt).
Length-normalised log-prob is NOT applied here; answers are short and comparable in length.

Usage:
  python pipeline/06_eval/eval_barexam.py --model <hf_id_or_adapter_path> --output out.csv
  python pipeline/06_eval/eval_barexam.py --model <adapter> --base-model <hf_id> --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

_CHOICE_FIELDS = ["choice_a", "choice_b", "choice_c", "choice_d"]
_ANSWER_MAP = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}


def build_barexam_pairs(*, seed: int = 42, limit: int | None = None) -> list[dict]:
    import csv
    import io
    from huggingface_hub import hf_hub_download

    # Load raw CSV directly — the custom dataset script is unsupported on newer datasets versions
    path = hf_hub_download(repo_id="reglab/barexam_qa", filename="data/qa/test.csv", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)
    pairs = []
    for row in rows:
        answer_key = str(row["answer"]).strip().upper()
        if answer_key not in _ANSWER_MAP:
            continue
        correct_field = _ANSWER_MAP[answer_key]
        chosen_text = str(row[correct_field]).strip()
        wrong_texts = [str(row[f]).strip() for f in _CHOICE_FIELDS if f != correct_field]
        wrong_texts = [t for t in wrong_texts if t]
        if not wrong_texts or not chosen_text:
            continue
        rejected_text = rng.choice(wrong_texts)
        gold_passage = str(row.get("gold_passage") or "").strip()
        fact_pattern = str(row.get("prompt") or "").strip()
        question = str(row.get("question") or "").strip()
        if gold_passage:
            prompt = f"Legal context:\n{gold_passage}\n\nFacts:\n{fact_pattern}\n\nQuestion:\n{question}\n\nAnswer: "
        else:
            prompt = f"Facts:\n{fact_pattern}\n\nQuestion:\n{question}\n\nAnswer: "
        pairs.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "split": "bar_exam",
        })
    if limit:
        rng.shuffle(pairs)
        pairs = pairs[:limit]
    return pairs


def _load_model(model_path: str, base_model: str | None, *, use_qlora: bool) -> tuple:
    path = Path(model_path)
    is_adapter = path.exists() and (path / "adapter_config.json").exists()
    base_id = base_model if is_adapter else model_path

    attn_impl = "sdpa"
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map="auto",
            attn_implementation=attn_impl,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation=attn_impl,
        )

    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = base
        tokenizer = AutoTokenizer.from_pretrained(base_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def sequence_log_prob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> float | None:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    max_comp = max_length - prompt_ids.shape[1]
    if max_comp <= 0:
        return None
    completion_ids = completion_ids[:, :max_comp]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    prompt_len = prompt_ids.shape[1]
    response_logits = logits[0, prompt_len - 1: -1, :].clone()
    del logits
    labels = input_ids[0, prompt_len:]
    if labels.shape[0] == 0:
        return 0.0
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    return log_probs[torch.arange(len(labels)), labels].sum().item()


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[dict],
    max_length: int,
) -> tuple[dict[str, dict], list[dict]]:
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    pair_records: list[dict] = []
    skipped = 0
    total = len(pairs)

    for idx, pair in enumerate(pairs, 1):
        lp_c = sequence_log_prob(model, tokenizer, pair["prompt"], pair["chosen"], max_length)
        torch.cuda.empty_cache()
        lp_r = sequence_log_prob(model, tokenizer, pair["prompt"], pair["rejected"], max_length)
        torch.cuda.empty_cache()
        if lp_c is None or lp_r is None:
            skipped += 1
            continue
        correct = int(lp_c > lp_r)
        split = pair.get("split", "overall")
        split_correct[split] += correct
        split_correct["overall"] += correct
        split_total[split] += 1
        split_total["overall"] += 1
        pair_records.append({"idx": idx - 1, "split": split, "correct": correct})
        if idx % 100 == 0 or idx == total:
            n = split_total["overall"]
            acc = split_correct["overall"] / n if n else 0.0
            print(f"[{idx}/{total}] scored={n} skipped={skipped} overall={acc:.3f}", flush=True)

    agg = {
        split: {
            "n": split_total[split],
            "correct": split_correct[split],
            "reward_accuracy": round(split_correct[split] / split_total[split], 4),
        }
        for split in split_total
    }
    return agg, pair_records


def write_pairs_jsonl(path: Path, pair_records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in pair_records:
            f.write(json.dumps(record) + "\n")


def write_csv(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.items(), key=lambda x: (x[0] != "overall", x[0]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "n", "correct", "reward_accuracy"])
        w.writeheader()
        for split, stats in rows:
            w.writerow({"split": split, **stats})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, default=None, metavar="PATH",
                        help="JSONL file to write per-pair results {idx, split, correct}.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = build_barexam_pairs(limit=args.limit)
    print(f"Loaded {len(pairs)} Bar Exam QA pairs", flush=True)

    model, tokenizer = _load_model(args.model, args.base_model, use_qlora=args.qlora)
    results, pair_records = evaluate(model, tokenizer, pairs, args.max_length)

    write_csv(args.output, results)
    overall = results.get("overall", {})
    print(
        f"\nBar Exam QA  n={overall.get('n')}  "
        f"reward_accuracy={overall.get('reward_accuracy')}",
        flush=True,
    )

    if args.pairs_output is not None:
        write_pairs_jsonl(args.pairs_output, pair_records)
        print(f"Wrote per-pair results to {args.pairs_output}")


if __name__ == "__main__":
    main()
