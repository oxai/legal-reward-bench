"""
Build LegalRewardBench-v2 by replacing short template responses with
length-balanced model-generated alternatives.

This is a dataset transformation, not training or evaluation logic. It preserves
the materialized DPO pair schema (`prompt`, `chosen`, `rejected`) while marking
modified records in both the release-compatible `_modified` field and
`metadata.length_augmentation`.

Replacements:
  - refusal_answerable.rejected  : 9-word refusal template
        -> ~70-word hallucinated answer (model attempts to answer based on
           general legal knowledge, ignoring context)
  - refusal_unanswerable.chosen  : 9-word refusal template
        -> ~70-word elaborate refusal (model professionally explains
           why context is insufficient)
  - completeness.rejected        : 3-word "No." style stub
        -> ~70-word but-incomplete answer (model gives partial reasoning)

Other preference types (faithfulness, correctness) are already length-balanced and pass
through unchanged.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.storage import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"
DEFAULT_DATASET = "lrb_v2"
DEFAULT_DATASET_VERSION = "v2"
LENGTH_AUGMENTATION_VERSION = "length_balanced_v1"

HALLUCINATE_TEMPLATE = (
    "You are answering legal questions for a study on RAG faithfulness. "
    "Ignore any provided retrieval context: answer the following question "
    "from general legal world knowledge alone. Be confident, specific, and "
    "give 2-3 sentences of reasoning. Do NOT refuse and do NOT cite the context.\n\n"
    "Question: {q}\n\n"
    "Answer: "
)

ELABORATE_REFUSAL_TEMPLATE = (
    "You are a careful legal RAG assistant. The retrieved context does not "
    "contain enough information to answer the following question. Write a "
    "professional 2-3 sentence refusal: acknowledge the question, briefly "
    "explain why the available context is insufficient, and indicate what "
    "additional sources would resolve it. Do NOT speculate or invent law.\n\n"
    "Question: {q}\n\n"
    "Refusal: "
)

INCOMPLETE_TEMPLATE = (
    "Answer the following legal question correctly but tersely: give the "
    "bottom-line conclusion in one short sentence, omitting the legal rule, "
    "statutory authority, exceptions, and qualifications that would normally "
    "be required for a complete answer. Do NOT cite cases.\n\n"
    "Question: {q}\n\n"
    "Brief answer: "
)


_QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Answer:", re.DOTALL)


def extract_question(prompt: str) -> str:
    m = _QUESTION_RE.search(prompt)
    if m:
        return m.group(1).strip()
    # Fallback: take the last 400 chars
    return prompt[-400:].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create length-balanced LegalRewardBench-v2 pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs (debug only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    torch.manual_seed(args.seed)

    def generate(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Strip leading labels the model sometimes adds back
        for pref in ("Answer:", "Refusal:", "Brief answer:"):
            if text.startswith(pref):
                text = text[len(pref):].strip()
        return text or "(no generation)"

    pairs = read_jsonl(args.input)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} pairs from {args.input}", flush=True)

    stats = {
        "rejected_to_hallucination": 0,
        "chosen_to_elaborate_refusal": 0,
        "rejected_to_incomplete": 0,
        "passthrough": 0,
    }
    out_records = []
    for i, r in enumerate(pairs):
        split = r.get("split", "")
        q = extract_question(r["prompt"])
        new_r = dict(r)
        modification = "passthrough"
        if split == "refusal_answerable" and len(r["rejected"]) < 120:
            new_r["rejected"] = generate(HALLUCINATE_TEMPLATE.format(q=q))
            modification = "rejected_to_hallucination"
            stats[modification] += 1
        elif split == "refusal_unanswerable" and len(r["chosen"]) < 120:
            new_r["chosen"] = generate(ELABORATE_REFUSAL_TEMPLATE.format(q=q))
            modification = "chosen_to_elaborate_refusal"
            stats[modification] += 1
        elif split == "completeness" and len(r["rejected"]) < 30:
            new_r["rejected"] = generate(INCOMPLETE_TEMPLATE.format(q=q))
            modification = "rejected_to_incomplete"
            stats[modification] += 1
        else:
            stats["passthrough"] += 1
        mark_augmentation(
            new_r,
            modification=modification,
            dataset=args.dataset,
            dataset_version=args.dataset_version,
            max_new_tokens=args.max_new_tokens,
        )
        out_records.append(new_r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)}  {stats}", flush=True)

    count = write_jsonl(args.output, out_records)
    print(f"Wrote {count} pairs to {args.output}")
    print(f"Final stats: {stats}")


def mark_augmentation(
    record: dict[str, Any],
    *,
    modification: str,
    dataset: str,
    dataset_version: str,
    max_new_tokens: int,
) -> None:
    if modification == "passthrough":
        record.pop("_modified", None)
    else:
        record["_modified"] = modification

    metadata = dict(record.get("metadata") or {})
    metadata["dataset"] = dataset
    metadata["dataset_version"] = dataset_version
    metadata["length_augmentation"] = {
        "version": LENGTH_AUGMENTATION_VERSION,
        "modified": modification != "passthrough",
        "modification": modification,
        "model": MODEL_ID,
        "max_new_tokens": max_new_tokens,
    }
    record["metadata"] = metadata


if __name__ == "__main__":
    main()
