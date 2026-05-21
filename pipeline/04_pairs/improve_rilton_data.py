"""
Improve Rilton training data by replacing 9-word template responses with
length-balanced model-generated alternatives, eliminating the length-asymmetry
artefact that swamped the original sum-log-probability metric.

Replacements:
  - refusal_answerable.rejected  : 9-word refusal template
        -> ~70-word hallucinated answer (model attempts to answer based on
           general legal knowledge, ignoring context)
  - refusal_unanswerable.chosen  : 9-word refusal template
        -> ~70-word elaborate refusal (model professionally explains
           why context is insufficient)
  - completeness.rejected        : 3-word "No." style stub
        -> ~70-word but-incomplete answer (model gives partial reasoning)

Other splits (faithfulness, correctness) are already length-balanced and pass
through unchanged.

Output: data/rilton/pairs_train_v2.jsonl  (940 pairs, same length as input)
        data/rilton/pairs_dev_v2.jsonl    (124 pairs)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs (debug only)")
    parser.add_argument("--seed", type=int, default=42)
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

    pairs = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} pairs from {args.input}", flush=True)

    stats = {"refusal_answerable_rej": 0, "refusal_unanswerable_cho": 0,
             "completeness_rej": 0, "passthrough": 0}
    out_records = []
    for i, r in enumerate(pairs):
        split = r.get("split", "")
        q = extract_question(r["prompt"])
        new_r = dict(r)
        if split == "refusal_answerable" and len(r["rejected"]) < 120:
            new_r["rejected"] = generate(HALLUCINATE_TEMPLATE.format(q=q))
            new_r["_modified"] = "rejected_to_hallucination"
            stats["refusal_answerable_rej"] += 1
        elif split == "refusal_unanswerable" and len(r["chosen"]) < 120:
            new_r["chosen"] = generate(ELABORATE_REFUSAL_TEMPLATE.format(q=q))
            new_r["_modified"] = "chosen_to_elaborate_refusal"
            stats["refusal_unanswerable_cho"] += 1
        elif split == "completeness" and len(r["rejected"]) < 30:
            new_r["rejected"] = generate(INCOMPLETE_TEMPLATE.format(q=q))
            new_r["_modified"] = "rejected_to_incomplete"
            stats["completeness_rej"] += 1
        else:
            stats["passthrough"] += 1
        out_records.append(new_r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)}  {stats}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(out_records)} pairs to {args.output}")
    print(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
