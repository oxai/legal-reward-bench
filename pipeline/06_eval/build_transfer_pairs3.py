"""Batch 3 of transferability benchmarks — more LegalBench tasks including
MCQ and additional contract NLI variations."""
from __future__ import annotations

import json
import random
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def save_jsonl(pairs, path):
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name:60s} n={len(pairs)}")


def build_lb_binary(task, label_to_text, n_max=400):
    rng = random.Random(SEED)
    try:
        ds = load_dataset("nguha/legalbench", task, split="test")
    except Exception as e:
        print(f"  [skip] {task}: {e}")
        return []
    pairs = []
    for row in ds:
        answer = str(row.get("answer", "")).strip()
        if not answer or answer not in label_to_text:
            continue
        chosen = label_to_text[answer]
        rejected_label = next((k for k in label_to_text if k != answer), None)
        if rejected_label is None:
            continue
        rejected = label_to_text[rejected_label]
        ctx = ""
        for col in ["text", "passage", "clause", "question", "rule"]:
            if col in row and row[col]:
                ctx = str(row[col]).strip()
                break
        question = str(row.get("question", "")).strip()
        prompt = (
            "Legal context:\n"
            f"{ctx}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        ) if question and ctx != question else (
            "Legal scenario:\n"
            f"{ctx}\n\n"
            "Answer: "
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": task})
    rng.shuffle(pairs)
    return pairs[:n_max]


# Contract NLI: many variants (each is a separate hypothesis tested per clause)
CONTRACT_NLI_LABELS = {
    "Entailment": "The contract clause entails this hypothesis.",
    "Contradiction": "The contract clause contradicts this hypothesis.",
    "NotMentioned": "The contract clause does not address this hypothesis.",
}
CONTRACT_NLI_TASKS = [
    "contract_nli_explicit_identification",
    "contract_nli_inclusion_of_verbally_conveyed_information",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_no_reverse_engineering",
    "contract_nli_no_solicitation",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_acquirement_of_similar_information",
    "contract_nli_permissible_copy",
    "contract_nli_permissible_development_of_similar_information",
    "contract_nli_permissible_post-agreement_possession",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
]

# Simple binary tasks not yet covered
EXTRA_BINARY = [
    ("legal_reasoning_causality",
        {"Yes": "Yes, this passage exhibits causal legal reasoning.",
         "No":  "No, this passage does not exhibit causal legal reasoning."}),
    ("corporate_lobbying",
        {"Yes": "Yes, this is a corporate lobbying activity.",
         "No":  "No, this is not a corporate lobbying activity."}),
    ("rule_qa",  None),  # open-ended, skip
    ("citation_prediction_classification",
        {"Yes": "Yes, this is the correct citation.",
         "No":  "No, this is not the correct citation."}),
]

for task in CONTRACT_NLI_TASKS:
    pairs = build_lb_binary(task, CONTRACT_NLI_LABELS)
    if pairs:
        save_jsonl(pairs, OUT_DIR / f"{task}_test.jsonl")

for task, mapping in EXTRA_BINARY:
    if mapping is None:
        continue
    pairs = build_lb_binary(task, mapping)
    if pairs:
        save_jsonl(pairs, OUT_DIR / f"{task}_test.jsonl")

print("\nAll transfer pair files:")
total = 0
for p in sorted(OUT_DIR.glob("*.jsonl")):
    n = sum(1 for _ in p.open(encoding="utf-8"))
    total += n
    print(f"  {p.name:60s} n={n}")
print(f"\nTotal: {total} pairs across {len(list(OUT_DIR.glob('*.jsonl')))} datasets")
