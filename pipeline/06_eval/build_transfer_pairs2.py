"""Second batch of transferability benchmarks — more LegalBench binary tasks
and a few MCQ tasks. Same JSONL format as build_transfer_pairs.py."""
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
    print(f"  wrote {path.name:50s} n={len(pairs)}")


def build_lb_binary(task, label_to_text, n_max=400, prompt_template=None):
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
        context_fields = ["text", "passage", "clause", "question", "rule"]
        context = next((str(row[c]).strip() for c in context_fields if c in row and row[c]), "")
        question = str(row.get("question", "")).strip()
        if prompt_template:
            prompt = prompt_template.format(context=context, question=question)
        else:
            prompt = (
                "Legal context:\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Answer: "
            ) if question else (
                "Legal scenario:\n"
                f"{context}\n\n"
                "Answer: "
            )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": task})
    rng.shuffle(pairs)
    return pairs[:n_max]


BINARY_TASKS = [
    ("ucc_v_common_law",      {"UCC": "This is governed by the UCC.", "Common Law": "This is governed by common law."}),
    ("jcrew_blocker",         {"Yes": "Yes, the clause is a J. Crew blocker.", "No": "No, the clause is not a J. Crew blocker."}),
    ("overruling",            {"overruling": "Yes, this opinion overrules a prior case.", "non-overruling": "No, this opinion does not overrule a prior case."}),
    ("canada_tax_court_outcomes", {"For Taxpayer": "The Tax Court ruled for the taxpayer.", "Against Taxpayer": "The Tax Court ruled against the taxpayer.", "Other": "The Tax Court issued a different ruling."}),
    ("successor_liability",   {"Yes": "Yes, successor liability applies.", "No": "No, successor liability does not apply."}),
    ("unfair_tos",            {"Yes": "Yes, this clause is unfair.", "No": "No, this clause is not unfair."}),
    ("nys_judicial_ethics",   {"Yes": "Yes, this conduct is ethical for a New York judge.", "No": "No, this conduct is not ethical for a New York judge."}),
    ("contract_nli_explicit_identification",
                              {"Entailment": "The contract clause entails this hypothesis.",
                               "Contradiction": "The contract clause contradicts this hypothesis.",
                               "NotMentioned": "The contract clause does not address this hypothesis."}),
    ("contract_nli_confidentiality_of_agreement",
                              {"Entailment": "The contract clause entails this hypothesis.",
                               "Contradiction": "The contract clause contradicts this hypothesis.",
                               "NotMentioned": "The contract clause does not address this hypothesis."}),
    ("contract_nli_inclusion_of_verbally_conveyed_information",
                              {"Entailment": "The contract clause entails this hypothesis.",
                               "Contradiction": "The contract clause contradicts this hypothesis.",
                               "NotMentioned": "The contract clause does not address this hypothesis."}),
    ("supply_chain_disclosure_disclosed_accountability",
                              {"Yes": "Yes, this disclosure addresses supply chain accountability.", "No": "No, this disclosure does not address supply chain accountability."}),
    ("opp115_data_retention", {"Yes": "Yes, this policy describes data retention practices.", "No": "No, this policy does not describe data retention practices."}),
    ("opp115_third_party_sharing_collection",
                              {"Yes": "Yes, this policy describes third-party data sharing.", "No": "No, this policy does not describe third-party data sharing."}),
    ("privacy_policy_qa",     {"Relevant": "Yes, this passage answers the privacy question.", "Irrelevant": "No, this passage does not answer the privacy question."}),
    ("definition_classification", None),  # 21-class, skip — too many
    ("function_of_decision_section", None),  # 18-class, skip
]


for task, mapping in BINARY_TASKS:
    if mapping is None:
        continue
    pairs = build_lb_binary(task, mapping)
    if pairs:
        save_jsonl(pairs, OUT_DIR / f"{task}_test.jsonl")

print("\nAll transfer pairs:")
for p in sorted(OUT_DIR.glob("*.jsonl")):
    n = sum(1 for _ in p.open(encoding="utf-8"))
    print(f"  {p.name:50s} n={n}")
