"""Build preference-pair JSONLs for additional transferability benchmarks.

For each public legal benchmark we can convert to (prompt, chosen, rejected)
pairs, we emit a JSONL compatible with eval_dpo.py --source pipeline.

Datasets covered (subject to availability):
  - CaseHOLD          (lex_glue, case_hold)        — 5-way MCQ, US case holdings
  - ConsumerContractsQA (LegalBench)               — 4-way MCQ, consumer contracts
  - Abercrombie       (LegalBench)                 — 5-way MCQ, trademark distinctiveness
  - Diversity_1       (LegalBench)                 — binary, federal diversity jurisdiction
  - Hearsay           (LegalBench)                 — binary, hearsay objection
  - ProA              (LegalBench)                 — binary, "pro-A" vote prediction
  - PersonalJurisdiction (LegalBench)              — binary, personal jurisdiction
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

YES_NO = ["No", "Yes"]


def save_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name:40s} n={len(pairs)}")


def build_casehold() -> list[dict]:
    rng = random.Random(SEED)
    ds = load_dataset("lex_glue", "case_hold", split="test")
    pairs = []
    for row in ds:
        ctx = row.get("context") or ""
        endings = row.get("endings") or []
        label = row.get("label")
        if label is None or not endings or label >= len(endings):
            continue
        chosen = endings[label]
        wrong = [e for i, e in enumerate(endings) if i != label and e]
        if not wrong:
            continue
        rejected = rng.choice(wrong)
        prompt = (
            "Legal context (case excerpt):\n"
            f"{ctx}\n\n"
            "Question: Which holding correctly completes the excerpt above?\n\n"
            "Answer: "
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": "casehold"})
    rng.shuffle(pairs)
    return pairs[:500]  # subsample to keep eval fast (~10 min per model)


def _legalbench_binary(task: str, label_to_text: dict[str, str], n_max: int = 400) -> list[dict]:
    """Generic binary-label LegalBench → preference pairs."""
    rng = random.Random(SEED)
    try:
        ds = load_dataset("nguha/legalbench", task, split="test")
    except Exception as e:
        print(f"  [skip] {task}: {e}")
        return []
    pairs = []
    for row in ds:
        answer = str(row.get("answer", "")).strip()
        text = row.get("text") or ""
        question = row.get("question") or ""
        if not answer or answer not in label_to_text:
            continue
        chosen = label_to_text[answer]
        rejected_label = next((k for k in label_to_text if k != answer), None)
        if rejected_label is None:
            continue
        rejected = label_to_text[rejected_label]
        prompt = (
            "Legal context:\n"
            f"{text}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": task})
    rng.shuffle(pairs)
    return pairs[:n_max]


def _legalbench_mcq(task: str, choice_cols: list[str], answer_col: str, n_max: int = 400) -> list[dict]:
    """Generic multi-choice LegalBench → preference pairs."""
    rng = random.Random(SEED)
    try:
        ds = load_dataset("nguha/legalbench", task, split="test")
    except Exception as e:
        print(f"  [skip] {task}: {e}")
        return []
    pairs = []
    for row in ds:
        answer = str(row.get(answer_col, "")).strip()
        if not answer:
            continue
        chosen = answer
        choices = [str(row.get(c, "")).strip() for c in choice_cols if str(row.get(c, "")).strip()]
        wrong = [c for c in choices if c != chosen]
        if not wrong:
            continue
        rejected = rng.choice(wrong)
        question = row.get("question") or row.get("text") or ""
        prompt = (
            "Legal scenario:\n"
            f"{question}\n\n"
            "Answer: "
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": task})
    rng.shuffle(pairs)
    return pairs[:n_max]


print("Building CaseHOLD...")
save_jsonl(build_casehold(), OUT_DIR / "casehold_test.jsonl")

print("Building LegalBench binary tasks...")
binary_tasks = [
    ("hearsay",              {"Yes": "Yes, this is hearsay.", "No": "No, this is not hearsay."}),
    ("personal_jurisdiction", {"Yes": "Yes, personal jurisdiction exists.", "No": "No, personal jurisdiction does not exist."}),
    ("proa",                 {"Yes": "Yes, the vote is pro-A.", "No": "No, the vote is not pro-A."}),
    ("diversity_1",          {"Yes": "Yes, there is federal diversity jurisdiction.", "No": "No, there is no federal diversity jurisdiction."}),
    ("diversity_2",          {"Yes": "Yes, there is federal diversity jurisdiction.", "No": "No, there is no federal diversity jurisdiction."}),
    ("diversity_3",          {"Yes": "Yes, there is federal diversity jurisdiction.", "No": "No, there is no federal diversity jurisdiction."}),
    ("abercrombie",          None),  # 5-way, handled separately
]
for task, mapping in binary_tasks:
    if mapping is None:
        continue
    save_jsonl(_legalbench_binary(task, mapping), OUT_DIR / f"{task}_test.jsonl")

print("Done.")
print(f"\nAll pairs in: {OUT_DIR}")
for p in sorted(OUT_DIR.glob("*.jsonl")):
    n = sum(1 for _ in p.open(encoding="utf-8"))
    print(f"  {p.name:40s} n={n}")
