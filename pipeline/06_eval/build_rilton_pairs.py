"""
Build prompt/chosen/rejected records from Rilton's Legal RAG Bench dataset.

Outputs:
  data/rilton/pairs_all.jsonl   -- all 1,220 pairs (for cross-benchmark transfer eval)
  data/rilton/pairs_train.jsonl -- 80% for DPO training
  data/rilton/pairs_test.jsonl  -- 20% held-out for in-distribution eval

Each record: {prompt, chosen, rejected, split, triple_id, chosen_labels, rejected_labels}

Split is derived from the decisive label dimension:
  answer_behavior differs -> "refusal"
  faithfulness differs    -> "faithfulness"
  correctness differs     -> "correctness"
  completeness differs    -> "completeness"
  (unanswerable context variants tagged with "_unanswerable" suffix)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "rilton"
PROMPT_TEMPLATE = (
    ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
).read_text(encoding="utf-8")

UNANSWERABLE_VARIANTS = {"bm25_hard_negative", "nomic_hard_negative", "random_context"}

MODELS = ["alibaba-qwen3-32b", "deepseek-3.2", "glm-5", "mistral-3-14B"]
VARIANTS = [
    "bm25_hard_negative",
    "gold_plus_bm25_distractors",
    "gold_plus_nomic_distractors",
    "gold_plus_random_distractors",
    "nomic_hard_negative",
    "random_context",
]


def load_triples() -> dict[str, dict]:
    triples = {}
    with (DATA / "triples" / "triples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            triples[t["id"]] = t
    return triples


def load_answers() -> dict[str, str]:
    answers: dict[str, str] = {}
    for model in MODELS:
        for variant in VARIANTS:
            path = DATA / "answers" / model / f"{variant}.jsonl"
            if not path.exists():
                print(f"  WARNING: missing {path}")
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    answers[row["id"]] = row["response"]
    return answers


def get_decisive_split(chosen_labels: dict, rejected_labels: dict, triple_id: str) -> str:
    variant = triple_id.split("__", 1)[1] if "__" in triple_id else ""
    unanswerable = variant in UNANSWERABLE_VARIANTS

    cl, rl = chosen_labels, rejected_labels
    if cl.get("answer_behavior") != rl.get("answer_behavior"):
        return "refusal_unanswerable" if unanswerable else "refusal_answerable"
    if cl.get("faithfulness") != rl.get("faithfulness"):
        return "faithfulness"
    if cl.get("correctness") != rl.get("correctness"):
        return "correctness"
    if cl.get("completeness") != rl.get("completeness"):
        return "completeness"
    return "tied"


def main() -> None:
    random.seed(42)

    print("Loading triples...", flush=True)
    triples = load_triples()
    print(f"  {len(triples)} triples")

    print("Loading answers...", flush=True)
    answers = load_answers()
    print(f"  {len(answers)} responses")

    print("Building pairs...", flush=True)
    records = []
    skipped = 0
    with (DATA / "pairs" / "oss120b.jsonl").open(encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            triple_id = pair["triple_id"]
            chosen_id = pair["chosen_response_id"]
            rejected_id = pair["rejected_response_id"]

            triple = triples.get(triple_id)
            chosen_text = answers.get(chosen_id)
            rejected_text = answers.get(rejected_id)

            if triple is None or chosen_text is None or rejected_text is None:
                skipped += 1
                continue

            prompt = PROMPT_TEMPLATE.format(
                context=triple["context"],
                question=triple["question"],
            )
            split = get_decisive_split(
                pair["chosen_labels"], pair["rejected_labels"], triple_id
            )

            records.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "split": split,
                "triple_id": triple_id,
                "chosen_labels": pair["chosen_labels"],
                "rejected_labels": pair["rejected_labels"],
            })

    print(f"  Built {len(records)} records ({skipped} skipped)")

    split_counts: dict[str, int] = {}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c}")

    # 80/10/10 question-level split (train/dev/test).
    # Split on base question ID (prefix before "__") so the same question never
    # appears across splits regardless of model/variant.
    from collections import defaultdict
    by_question: dict[str, list] = defaultdict(list)
    for r in records:
        base_qid = r["triple_id"].split("__")[0]
        by_question[base_qid].append(r)

    question_ids = sorted(by_question.keys())
    random.shuffle(question_ids)
    n_test_q = max(1, round(len(question_ids) * 0.1))
    n_dev_q  = max(1, round(len(question_ids) * 0.1))
    test_qids  = set(question_ids[:n_test_q])
    dev_qids   = set(question_ids[n_test_q:n_test_q + n_dev_q])

    train_records, dev_records, test_records = [], [], []
    for r in records:
        base_qid = r["triple_id"].split("__")[0]
        if base_qid in test_qids:
            test_records.append(r)
        elif base_qid in dev_qids:
            dev_records.append(r)
        else:
            train_records.append(r)

    random.shuffle(train_records)
    random.shuffle(dev_records)
    random.shuffle(test_records)

    n_train_q = len(question_ids) - n_test_q - n_dev_q
    print(f"\nQuestion-level 80/10/10 split: {n_train_q} train / {n_dev_q} dev / {n_test_q} test questions")

    def write_jsonl(path: Path, recs: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(recs)} records to {path}")

    write_jsonl(DATA / "pairs_all.jsonl", records)
    write_jsonl(DATA / "pairs_train.jsonl", train_records)
    write_jsonl(DATA / "pairs_dev.jsonl", dev_records)
    write_jsonl(DATA / "pairs_test.jsonl", test_records)

    print(f"\nDone. Train={len(train_records)}, Dev={len(dev_records)}, Test={len(test_records)}")


if __name__ == "__main__":
    main()
