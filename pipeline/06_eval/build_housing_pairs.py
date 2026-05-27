"""Build Housing Statute QA preference pairs from reglab/housing_qa.

Each row is a yes/no question on US state housing law with a gold statute excerpt.
We construct a preference pair as:
  chosen = the correct yes/no answer rendered as a short rationale
  rejected = the opposite answer rendered as a short rationale
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[2] / "data" / "transfer" / "housing_qa_test.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
SEED = 42

rng = random.Random(SEED)

# reglab/housing_qa uses a script loader (deprecated). Fetch raw zip + tsv files.
import zipfile
qz = hf_hub_download(repo_id="reglab/housing_qa", filename="data/questions.json.zip", repo_type="dataset")
sz = hf_hub_download(repo_id="reglab/housing_qa", filename="data/statutes.tsv", repo_type="dataset")

with zipfile.ZipFile(qz) as z:
    names = z.namelist()
    print(f"questions.json.zip contents: {names}")
    with z.open(names[0]) as f:
        qa_data = json.load(f)

# Load statutes lookup
import csv as _csv
_csv.field_size_limit(10**9)  # statute texts can be very long
statutes = {}
with open(sz, encoding="utf-8") as f:
    reader = _csv.DictReader(f, delimiter="\t")
    print(f"  statutes columns: {reader.fieldnames}")
    for row in reader:
        sid = str(row.get("statute_id") or row.get("id") or "").strip()
        text = str(row.get("text") or row.get("statute") or "").strip()
        if sid and text:
            statutes[sid] = text

print(f"Statutes loaded: {len(statutes)}")
print(f"QA structure type: {type(qa_data).__name__}")
if isinstance(qa_data, dict):
    print(f"QA top-level keys: {list(qa_data.keys())[:10]}")
    # Try common nestings
    for k in ("test", "questions", "data"):
        if k in qa_data:
            qa_data = qa_data[k]
            print(f"  Drilled into '{k}', type now: {type(qa_data).__name__}")
            break
if isinstance(qa_data, list):
    print(f"  {len(qa_data)} entries; sample keys: {list(qa_data[0].keys()) if qa_data else 'EMPTY'}")
    print(f"  Sample: {qa_data[0] if qa_data else 'EMPTY'}")

ds = qa_data if isinstance(qa_data, list) else list(qa_data.values())

pairs = []
for row in ds:
    answer = str(row.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        continue
    question = str(row.get("question", "")).strip()
    if not question:
        continue
    # Statutes are a list of dicts with {citation, excerpt}; join the excerpts
    statutes_list = row.get("statutes") or []
    statute_text = "\n\n".join(
        f"[{s.get('citation', '')}]\n{s.get('excerpt', '')}".strip()
        for s in statutes_list if s.get("excerpt")
    )
    state = str(row.get("state", "")).strip()
    yes_text = "Yes, based on the relevant statute."
    no_text = "No, based on the relevant statute."
    chosen, rejected = (yes_text, no_text) if answer == "yes" else (no_text, yes_text)
    header = f"State: {state}\n\n" if state else ""
    prompt = (
        f"{header}Legal statute (housing law):\n"
        f"{statute_text}\n\n"
        f"Question: {question}\n\n"
        "Answer: "
    ) if statute_text else (
        f"{header}Question: {question}\n\nAnswer: "
    )
    pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "split": "housing_qa"})

rng.shuffle(pairs)
# Match paper: 500 stratified pairs. Take first 500.
pairs = pairs[:500]
with OUT.open("w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"\nWrote {len(pairs)} pairs to {OUT}")
