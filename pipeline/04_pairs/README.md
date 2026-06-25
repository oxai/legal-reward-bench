# Stage 04: Pairs

Builds materialized chosen/rejected preference pairs from pointwise labels.

Input: triples, labels, and response JSONL files.

Output: pair JSONL with `prompt`, `chosen`, and `rejected`.

Scripts:
- `build.py`: construct strict preference pairs
- `split.py`: deterministic question-level train/dev/test split
- `length_augment.py`: create LegalRewardBench-v2 length-balanced pairs

Preference hierarchy: answer behaviour, faithfulness, correctness,
completeness. Ties are discarded.

Use the released Hugging Face split files for paper reproduction.
