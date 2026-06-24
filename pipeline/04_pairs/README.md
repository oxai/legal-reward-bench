# Pairs

Stage 04 converts pointwise labels into fully materialized DPO preference
pairs. It also owns dataset-shaped transformations: train/dev/test splitting
and LegalRewardBench-v2 length augmentation.

Stages 05 and 06 should consume JSONL files from this stage directly. They
should not rebuild pairs or perform dataset augmentation.

## Pair Construction

Within each question/context-variant cell, all unordered pairs among candidate
responses are compared with the paper's hierarchy:

1. answer behavior
2. faithfulness
3. correctness
4. completeness

For answerable contexts, attempted answers are preferred over abstentions or
unusable responses. If both responses attempted an answer, semantic labels are
compared in the hierarchy above. For unanswerable contexts, abstentions are
preferred over attempted answers, and responses tied on answer behavior do not
produce a strict preference.

```bash
uv run python pipeline/04_pairs/build.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --output data/lrb/pairs_all.jsonl
```

## Canonical Pair Schema

Every output record is materialized. The model-training fields are:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Traceability fields are retained alongside the materialized text:

```json
{
  "id": "...",
  "triple_id": "...",
  "chosen_response_id": "...",
  "rejected_response_id": "...",
  "chosen_label_id": "...",
  "rejected_label_id": "...",
  "chosen_labels": {},
  "rejected_labels": {},
  "split": "refusal_answerable",
  "metadata": {
    "policy_version": "hierarchical_v1",
    "preference_type": "refusal_answerable",
    "dataset_split": "train",
    "base_triple_id": "...",
    "context_variant": "...",
    "answerability": "answerable"
  }
}
```

The top-level `split` field is the preference/evaluation category used by the
evaluation scripts. Train/dev/test membership is recorded in
`metadata.dataset_split` and by the output file name.

## Train/Dev/Test Splits

The released Hugging Face dataset provides the canonical train/dev/test files
used by the paper. Use those files for paper reproduction.

`split.py` is only a local utility for deterministic experimental splits when
rebuilding pairs from scratch. Splitting is question-level: all context variants
and model comparisons for the same base question stay in the same split.

```bash
uv run python pipeline/04_pairs/split.py \
  --input data/lrb/pairs_all.jsonl \
  --output-dir data/lrb \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --seed 42 \
  --write-valtest
```

## LegalRewardBench-v2 Length Augmentation

`length_augment.py` creates LRB-v2 by regenerating the short-template side of
length-asymmetric pairs with Ministral-8B-Instruct-2410. This preserves the
preference semantics while reducing response-length artefacts.

Run it separately for each split:

```bash
uv run python pipeline/04_pairs/length_augment.py \
  --input data/lrb/pairs_train.jsonl \
  --output data/lrb_v2/pairs_train.jsonl

uv run python pipeline/04_pairs/length_augment.py \
  --input data/lrb/pairs_dev.jsonl \
  --output data/lrb_v2/pairs_dev.jsonl

uv run python pipeline/04_pairs/length_augment.py \
  --input data/lrb/pairs_test.jsonl \
  --output data/lrb_v2/pairs_test.jsonl
```

Modified records are marked with the release-compatible top-level `_modified`
field and with richer details in `metadata.length_augmentation`.
