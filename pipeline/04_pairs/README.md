# Pairs

Converts pointwise labels into chosen/rejected preference pairs.

Within each question/context-variant cell, all unordered pairs among candidate
responses are compared with the paper's hierarchy:

1. answer behavior
2. faithfulness
3. correctness
4. completeness

For answerable contexts, attempted answers are preferred over abstentions or
unusable responses. If both responses attempted an answer, semantic labels are
compared in the order above. For unanswerable contexts, abstentions are
preferred over attempted answers, and responses tied on answer behavior do not
produce a strict preference.

Output records include traceability fields (`chosen_response_id`,
`rejected_response_id`, label ids, and labels) plus materialized `prompt`,
`chosen`, and `rejected` fields for direct DPO training/evaluation.

```bash
uv run python pipeline/04_pairs/build.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --output pipeline/04_pairs/outputs/pairs.jsonl
```
