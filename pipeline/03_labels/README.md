# Labels

Labels candidate responses with a two-step judge:

1. `answer_behavior`: `attempted`, `abstained`, or `unusable`
2. Semantic labels for attempted answers: `faithfulness`, `correctness`,
   `completeness`, and `conciseness`

Input:

```text
pipeline/02_responses/outputs/*.jsonl
```

Outputs:

```text
outputs/pointwise_labels__<rubric>__<judge>.jsonl
outputs/metrics__*.csv
```

Commands:

```bash
uv run python pipeline/03_labels/label.py --responses pipeline/02_responses/outputs/<responses>.jsonl
uv run python pipeline/03_labels/metrics.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl
```
