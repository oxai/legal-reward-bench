# Labels

Labels candidate responses with a conditional judge:

1. `answer_behavior`: `attempted`, `abstained`, or `unusable`
2. For attempted answers, one focused judge call each for `faithfulness`,
   `correctness`, and `completeness`

Input:

```text
pipeline/02_responses/outputs/*.jsonl
```

Outputs:

```text
outputs/pointwise_labels__<rubric>__<judge>.jsonl
outputs/metrics__*.csv
```

Paper labelling command:

```bash
uv run python pipeline/03_labels/build.py \
  --responses pipeline/02_responses/outputs/<responses>.jsonl
```

The default judge is `digitalocean/openai-gpt-oss-120b`, decoded
deterministically with the paper's `--max-tokens 2048` cap and
`--concurrency 30`.

Extract label metrics:

```bash
uv run python pipeline/03_labels/metrics.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl
```
