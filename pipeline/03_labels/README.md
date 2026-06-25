# Stage 03: Labels

Labels generated responses with a structured judge.

Input: triples JSONL and response JSONL.

Output: `pipeline/03_labels/outputs/pointwise_labels__*.jsonl`.

Run:

```bash
uv run python pipeline/03_labels/build.py \
  --responses pipeline/02_responses/outputs/<responses>.jsonl
```

Defaults: `digitalocean/openai-gpt-oss-120b`, `--max-tokens 2048`, `--concurrency 30`.

Metrics:

```bash
uv run python pipeline/03_labels/metrics.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl
```
