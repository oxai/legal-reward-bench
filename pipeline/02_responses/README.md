# Stage 02: Responses

Generates candidate answers from triples.

Input: `pipeline/01_triples/outputs/triples.jsonl`.

Output: `pipeline/02_responses/outputs/*.jsonl`.

Run one model:

```bash
uv run python pipeline/02_responses/generate.py --model digitalocean/mistral-3-14B
```

Defaults: `--temperature 0.0`, `--max-tokens 8192`, `--concurrency 48`.

Smoke test:

```bash
uv run python pipeline/02_responses/generate.py \
  --model qwen3.5:4b \
  --no-think \
  --limit 10
```
