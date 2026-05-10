## Setup

Use `uv` to create and sync the local Python environment:

```bash
uv sync
```

For local answer generation, make sure Ollama is running:

```bash
brew services start ollama
```

## Pipeline

```text
pipeline/01_triples/
  Legal RAG Bench -> base triples -> context-variant triples

pipeline/02_responses/
  context-variant triples -> candidate model responses

pipeline/03_labels/
  candidate responses -> pointwise judge labels and metrics

pipeline/04_pairs/
  pointwise labels -> chosen/rejected preference pairs

pipeline/05_reward_model/
  preference pairs -> reward model training

pipeline/06_eval/
  reward model and policy evaluation
```

## Current Commands

Build triples:

```bash
uv run python pipeline/01_triples/build.py
```

Generate candidate responses:

```bash
uv run python pipeline/02_responses/generate.py --model qwen3.5:4b --limit 50 --no-think
```

Label responses:

```bash
uv run python pipeline/03_labels/label.py --responses pipeline/02_responses/outputs/<responses>.jsonl
```

Extract label metrics:

```bash
uv run python pipeline/03_labels/metrics.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl
```
