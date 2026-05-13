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
uv run python pipeline/03_labels/build.py --responses pipeline/02_responses/outputs/<responses>.jsonl
```

Extract label metrics:

```bash
uv run python pipeline/03_labels/metrics.py \
  --triples pipeline/01_triples/outputs/triples.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl \
  --labels pipeline/03_labels/outputs/<labels>.jsonl
```

Build preference pairs:

```bash
uv run python pipeline/04_pairs/build.py \
  --labels pipeline/03_labels/outputs/<labels>.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl [<more>.jsonl ...]
```

DPO fine-tuning:

```bash
uv run python pipeline/05_reward_model/train_dpo.py \
  --pairs pipeline/04_pairs/outputs/pairs.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct
```

Evaluate reward accuracy (before/after DPO):

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --source contextual_judge_bench

uv run python pipeline/06_eval/eval_dpo.py \
  --model pipeline/05_reward_model/outputs/dpo_model \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --source contextual_judge_bench
```
