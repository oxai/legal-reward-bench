# AI for high-stakes: Legal Reward Modeling

## Setup

Use `uv` to create and sync the local Python environment:

```bash
uv sync
```

For answer generation, make sure you have ollama running locally, e.g.

```bash
brew services start ollama
```

## Pipeline

```text
pipeline/01_dataset/
  Legal RAG Bench dataset -> triples -> candidate responses

pipeline/02_preference/
  candidate responses -> labels -> preference pairs

pipeline/03_judge/
  preference pairs -> judge model
```

### Dataset

Builds the base research dataset: triples and candidate responses.

```text
outputs/triples/legal_rag_bench.jsonl
outputs/candidate_responses/<model>__<prompt_version>.jsonl
```

Build triples:

```bash
uv run python pipeline/01_dataset/build_triples.py
```

Generate candidate responses:

```bash
uv run python pipeline/01_dataset/generate_responses.py --model qwen3.5:4b --limit 50 --no-think
```

Triple schema:

```json
{
  "id": "...",
  "question": "...",
  "context": "...",
  "answer": "...",
  "metadata": {}
}
```

## Preference

Labels the candidate answers and produces preference pairs.

## Judge

The actual model
