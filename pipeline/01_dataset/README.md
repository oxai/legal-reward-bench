# Dataset

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
