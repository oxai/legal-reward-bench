# Responses

Generates candidate answers from triples.

Input:

```text
pipeline/01_triples/outputs/*.jsonl
```

Output:

```text
outputs/<model>__<prompt_version>.jsonl
```

Command:

```bash
uv run python pipeline/02_responses/generate.py --model qwen3.5:4b --no-think
```
