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

Generation is serial by default. For local Ollama parallelism, set Ollama's parallel request limit and pass `--concurrency`:

```bash
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
uv run python pipeline/02_responses/generate.py \
  --model qwen3.5:9b \
  --no-think \
  --max-tokens 2048 \
  --concurrency 2
```
