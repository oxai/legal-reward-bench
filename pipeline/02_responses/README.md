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

Paper generation command:

```bash
for model in \
  digitalocean/alibaba-qwen3-32b \
  digitalocean/deepseek-3.2 \
  digitalocean/glm-5 \
  digitalocean/mistral-3-14B
do
  uv run python pipeline/02_responses/generate.py \
    --model "$model"
done
```

Generation uses deterministic decoding by default (`--temperature 0.0`) with the
paper's `--max-tokens 8192` cap and `--concurrency 48`. For local Ollama smoke
tests, use a local model and disable reasoning mode explicitly:

```bash
uv run python pipeline/02_responses/generate.py \
  --model qwen3.5:4b \
  --no-think \
  --limit 10
```
