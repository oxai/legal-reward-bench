# Eval

Measures reward accuracy: given a preference pair, does the model assign higher log-probability to the chosen response than the rejected one? This is the standard cheap metric for DPO — no judge calls needed.

## Commands

Baseline (before DPO, on ContextualJudgeBench QA splits):

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --source contextual_judge_bench
```

After DPO (LoRA adapter):

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model pipeline/05_reward_model/outputs/dpo_model \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --source contextual_judge_bench
```

Evaluate on our own pipeline pairs:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model pipeline/05_reward_model/outputs/dpo_model \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --pairs pipeline/04_pairs/outputs/pairs.jsonl
```

Limit pairs for a quick sanity check:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --source contextual_judge_bench \
  --limit 50
```

## Output

Prints a table and writes a CSV to `outputs/`:

```
split                  n    reward_accuracy
-------------------------------------------
overall             1250           0.6340
completeness_qa      250           0.6120
conciseness_qa       250           0.6280
faithfulness_qa      250           0.6480
refusal_answerable   250           0.6400
refusal_unanswerable 250           0.6440
```

## Key arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | HF model id or path to saved model or LoRA adapter |
| `--base-model` | — | Required when `--model` is a LoRA adapter |
| `--source` | `pipeline` | `pipeline` or `contextual_judge_bench` |
| `--splits` | all QA splits | Comma-separated CJB splits to evaluate |
| `--max-length` | `1024` | Max tokens for prompt + completion |
| `--limit` | — | Cap number of pairs (useful for quick checks) |
