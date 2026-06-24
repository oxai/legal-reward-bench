# Eval

Measures reward accuracy: given a preference pair, does the model assign higher
log-probability to the chosen response than the rejected one?

## Commands

LegalRewardBench-v2 evaluation uses the full-prompt val+test split and
length-normalized scoring:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model outputs/ministral-8b-combined-v2 \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --source pipeline \
  --pairs data/lrb_v2/pairs_valtest.jsonl \
  --output results/eval_lrb.csv \
  --length-normalize \
  --max-length 8192
```

ContextualJudgeBench evaluation keeps summed log probability for comparability
with CJB-style sequence scoring:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model outputs/ministral-8b-combined-v2 \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --source pipeline \
  --pairs data/cjb/cjb_test.jsonl \
  --output results/eval_cjb.csv \
  --max-length 4096
```

Baseline model sanity check:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model mistralai/Ministral-8B-Instruct-2410 \
  --source pipeline \
  --pairs data/lrb_v2/pairs_valtest.jsonl \
  --length-normalize \
  --max-length 8192 \
  --limit 50
```

Limit pairs for a quick sanity check:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model outputs/ministral-8b-combined-v2 \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --source pipeline \
  --pairs data/lrb_v2/pairs_valtest.jsonl \
  --length-normalize \
  --max-length 8192 \
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
| `--max-length` | `4096` | Max tokens for prompt + completion |
| `--limit` | — | Cap number of pairs (useful for quick checks) |
| `--length-normalize` | off | Use mean per-token log-probability instead of summed log-probability |
