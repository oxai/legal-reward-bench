# Reward Model

Fine-tunes a small causal LM on preference pairs using Direct Preference Optimization (DPO).

## Commands

Train from pipeline pairs:

```bash
uv run python pipeline/05_reward_model/train_dpo.py \
  --pairs pipeline/04_pairs/outputs/pairs.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct
```

Train from ContextualJudgeBench (update `DATASET_NAME` in `sources/contextual_judge_bench.py` first):

```bash
uv run python pipeline/05_reward_model/train_dpo.py --source contextual_judge_bench
```

Quick smoke-test (10 pairs, 1 epoch):

```bash
uv run python pipeline/05_reward_model/train_dpo.py \
  --pairs pipeline/04_pairs/outputs/pairs.jsonl \
  --limit 10 --epochs 1
```

## Key arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model id |
| `--source` | `pipeline` | `pipeline` or `contextual_judge_bench` |
| `--beta` | `0.1` | DPO KL penalty coefficient |
| `--lora-r` | `8` | LoRA rank |
| `--max-length` | `1024` | Max total sequence length (prompt + completion) |
| `--eval-split` | `0.1` | Fraction of pairs held out for eval |
| `--limit` | — | Cap the dataset size for quick runs |
| `--output-dir` | `outputs/dpo_model` | Where to save the adapter weights |

## Scoring policy (stage 04)

Responses are scored before pairing. For **answerable** triples the score sums:
faithfulness (+4 fully / +2 partial / -2 unsupported / -4 contradicted),
correctness (+4 correct / -4 incorrect),
completeness (+2 complete / -1 incomplete),
conciseness (+1 concise / -1 not_concise). Abstaining on an answerable triple scores -8.

For **unanswerable** triples abstaining scores +5; attempting scores -5.

Only pairs with a score gap ≥ `--min-score-gap` (default 1.0) are kept.
