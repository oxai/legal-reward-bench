# Reward Model

Fine-tunes a small causal LM on preference pairs via SFT warm-up followed by DPO.
Default base model: `Qwen/Qwen3.5-2B` (instruct, 2B parameters).

## Recommended workflow

### 1. SFT warm-up (train on chosen responses)

```bash
python pipeline/05_reward_model/train_sft.py --source contextual_judge_bench
```

### 2. DPO from SFT checkpoint

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source contextual_judge_bench \
  --sft-model pipeline/05_reward_model/outputs/sft_model
```

### DPO only (skip SFT)

```bash
python pipeline/05_reward_model/train_dpo.py --source contextual_judge_bench
```

### From pipeline pairs

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source pipeline \
  --pairs pipeline/04_pairs/outputs/pairs.jsonl
```

### Quick smoke-test (10 pairs, 1 epoch)

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source pipeline --pairs pipeline/04_pairs/outputs/pairs.jsonl \
  --limit 10 --epochs 1
```

## Shared arguments (SFT and DPO)

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen3.5-2B` | HuggingFace model id or local path |
| `--source` | `pipeline` | `pipeline` or `contextual_judge_bench` |
| `--splits` | QA splits | Comma-separated CJB splits |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--learning-rate` | `2e-5` | Learning rate |
| `--warmup-ratio` | `0.1` | LR warmup fraction |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--max-length` | `1024` | Max sequence length |
| `--eval-split` | `0.0` | Held-out eval fraction (0 = no eval, avoids OOM) |
| `--limit` | — | Cap dataset size for quick runs |

## DPO-only arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--sft-model` | — | Path to SFT adapter to merge before DPO |
| `--beta` | `0.1` | KL penalty coefficient |
| `--output-dir` | `outputs/dpo_model` | Where to save the adapter |

## SFT-only arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `outputs/sft_model` | Where to save the adapter |

## Scoring policy (stage 04)

Responses are scored before pairing. For **answerable** triples the score sums:
faithfulness (+4 fully / +2 partial / -2 unsupported / -4 contradicted),
correctness (+4 correct / -4 incorrect),
completeness (+2 complete / -1 incomplete),
conciseness (+1 concise / -1 not_concise). Abstaining on an answerable triple scores -8.

For **unanswerable** triples abstaining scores +5; attempting scores -5.

Only pairs with a score gap ≥ `--min-score-gap` (default 1.0) are kept.
