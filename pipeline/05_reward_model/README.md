# Reward Model

Fine-tunes causal LMs on materialized preference pairs. The paper headline run
uses DPO on a combined ContextualJudgeBench + LegalRewardBench-v2 training file,
with `mistralai/Ministral-8B-Instruct-2410` as the base model.

The script defaults are lightweight development defaults. For paper
reproduction, use the explicit commands below.

## Recommended workflow

### 0. Prepare combined CJB+LRB-v2 training data

The released LRB-v2 benchmark files keep full prompts. The combined DPO
training artifact tail-truncates only LRB-v2 train prompts, then appends
ContextualJudgeBench training pairs.

```bash
python pipeline/05_reward_model/prepare_training_data.py \
  --lrb-train data/lrb_v2/pairs_train.jsonl \
  --output data/training/cjb_lrb_v2_train_dpo_2048.jsonl
```

### 1. DPO training for the paper run

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source pipeline \
  --pairs data/training/cjb_lrb_v2_train_dpo_2048.jsonl \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --output-dir outputs/ministral-8b-combined-v2 \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --seed 42
```

### Optional SFT warm-up

```bash
python pipeline/05_reward_model/train_sft.py \
  --source pipeline \
  --pairs data/training/cjb_lrb_v2_train_dpo_2048.jsonl \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --output-dir outputs/ministral-8b-sft \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64
```

Then pass the saved SFT adapter to DPO:

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source pipeline \
  --pairs data/training/cjb_lrb_v2_train_dpo_2048.jsonl \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --sft-model outputs/ministral-8b-sft \
  --output-dir outputs/ministral-8b-sft-dpo \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64
```

### CJB-only baseline

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source contextual_judge_bench \
  --splits refusal_answerable,refusal_unanswerable,faithfulness_qa,completeness_qa,conciseness_qa,faithfulness_summ,completeness_summ,conciseness_summ \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --output-dir outputs/ministral-8b-cjb-only \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64
```

### Quick smoke-test (10 pairs, 1 epoch)

```bash
python pipeline/05_reward_model/train_dpo.py \
  --source pipeline \
  --pairs data/training/cjb_lrb_v2_train_dpo_2048.jsonl \
  --limit 10 --epochs 1
```

## Shared arguments (SFT and DPO)

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen3.5-2B` | HuggingFace model id or local path |
| `--source` | `pipeline` | `pipeline` or `contextual_judge_bench` |
| `--splits` | QA splits | Comma-separated CJB splits |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--learning-rate` | `2e-5` | Learning rate |
| `--warmup-ratio` | `0.1` | LR warmup fraction |
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha |
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

## Pairing policy (stage 04)

Stage 04 does not use scalar score gaps. It creates strict preferences by
comparing responses within the same question/context-variant cell using the
hierarchy from the paper: answer behavior, then faithfulness, then correctness,
then completeness. Ties are discarded.

The resulting JSONL is directly consumable by DPO/SFT code because each record
contains `prompt`, `chosen`, and `rejected`.
