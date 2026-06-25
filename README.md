# Legal Reward Modeling

Pipeline for constructing LegalRewardBench, generating and labelling candidate
answers, building preference pairs, and training/evaluating contextual reward
models.

Released data: [`riltonfranzone/legal-reward-bench`](https://huggingface.co/datasets/riltonfranzone/legal-reward-bench)

## Setup

```bash
uv sync --locked
cp .env.example .env
```

Set credentials only for the stages you run:

```bash
export PYTHONPATH=$(pwd)
export DIGITAL_OCEAN_ACCESS_TOKEN=<token>      # generation and labels
export HUGGING_FACE_HUB_TOKEN=<token>          # gated HF models, if needed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The locked stack uses `torch==2.4.1`, `transformers==4.45.2`,
`trl==0.11.4`, `peft==0.12.0`, and `accelerate==0.34.2`.

## Pipeline

| Stage | Path | Output |
|---|---|---|
| 01 | `pipeline/01_triples` | Legal RAG Bench triples and context variants |
| 02 | `pipeline/02_responses` | Candidate model responses |
| 03 | `pipeline/03_labels` | Pointwise judge labels |
| 04 | `pipeline/04_pairs` | Materialized preference pairs |
| 05 | `pipeline/05_reward_model` | SFT/DPO adapters |
| 06 | `pipeline/06_eval` | Reward-accuracy results |

Each stage directory has a short README with inputs, outputs, and script names.

## Use Released Data

```bash
uv run huggingface-cli download riltonfranzone/legal-reward-bench \
  --repo-type dataset \
  --local-dir data/hf/legal-reward-bench

mkdir -p data/lrb_v2 data/training data/cjb

cp data/hf/legal-reward-bench/data/legal_reward_bench_v2/train.jsonl data/lrb_v2/pairs_train.jsonl
cp data/hf/legal-reward-bench/data/legal_reward_bench_v2/dev.jsonl data/lrb_v2/pairs_dev.jsonl
cp data/hf/legal-reward-bench/data/legal_reward_bench_v2/test.jsonl data/lrb_v2/pairs_test.jsonl
cp data/hf/legal-reward-bench/data/legal_reward_bench_v2/valtest.jsonl data/lrb_v2/pairs_valtest.jsonl
cp data/hf/legal-reward-bench/data/training/cjb_lrb_v2_train_dpo.jsonl data/training/cjb_lrb_v2_train_dpo.jsonl
cp data/hf/legal-reward-bench/data/contextual_judge_bench/test.jsonl data/cjb/cjb_test.jsonl
```

## Train

Paper-style Ministral-8B DPO run:

```bash
uv run python pipeline/05_reward_model/train_dpo.py \
  --source pipeline \
  --pairs data/training/cjb_lrb_v2_train_dpo.jsonl \
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

Use `--qlora` for larger models that need 4-bit NF4 loading.

## Evaluate

LegalRewardBench-v2 uses length-normalised scoring:

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

ContextualJudgeBench uses summed log probability:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
  --model outputs/ministral-8b-combined-v2 \
  --base-model mistralai/Ministral-8B-Instruct-2410 \
  --source pipeline \
  --pairs data/cjb/cjb_test.jsonl \
  --output results/eval_cjb.csv \
  --max-length 4096
```

## Rebuild Data

```bash
# 01. triples and context variants
uv run python pipeline/01_triples/build.py --variants

# 02. candidate responses
for model in \
  digitalocean/alibaba-qwen3-32b \
  digitalocean/deepseek-3.2 \
  digitalocean/glm-5 \
  digitalocean/mistral-3-14B
do
  uv run python pipeline/02_responses/generate.py --model "$model"
done

# 03. labels
uv run python pipeline/03_labels/build.py \
  --responses pipeline/02_responses/outputs/<responses>.jsonl

# 04. preference pairs
uv run python pipeline/04_pairs/build.py \
  --labels pipeline/03_labels/outputs/<labels>.jsonl \
  --responses pipeline/02_responses/outputs/<responses>.jsonl [<more>.jsonl ...] \
  --output data/lrb/pairs_all.jsonl
```

Use the released HF split files for paper reproduction. `pipeline/04_pairs/split.py`
is available for local deterministic question-level splits when rebuilding from
scratch.
