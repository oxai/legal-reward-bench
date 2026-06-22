## Overview

This repository contains the full pipeline for training and evaluating contextual reward models for legal reasoning. The pipeline covers data construction, preference pair generation, DPO fine-tuning, and evaluation on legal benchmarks.

Trained adapters are available at `VNoelDVT/legal-rm-adapters` (private HuggingFace repository).

---

## Setup

Use `uv` to create and sync the local Python environment:

```bash
uv sync
```

For local answer generation, make sure Ollama is running:

```bash
brew services start ollama
```

For GPU training and evaluation, install the pinned dependencies directly:

```bash
pip install \
    'transformers>=4.45.0,<4.46.0' \
    'tokenizers>=0.20.0' \
    'trl==0.11.4' \
    'peft==0.12.0' \
    'accelerate==0.34.2' \
    datasets bitsandbytes
```

The version pins are load-bearing: `trl==0.11.4` requires `transformers<4.46`.

---

## Pipeline

```text
pipeline/01_triples/
  Legal RAG Bench -> base triples -> context-variant triples

pipeline/02_responses/
  context-variant triples -> candidate model responses

pipeline/03_labels/
  candidate responses -> pointwise judge labels and metrics

pipeline/04_pairs/
  pointwise labels -> chosen/rejected preference pairs

pipeline/05_reward_model/
  preference pairs -> DPO or SFT reward model training

pipeline/06_eval/
  reward model and policy evaluation across benchmarks
```

---

## Reproducing the headline result

The headline result is Ministral-8B fine-tuned with DPO on the combined CJB + LRB-v2 dataset, evaluated on LegalRewardBench-v2.

Run the full training and evaluation in one step:

```bash
bash reproduce.sh
```

Expected output: overall pairwise reward accuracy ~84.9% on LegalRewardBench-v2 (val+test, length-normalised scoring). This matches the mean reported across three seeds in the paper.

Required hardware: one GPU with at least 40 GB VRAM (tested on H100 80 GB). Training takes approximately 80 minutes.

### Environment variables

```bash
export PYTHONPATH=$(pwd)
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGING_FACE_HUB_TOKEN=<your_token>   # if base model requires authentication
export HF_HOME=/path/to/hf_cache             # optional
```

### Training a single model manually

```bash
python3 pipeline/05_reward_model/train_dpo.py \
    --source pipeline \
    --pairs pipeline/06_eval/outputs/model_sweep/combined_train_v2_truncated.jsonl \
    --base-model mistralai/Ministral-8B-Instruct-2410 \
    --output-dir outputs/ministral-8b-combined-v2 \
    --max-length 2048 --batch-size 1 --grad-accum 16 \
    --lora-r 32 --lora-alpha 64 --epochs 3 --seed 42
```

For models larger than 14B, add `--qlora` to enable 4-bit NF4 quantisation.

### Evaluating a trained adapter

On LegalRewardBench-v2 (length-normalised, used for LRB-v2 results):

```bash
python3 pipeline/06_eval/eval_dpo.py \
    --model outputs/ministral-8b-combined-v2 \
    --base-model mistralai/Ministral-8B-Instruct-2410 \
    --source pipeline \
    --pairs data/rilton/pairs_valtest_v2.jsonl \
    --output results/eval_lrb.csv \
    --length-normalize \
    --max-length 8192
```

On ContextualJudgeBench (sum log-probability, used for CJB results):

```bash
python3 pipeline/06_eval/eval_dpo.py \
    --model outputs/ministral-8b-combined-v2 \
    --base-model mistralai/Ministral-8B-Instruct-2410 \
    --source pipeline \
    --pairs pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl \
    --output results/eval_cjb.csv \
    --max-length 4096
```

Note: CJB uses sum log-probability (no `--length-normalize`). Using length-normalised scoring on CJB gives different numbers and breaks comparability with the paper.

---

## Data

### LegalRewardBench-v2

The augmented Rilton preference dataset with length-balanced alternatives:

```bash
python3 pipeline/04_pairs/improve_rilton_data.py \
    --input data/rilton/pairs_train.jsonl \
    --output data/rilton/pairs_train_v2.jsonl

python3 pipeline/04_pairs/truncate_rilton_prompts.py \
    --input data/rilton/pairs_train_v2.jsonl \
    --output data/rilton/pairs_train_v2_truncated.jsonl \
    --max-chars 3000
```

`improve_rilton_data.py` uses Ministral-8B to generate length-matched alternatives for the short refusal templates in the original dataset. Takes approximately 30 minutes on an H100.

### Transfer benchmarks

Bar Exam QA and Housing Statute QA pairs are built from the Stanford RegLab suite:

```bash
python3 pipeline/06_eval/eval_barexam.py --model <adapter> --base-model <base>

python3 pipeline/06_eval/build_housing_pairs.py
python3 pipeline/06_eval/eval_dpo.py \
    --model <adapter> --base-model <base> \
    --pairs data/transfer/housing_qa_test.jsonl \
    --output results/eval_housing.csv \
    --length-normalize --max-length 8192
```

---

## Full pipeline commands

Build triples:

```bash
uv run python pipeline/01_triples/build.py
```

Generate candidate responses:

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

Label responses:

```bash
uv run python pipeline/03_labels/build.py \
    --responses pipeline/02_responses/outputs/<responses>.jsonl
```

Extract label metrics:

```bash
uv run python pipeline/03_labels/metrics.py \
    --triples pipeline/01_triples/outputs/triples.jsonl \
    --responses pipeline/02_responses/outputs/<responses>.jsonl \
    --labels pipeline/03_labels/outputs/<labels>.jsonl
```

Build preference pairs:

```bash
uv run python pipeline/04_pairs/build.py \
    --labels pipeline/03_labels/outputs/<labels>.jsonl \
    --responses pipeline/02_responses/outputs/<responses>.jsonl [<more>.jsonl ...]
```

Evaluate reward accuracy:

```bash
uv run python pipeline/06_eval/eval_dpo.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --source contextual_judge_bench

uv run python pipeline/06_eval/eval_dpo.py \
    --model pipeline/05_reward_model/outputs/dpo_model \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --source contextual_judge_bench
```
