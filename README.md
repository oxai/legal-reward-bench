## Overview

This repository contains the full pipeline for training and evaluating contextual reward models for legal reasoning. The pipeline covers data construction, preference pair generation, DPO fine-tuning, and evaluation on legal benchmarks.

Trained adapters are available at `VNoelDVT/legal-rm-adapters` (private HuggingFace repository).

---

## Setup

Use `uv` to create and sync the local Python environment:

```bash
uv sync
```

Copy the environment template and fill in private credentials:

```bash
cp .env.example .env
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

Expected output: overall pairwise reward accuracy ~84.9% on LegalRewardBench-v2 (val+test, length-normalised scoring). This matches the mean reported across three seeds in the paper.

Required hardware: one GPU with at least 40 GB VRAM (tested on H100 80 GB). Training takes approximately 80 minutes.

### Environment variables

```bash
export PYTHONPATH=$(pwd)
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DIGITAL_OCEAN_ACCESS_TOKEN=<your_token> # for DigitalOcean-hosted generation and labels
export HUGGING_FACE_HUB_TOKEN=<your_token>   # if base model requires authentication
export HF_HOME=/path/to/hf_cache             # optional
```

### Training a single model manually

Prepare the combined CJB+LRB-v2 training file:

```bash
python3 pipeline/05_reward_model/prepare_training_data.py \
    --lrb-train data/lrb_v2/pairs_train.jsonl \
    --output data/training/cjb_lrb_v2_train_dpo_2048.jsonl
```

```bash
python3 pipeline/05_reward_model/train_dpo.py \
    --source pipeline \
    --pairs data/training/cjb_lrb_v2_train_dpo_2048.jsonl \
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
    --pairs data/lrb_v2/pairs_valtest.jsonl \
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
    --pairs data/cjb/cjb_test.jsonl \
    --output results/eval_cjb.csv \
    --max-length 4096
```

Note: CJB uses sum log-probability (no `--length-normalize`). Using length-normalised scoring on CJB gives different numbers and breaks comparability with the paper.

---

## Data

### Download released artifacts

The paper data release is hosted at `VNoelDVT/legal-rm-data`. The current HF
release keeps legacy `data/rilton/` path names; the commands below copy those
files into neutral local `data/lrb_v2/` paths used by this repository.

```bash
huggingface-cli download VNoelDVT/legal-rm-data \
    --repo-type dataset \
    --local-dir data/hf/legal-rm-data

mkdir -p data/lrb_v2 data/training data/cjb

cp data/hf/legal-rm-data/data/rilton/pairs_train_v2.jsonl \
   data/lrb_v2/pairs_train.jsonl
cp data/hf/legal-rm-data/data/rilton/pairs_dev_v2.jsonl \
   data/lrb_v2/pairs_dev.jsonl
cp data/hf/legal-rm-data/data/rilton/pairs_test_v2.jsonl \
   data/lrb_v2/pairs_test.jsonl
cp data/hf/legal-rm-data/data/rilton/pairs_valtest_v2.jsonl \
   data/lrb_v2/pairs_valtest.jsonl

cp data/hf/legal-rm-data/eval_outputs/model_sweep/combined_train_v2_truncated.jsonl \
   data/training/cjb_lrb_v2_train_dpo_2048.jsonl
cp data/hf/legal-rm-data/eval_outputs/model_sweep/cjb_test.jsonl \
   data/cjb/cjb_test.jsonl
```

### LegalRewardBench-v2

LegalRewardBench-v2 is the length-balanced version of LegalRewardBench. For
paper reproduction, use the canonical train/dev/test files from the released
Hugging Face dataset.

Length augmentation is a Stage 04 dataset transformation. If rebuilding the
dataset locally, run it on each split:

```bash
python3 pipeline/04_pairs/length_augment.py \
    --input data/lrb/pairs_train.jsonl \
    --output data/lrb_v2/pairs_train.jsonl

python3 pipeline/04_pairs/length_augment.py \
    --input data/lrb/pairs_dev.jsonl \
    --output data/lrb_v2/pairs_dev.jsonl

python3 pipeline/04_pairs/length_augment.py \
    --input data/lrb/pairs_test.jsonl \
    --output data/lrb_v2/pairs_test.jsonl
```

`length_augment.py` uses Ministral-8B to generate length-matched alternatives
for the short refusal templates in the original dataset. Takes approximately
30 minutes on an H100 for the training split.

### Combined DPO training data

The released LRB-v2 benchmark files keep full prompts. For the combined DPO
training run, the LRB-v2 train prompts are tail-truncated before concatenating
with ContextualJudgeBench training pairs so the legal question and answer marker
fit inside the fixed DPO context budget. This transformation is training-only;
evaluation uses full prompts.

```bash
python3 pipeline/05_reward_model/prepare_training_data.py \
    --lrb-train data/lrb_v2/pairs_train.jsonl \
    --output data/training/cjb_lrb_v2_train_dpo_2048.jsonl
```

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

Build triples with all context variants used by the paper:

```bash
uv run python pipeline/01_triples/build.py --variants
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
    --responses pipeline/02_responses/outputs/<responses>.jsonl [<more>.jsonl ...] \
    --output data/lrb/pairs_all.jsonl
```

The canonical paper splits are distributed as split files in the released
Hugging Face dataset. For local validation after rebuilding `pairs_all.jsonl`,
you can create a deterministic question-level ratio split:

```bash
uv run python pipeline/04_pairs/split.py \
    --input data/lrb/pairs_all.jsonl \
    --output-dir data/lrb \
    --train-ratio 0.8 \
    --dev-ratio 0.1 \
    --seed 42 \
    --write-valtest
```

Evaluate reward accuracy:

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
