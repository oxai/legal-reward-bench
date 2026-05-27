#!/bin/bash
# Reproduces the headline DPO-CJB+LRB-v2 result from the paper.
# Tested on: H100 80GB, Python 3.11, CUDA 12.4
# Runtime: ~80 min training + ~10 min eval

set -e

# ── dependencies ──────────────────────────────────────────────────────────────
pip install \
    'transformers>=4.45.0,<4.46.0' \
    'tokenizers>=0.20.0' \
    'trl==0.11.4' \
    'peft==0.12.0' \
    'accelerate==0.34.2' \
    datasets bitsandbytes

# ── environment ───────────────────────────────────────────────────────────────
export PYTHONPATH=$(pwd)
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/path/to/hf_cache   # optional: point to a pre-populated cache
# export HUGGING_FACE_HUB_TOKEN=...  # required if base model is gated

BASE=mistralai/Ministral-8B-Instruct-2410
PAIRS=pipeline/06_eval/outputs/model_sweep/combined_train_v2_truncated.jsonl
ADAPTER=outputs/ministral-8b-combined-v2
EVAL_PAIRS=data/rilton/pairs_valtest_v2.jsonl
EVAL_OUT=outputs/eval_combined_v2.csv

mkdir -p outputs

# ── train ─────────────────────────────────────────────────────────────────────
echo "=== Training DPO-CJB+LRB-v2 (seed=42) ==="
python3 pipeline/05_reward_model/train_dpo.py \
    --source pipeline \
    --pairs "$PAIRS" \
    --base-model "$BASE" \
    --output-dir "$ADAPTER" \
    --max-length 2048 --batch-size 1 --grad-accum 16 \
    --lora-r 32 --lora-alpha 64 --epochs 3 --seed 42

# ── eval on LRB-v2 ────────────────────────────────────────────────────────────
echo "=== Evaluating on LegalRewardBench-v2 (val+test) ==="
python3 pipeline/06_eval/eval_dpo.py \
    --model "$ADAPTER" \
    --base-model "$BASE" \
    --source pipeline \
    --pairs "$EVAL_PAIRS" \
    --output "$EVAL_OUT" \
    --length-normalize \
    --max-length 8192

echo "=== Results ==="
grep '^overall' "$EVAL_OUT"
# Expected: overall accuracy ~0.849 (mean across 3 seeds)
