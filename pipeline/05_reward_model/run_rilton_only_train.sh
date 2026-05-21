#!/bin/bash
# Train a TRUE Rilton-only DPO adapter on Ministral-8B (940 Legal RAG Bench pairs only).
set -e
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp

WORK=/home/valentin/work/legal-reward-modelling
TRAIN=$WORK/pipeline/05_reward_model/train_dpo.py
PAIRS=$WORK/data/rilton/pairs_train.jsonl
OUT=/workspace/adapters/ministral-8b-rilton-only
mkdir -p $OUT

BASE=mistralai/Ministral-8B-Instruct-2410

echo "Training Rilton-only DPO on $BASE ($(date))"
echo "Pairs: $(wc -l < $PAIRS) Rilton training pairs"

python3 $TRAIN \
    --base-model "$BASE" \
    --source pipeline \
    --pairs "$PAIRS" \
    --output-dir "$OUT" \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --learning-rate 2e-5 \
    --lora-r 32 \
    --lora-alpha 64 \
    --max-length 2048 \
    --gradient-checkpointing \
    --seed 42

# Clean up intermediate checkpoints, keep only final adapter
rm -rf $OUT/checkpoint-* 2>/dev/null || true
echo "Training done: $(date)"
ls $OUT/
