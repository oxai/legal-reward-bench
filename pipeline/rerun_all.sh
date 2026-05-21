#!/bin/bash
# Master rerun after pod crash. All adapters now save to /workspace/adapters (persistent).
# Run this once on pod boot; it launches each stage in the background in sequence.
# Priority: ministral combined-v2 first (key paper result), then qwen, then sprint.
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /dev/shm/hf_cache /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
LOG=/tmp/rerun_all.log
echo "==== rerun_all started: $(date) ====" | tee $LOG

# ── Stage 1: overnight_pipeline (trains combined-v2 = key 0.795 result + seeds + SFT) ──
echo "Stage 1: overnight_pipeline (trains combined-v2 + seeds + SFT)" | tee -a $LOG
nohup bash $WORK/pipeline/overnight_pipeline.sh >> /tmp/overnight_master.log 2>&1 &
echo "  PID=$!" | tee -a $LOG

# ── Stage 2: post_pipeline waits for Stage 1 internally ──
echo "Stage 2: post_pipeline (queued, waits for overnight)" | tee -a $LOG
nohup bash $WORK/pipeline/post_pipeline.sh >> /tmp/post_pipeline.log 2>&1 &
echo "  PID=$!" | tee -a $LOG

# ── Stage 3: qwen_augmentation waits for GPU internally ──
echo "Stage 3: qwen_augmentation (queued, waits for GPU)" | tee -a $LOG
nohup bash $WORK/pipeline/qwen_augmentation.sh >> /tmp/qwen_augmentation.log 2>&1 &
echo "  PID=$!" | tee -a $LOG

# ── Stage 4: emnlp_sprint waits for prior jobs internally ──
echo "Stage 4: emnlp_sprint (queued, waits for GPU + no train_dpo.py running)" | tee -a $LOG
nohup bash $WORK/pipeline/emnlp_sprint.sh >> /tmp/emnlp_sprint.log 2>&1 &
echo "  PID=$!" | tee -a $LOG

echo "" | tee -a $LOG
echo "All stages launched. Monitor with:" | tee -a $LOG
echo "  tail -f /tmp/overnight_master.log" | tee -a $LOG
echo "  tail -f /tmp/post_pipeline.log" | tee -a $LOG
echo "  tail -f /tmp/qwen_augmentation.log" | tee -a $LOG
echo "  tail -f /tmp/emnlp_sprint.log" | tee -a $LOG
echo "==== rerun_all queued: $(date) ====" | tee -a $LOG
