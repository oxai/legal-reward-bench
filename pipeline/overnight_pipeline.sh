#!/bin/bash
# Overnight pipeline (~6-8h budget) — ICML AI4Law workshop submission prep.
# Uses the EXACT training settings that produced the successful 0.737 combined adapter:
#   batch=4, grad_accum=4, default max_length=1024, no gradient_checkpointing.
# Rilton prompts are pre-truncated to ~3000 chars so they fit in max_length=1024
# (otherwise the prompt overflows and response tokens get masked → zero gradient).
set -u  # NOT -e so one failure doesn't kill downstream

export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
TRAIN_DPO=$WORK/pipeline/05_reward_model/train_dpo.py
TRAIN_SFT=$WORK/pipeline/05_reward_model/train_sft.py
IMPROVE=$WORK/pipeline/04_pairs/improve_rilton_data.py
TRUNCATE=$WORK/pipeline/04_pairs/truncate_rilton_prompts.py

BASE=mistralai/Ministral-8B-Instruct-2410
CJB_TEST=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl
CJB_TRAIN=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_train.jsonl
COMBINED_TRAIN=$WORK/pipeline/06_eval/outputs/model_sweep/combined_train.jsonl
RIL_TRAIN=$WORK/data/rilton/pairs_train.jsonl
RIL_TRAIN_T=$WORK/data/rilton/pairs_train_truncated.jsonl
RIL_TRAIN_V2=$WORK/data/rilton/pairs_train_v2.jsonl
RIL_TRAIN_V2_T=$WORK/data/rilton/pairs_train_v2_truncated.jsonl
COMBINED_V2_T=$WORK/pipeline/06_eval/outputs/model_sweep/combined_train_v2_truncated.jsonl
RIL_TEST=$WORK/data/rilton/pairs_test.jsonl
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl

# Adapters
ADAP_RIL_V1=/workspace/adapters/ministral-8b-rilton-only
ADAP_RIL_V2=/workspace/adapters/ministral-8b-rilton-v2
ADAP_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2
ADAP_COMB_S1=/workspace/adapters/ministral-8b-combined-s1
ADAP_COMB_S2=/workspace/adapters/ministral-8b-combined-s2
ADAP_SFT=/workspace/adapters/ministral-8b-combined-sft

OUT_3X3=$WORK/pipeline/06_eval/outputs/transfer_3x3
OUT_3X3V2=$WORK/pipeline/06_eval/outputs/transfer_3x3_v2
OUT_SEEDS=$WORK/pipeline/06_eval/outputs/multiseed
OUT_SFT=$WORK/pipeline/06_eval/outputs/sft_ablation
mkdir -p $OUT_3X3 $OUT_3X3V2 $OUT_SEEDS $OUT_SFT

PY=python3
LOG=/tmp/overnight_master.log
echo "==== overnight pipeline started: $(date) ====" | tee $LOG

cleanup_ckpts() {
    find /workspace/adapters $WORK/pipeline/06_eval/outputs/model_sweep -type d -name 'checkpoint-*' \
        -exec rm -rf {} + 2>/dev/null || true
}

free_status() {
    local label=$1
    echo "  [status:$label] disk=$(df -h / | awk 'NR==2{print $4}') free, shm=$(df -h /dev/shm | awk 'NR==2{print $4}') free, GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)MB" | tee -a $LOG
}

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 PAIRS=$4 OUT_CSV=$5 MAX=${6:-4096}
    if [ -f "$OUT_CSV" ]; then echo "  [skip] $TAG" | tee -a $LOG; return 0; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="--base-model $BASEARG"
    echo "  [run] $TAG -> $OUT_CSV" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source pipeline --pairs "$PAIRS" \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length $MAX $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        local ovr=$(awk -F, '/^overall,/ {print $4" CI=["$5","$6"]"}' "$OUT_CSV")
        echo "    ok: $ovr" | tee -a $LOG
    else
        echo "    FAILED — see ${OUT_CSV%.csv}.log" | tee -a $LOG
    fi
}

# DPO training with the PROVEN settings (matches the original 0.737 combined run).
do_train_dpo() {
    local TAG=$1 OUT=$2 PAIRS=$3 SEED=$4
    if [ -f "$OUT/adapter_config.json" ]; then echo "  [skip-train] $TAG" | tee -a $LOG; return 0; fi
    cleanup_ckpts
    mkdir -p $OUT
    echo "  [train-dpo] $TAG (seed=$SEED) -> $OUT" | tee -a $LOG
    free_status "before-$TAG"
    $PY $TRAIN_DPO \
        --base-model "$BASE" --source pipeline --pairs "$PAIRS" \
        --output-dir "$OUT" --epochs 3 --batch-size 4 --grad-accum 4 \
        --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 --seed $SEED \
        > $OUT/train.log 2>&1
    rm -rf $OUT/checkpoint-* 2>/dev/null || true
    if [ -f "$OUT/adapter_config.json" ]; then
        # Sanity-check: grad_norm should be > 0 in the log; flag if zero.
        local gn=$(tr '\r' '\n' < $OUT/train.log | grep -oP "'grad_norm':\s*'[^']+'" | tail -1)
        echo "    train done; last $gn" | tee -a $LOG
    else
        echo "    TRAIN FAILED — see $OUT/train.log" | tee -a $LOG
    fi
    free_status "after-$TAG"
}

# SFT — same proven settings.
do_train_sft() {
    local TAG=$1 OUT=$2 PAIRS=$3 SEED=$4
    if [ -f "$OUT/adapter_config.json" ]; then echo "  [skip-sft] $TAG" | tee -a $LOG; return 0; fi
    cleanup_ckpts
    mkdir -p $OUT
    echo "  [train-sft] $TAG (seed=$SEED) -> $OUT" | tee -a $LOG
    free_status "before-$TAG"
    $PY $TRAIN_SFT \
        --base-model "$BASE" --source pipeline --pairs "$PAIRS" \
        --output-dir "$OUT" --epochs 3 --batch-size 4 --grad-accum 4 \
        --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 --seed $SEED \
        > $OUT/train.log 2>&1
    rm -rf $OUT/checkpoint-* 2>/dev/null || true
    if [ -f "$OUT/adapter_config.json" ]; then
        echo "    sft done" | tee -a $LOG
    else
        echo "    SFT FAILED — see $OUT/train.log" | tee -a $LOG
    fi
    free_status "after-$TAG"
}

eval_triple() {
    local TAG=$1 MODEL=$2 OUTDIR=$3
    mkdir -p "$OUTDIR"
    do_eval "$TAG/CJB"     "$MODEL" "$BASE" "$CJB_TEST" "$OUTDIR/${TAG}__cjb.csv"     4096
    do_eval "$TAG/Rilton"  "$MODEL" "$BASE" "$RIL_TEST" "$OUTDIR/${TAG}__rilton.csv"  8192
    do_eval "$TAG/BarExam" "$MODEL" "$BASE" "$BAR_TEST" "$OUTDIR/${TAG}__barexam.csv" 4096
}

# ============================================================
# PHASE 0 — wait for GPU
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 0: wait for GPU ===" | tee -a $LOG
gpu_busy() {
    pgrep -f 'python3.*train_dpo.py' >/dev/null 2>&1 && return 0
    pgrep -f 'python3.*train_sft.py' >/dev/null 2>&1 && return 0
    pgrep -f 'python3.*improve_rilton_data' >/dev/null 2>&1 && return 0
    local mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${mb:-0}" -gt 8000 ] && return 0
    return 1
}
while gpu_busy; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "  waiting (GPU=${mb}MB) $(date +%H:%M:%S)" | tee -a $LOG
    sleep 30
done
cleanup_ckpts
free_status "phase0-done"

# ============================================================
# PHASE 1 — prepare truncated Rilton data and improved v2 data
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 1: prepare data (truncate + improve) ===" | tee -a $LOG

# Truncate the ORIGINAL rilton train (for the proper rilton-only adapter)
if [ ! -f "$RIL_TRAIN_T" ]; then
    $PY $TRUNCATE --input "$RIL_TRAIN" --output "$RIL_TRAIN_T" --max-chars 3000 | tee -a $LOG
else
    echo "  [skip] $RIL_TRAIN_T already exists" | tee -a $LOG
fi

# Improve Rilton data (generate length-balanced replacements). This needs GPU.
if [ ! -f "$RIL_TRAIN_V2" ]; then
    echo "  generating improved Rilton pairs (~750 generations, ~30 min)…" | tee -a $LOG
    $PY $IMPROVE --input "$RIL_TRAIN" --output "$RIL_TRAIN_V2" > /tmp/improve_data.log 2>&1
    tail -3 /tmp/improve_data.log | tee -a $LOG
else
    echo "  [skip] $RIL_TRAIN_V2 already exists" | tee -a $LOG
fi

# Truncate the improved v2 file
if [ ! -f "$RIL_TRAIN_V2_T" ] && [ -f "$RIL_TRAIN_V2" ]; then
    $PY $TRUNCATE --input "$RIL_TRAIN_V2" --output "$RIL_TRAIN_V2_T" --max-chars 3000 | tee -a $LOG
fi

# Build combined-v2 truncated: CJB train (untouched, short prompts) + truncated Rilton v2
if [ ! -f "$COMBINED_V2_T" ] && [ -f "$RIL_TRAIN_V2_T" ]; then
    cat "$CJB_TRAIN" "$RIL_TRAIN_V2_T" > "$COMBINED_V2_T"
    echo "  built combined_v2_truncated: $(wc -l < $COMBINED_V2_T) pairs" | tee -a $LOG
fi

# ============================================================
# PHASE 2 — train Rilton-only and Combined-v2 with PROVEN settings on truncated data
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 2: train improved adapters (proven settings) ===" | tee -a $LOG
do_train_dpo "rilton-only" "$ADAP_RIL_V1" "$RIL_TRAIN_T"   42
do_train_dpo "rilton-v2"   "$ADAP_RIL_V2" "$RIL_TRAIN_V2_T" 42
do_train_dpo "combined-v2" "$ADAP_COMB_V2" "$COMBINED_V2_T" 42

# ============================================================
# PHASE 3 — 3x3 transfer matrix (use existing CJB & combined; add rilton-only)
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 3: 3x3 transfer matrix ===" | tee -a $LOG
eval_triple "rilton" "$ADAP_RIL_V1" "$OUT_3X3"

# ============================================================
# PHASE 4 — 3x3 v2 transfer matrix (improved adapters)
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 4: 3x3 transfer matrix v2 (improved Rilton data) ===" | tee -a $LOG
eval_triple "rilton-v2"   "$ADAP_RIL_V2"  "$OUT_3X3V2"
eval_triple "combined-v2" "$ADAP_COMB_V2" "$OUT_3X3V2"

# ============================================================
# PHASE 5 — multi-seed combined (seeds 1, 2) on ORIGINAL combined corpus
# (matches seed 0 = existing ministral-8b/dpo_adapter for direct comparison)
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 5: multi-seed combined ===" | tee -a $LOG
do_train_dpo "combined-s1" "$ADAP_COMB_S1" "$COMBINED_TRAIN" 1
eval_triple "combined-s1" "$ADAP_COMB_S1" "$OUT_SEEDS"
do_train_dpo "combined-s2" "$ADAP_COMB_S2" "$COMBINED_TRAIN" 2
eval_triple "combined-s2" "$ADAP_COMB_S2" "$OUT_SEEDS"

# ============================================================
# PHASE 6 — SFT ablation on combined
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase 6: SFT ablation ===" | tee -a $LOG
do_train_sft "sft-combined" "$ADAP_SFT" "$COMBINED_TRAIN" 42
eval_triple "sft" "$ADAP_SFT" "$OUT_SFT"

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "" | tee -a $LOG
echo "=== FINAL SUMMARY ===" | tee -a $LOG
for outdir in "$OUT_3X3" "$OUT_3X3V2" "$OUT_SEEDS" "$OUT_SFT"; do
    [ -d "$outdir" ] || continue
    echo "--- $(basename $outdir) ---" | tee -a $LOG
    for f in $outdir/*.csv; do
        [ -f "$f" ] || continue
        ovr=$(awk -F, '/^overall,/ {print $4}' "$f")
        mac=$(awk -F, '/^macro,/ {print $4}' "$f")
        lc=$(awk -F, '/^length_controlled,/ {print $4}' "$f")
        tag=$(basename "$f" .csv)
        printf "%-30s overall=%-8s macro=%-8s LC=%-8s\n" "$tag" "${ovr:-N/A}" "${mac:-N/A}" "${lc:-N/A}" | tee -a $LOG
    done
done
echo "" | tee -a $LOG
echo "==== overnight pipeline finished: $(date) ====" | tee -a $LOG
