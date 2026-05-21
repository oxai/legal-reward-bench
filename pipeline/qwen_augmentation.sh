#!/bin/bash
# Train Qwen-9B + Qwen-27B on the rilton-v2 (length-augmented) and combined-v2
# corpora, then evaluate each on augmented Rilton + Bar Exam + Housing QA.
# Closes the "augmentation generalizes across model families" gap.
#
# Proven training settings: batch=4, grad_accum=4, default max_length=1024,
# no grad_checkpointing. Same as the original combined adapter run.
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
TRAIN_DPO=$WORK/pipeline/05_reward_model/train_dpo.py
EVAL=$WORK/pipeline/06_eval/eval_dpo.py

RIL_V2_TRAIN=$WORK/data/rilton/pairs_train_v2_truncated.jsonl
COMBINED_V2_TRAIN=$WORK/pipeline/06_eval/outputs/model_sweep/combined_train_v2_truncated.jsonl
RIL_TEST_V2=$WORK/data/rilton/pairs_test_v2.jsonl
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl
HOU_TEST=$WORK/data/housing_qa_test_pairs.jsonl

OUT=$WORK/pipeline/06_eval/outputs/qwen_v2
mkdir -p $OUT

Q9_BASE=Qwen/Qwen3.5-9B
Q27_BASE=Qwen/Qwen3.5-27B

PY=python3
LOG=/tmp/qwen_augmentation.log
echo "==== qwen-augmentation started: $(date) ====" | tee $LOG

cleanup_ckpts() {
    find /workspace/adapters -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
}

clear_cache() { rm -rf "$HF_HOME/hub/models--${1}" 2>/dev/null || true; }

free_status() {
    echo "  [status:$1] disk=$(df -h / | awk 'NR==2{print $4}') shm=$(df -h /dev/shm | awk 'NR==2{print $4}') GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')MB" | tee -a $LOG
}

do_train() {
    local TAG=$1 BASE=$2 OUT_DIR=$3 PAIRS=$4 SEED=${5:-42} QLORA=${6:-}
    if [ -f "$OUT_DIR/adapter_config.json" ]; then echo "  [skip-train] $TAG" | tee -a $LOG; return 0; fi
    cleanup_ckpts
    mkdir -p "$OUT_DIR"
    echo "  [train] $TAG (seed=$SEED qlora=$QLORA) -> $OUT_DIR" | tee -a $LOG
    free_status "before-$TAG"
    local EXTRA=""
    [ "$QLORA" = "1" ] && EXTRA="--qlora"
    $PY $TRAIN_DPO \
        --base-model "$BASE" --source pipeline --pairs "$PAIRS" \
        --output-dir "$OUT_DIR" --epochs 3 --batch-size 4 --grad-accum 4 \
        --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 --seed $SEED $EXTRA \
        > "$OUT_DIR/train.log" 2>&1
    rm -rf "$OUT_DIR"/checkpoint-* 2>/dev/null || true
    if [ -f "$OUT_DIR/adapter_config.json" ]; then
        local gn=$(tr '\r' '\n' < $OUT_DIR/train.log | grep -oP "'grad_norm':\s*'[^']+'" | tail -1)
        local tl=$(tr '\r' '\n' < $OUT_DIR/train.log | grep -oP "'train_loss':\s*'[^']+'" | tail -1)
        echo "    train done; $gn $tl" | tee -a $LOG
    else
        echo "    TRAIN FAILED — see $OUT_DIR/train.log" | tee -a $LOG
    fi
    free_status "after-$TAG"
}

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 PAIRS=$4 OUT_CSV=$5 MAX=${6:-4096} QLORA=${7:-}
    if [ -f "$OUT_CSV" ]; then echo "  [skip-eval] $TAG" | tee -a $LOG; return 0; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="$EXTRA --base-model $BASEARG"
    [ "$QLORA" = "1" ] && EXTRA="$EXTRA --qlora"
    echo "  [eval] $TAG" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source pipeline --pairs "$PAIRS" \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length $MAX $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        local ov=$(awk -F, '/^overall,/ {print $4}' "$OUT_CSV")
        echo "    ok: overall=$ov" | tee -a $LOG
    else
        echo "    FAILED" | tee -a $LOG
    fi
}

eval_triple() {
    local TAG=$1 MODEL=$2 BASE=$3 QLORA=$4 OUT_PREFIX=$5
    do_eval "$TAG/RilV2" "$MODEL" "$BASE" "$RIL_TEST_V2" "${OUT_PREFIX}__rilV2.csv" 8192 "$QLORA"
    do_eval "$TAG/Bar"   "$MODEL" "$BASE" "$BAR_TEST"    "${OUT_PREFIX}__bar.csv"   4096 "$QLORA"
    do_eval "$TAG/Hou"   "$MODEL" "$BASE" "$HOU_TEST"    "${OUT_PREFIX}__hou.csv"   4096 "$QLORA"
}

# Wait for any lingering GPU activity
while true; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${mb:-0}" -lt 8000 ] && break
    echo "  waiting for GPU (${mb}MB used) $(date +%H:%M:%S)" | tee -a $LOG
    sleep 60
done
free_status "phase0-done"

# ============================================================
# Qwen3.5-9B (Apache 2.0)
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase A: Qwen3.5-9B training + eval ===" | tee -a $LOG

ADAP_Q9_RIL_V2=/workspace/adapters/qwen3.5-9b-rilton-v2
ADAP_Q9_COMB_V2=/workspace/adapters/qwen3.5-9b-combined-v2

do_train "q9-rilton-v2"   "$Q9_BASE" "$ADAP_Q9_RIL_V2"   "$RIL_V2_TRAIN"     42 ""
do_train "q9-combined-v2" "$Q9_BASE" "$ADAP_Q9_COMB_V2"  "$COMBINED_V2_TRAIN" 42 ""

eval_triple "q9-rilton-v2"   "$ADAP_Q9_RIL_V2"  "$Q9_BASE" "" "$OUT/qwen9b__rilton_v2"
eval_triple "q9-combined-v2" "$ADAP_Q9_COMB_V2" "$Q9_BASE" "" "$OUT/qwen9b__combined_v2"

clear_cache "Qwen--Qwen3.5-9B"

# Aggressive cleanup before Qwen-27B (largest model, ~50GB download)
echo "  [flush] dropping all stale model caches before Qwen-27B" | tee -a $LOG
for d in /dev/shm/hf_cache/hub/models--*; do
    case "$(basename $d)" in
        models--Qwen--Qwen3.5-27B) ;;
        *) rm -rf "$d" 2>/dev/null ;;
    esac
done
free_status "before-q27"

# ============================================================
# Qwen3.5-27B (Apache 2.0, QLoRA)
# ============================================================
echo "" | tee -a $LOG
echo "=== Phase B: Qwen3.5-27B training + eval (QLoRA) ===" | tee -a $LOG

ADAP_Q27_RIL_V2=/workspace/adapters/qwen3.5-27b-rilton-v2
ADAP_Q27_COMB_V2=/workspace/adapters/qwen3.5-27b-combined-v2

do_train "q27-rilton-v2"   "$Q27_BASE" "$ADAP_Q27_RIL_V2"   "$RIL_V2_TRAIN"     42 "1"
do_train "q27-combined-v2" "$Q27_BASE" "$ADAP_Q27_COMB_V2"  "$COMBINED_V2_TRAIN" 42 "1"

eval_triple "q27-rilton-v2"   "$ADAP_Q27_RIL_V2"  "$Q27_BASE" "1" "$OUT/qwen27b__rilton_v2"
eval_triple "q27-combined-v2" "$ADAP_Q27_COMB_V2" "$Q27_BASE" "1" "$OUT/qwen27b__combined_v2"

clear_cache "Qwen--Qwen3.5-27B"

# ============================================================
# Summary
# ============================================================
echo "" | tee -a $LOG
echo "=== QWEN AUGMENTATION SUMMARY ===" | tee -a $LOG
for f in $OUT/*.csv; do
    [ -f "$f" ] || continue
    ovr=$(awk -F, '/^overall,/ {print $4}' "$f")
    mac=$(awk -F, '/^macro,/ {print $4}' "$f")
    tag=$(basename "$f" .csv)
    printf "%-32s overall=%-8s macro=%-8s\n" "$tag" "${ovr:-N/A}" "${mac:-N/A}" | tee -a $LOG
done
echo "==== qwen-augmentation finished: $(date) ====" | tee -a $LOG
