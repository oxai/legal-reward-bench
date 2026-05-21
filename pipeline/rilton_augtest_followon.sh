#!/bin/bash
# Rilton-test-augmentation follow-on: runs after multimodel follow-on finishes.
# Augments the Rilton TEST pairs to be length-balanced (same approach as
# training-time augmentation), then re-evaluates every key adapter on the
# augmented test set. This removes the structural ~0.353 ceiling caused by
# the 9-word refusal templates in the original test pairs.
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
IMPROVE=$WORK/pipeline/04_pairs/improve_rilton_data.py
RIL_TEST=$WORK/data/rilton/pairs_test.jsonl
RIL_TEST_V2=$WORK/data/rilton/pairs_test_v2.jsonl

OUT=$WORK/pipeline/06_eval/outputs/rilton_test_v2
mkdir -p $OUT

BASE=mistralai/Ministral-8B-Instruct-2410
PY=python3
LOG=/tmp/rilton_augtest_followon.log
echo "==== rilton-augtest follow-on started: $(date) ====" | tee $LOG

# Wait for the multimodel follow-on to finish
echo "Waiting for multi-model follow-on to complete..." | tee -a $LOG
while ! grep -q 'multi-model follow-on finished' /tmp/multimodel_followon.log 2>/dev/null; do
    sleep 60
done
echo "Multi-model done. Waiting for GPU to be free..." | tee -a $LOG
while true; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${mb:-0}" -lt 8000 ] && break
    sleep 30
done

# Step 1: augment Rilton test pairs
if [ ! -f "$RIL_TEST_V2" ]; then
    echo "Augmenting Rilton test pairs (~5 min)..." | tee -a $LOG
    $PY $IMPROVE --input "$RIL_TEST" --output "$RIL_TEST_V2" > /tmp/augment_test.log 2>&1
    tail -3 /tmp/augment_test.log | tee -a $LOG
else
    echo "[skip] $RIL_TEST_V2 already exists" | tee -a $LOG
fi

clear_cache() { rm -rf "$HF_HOME/hub/models--${1}" 2>/dev/null || true; }

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 OUT_CSV=$4 QLORA=${5:-}
    if [ -f "$OUT_CSV" ]; then echo "  [skip] $TAG" | tee -a $LOG; return 0; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="$EXTRA --base-model $BASEARG"
    [ "$QLORA" = "1" ] && EXTRA="$EXTRA --qlora"
    echo "  [run] $TAG" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source pipeline --pairs "$RIL_TEST_V2" \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length 8192 $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        echo "    ok: $(awk -F, '/^overall,/ {print $4\" CI=[\"$5\",\"$6\"]\"}' $OUT_CSV)" | tee -a $LOG
    else
        echo "    FAILED" | tee -a $LOG
    fi
}

# Adapters (Ministral)
ADAP_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_RIL_V1=/workspace/adapters/ministral-8b-rilton-only
ADAP_RIL_V2=/workspace/adapters/ministral-8b-rilton-v2
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2
ADAP_COMB_S1=/workspace/adapters/ministral-8b-combined-s1
ADAP_COMB_S2=/workspace/adapters/ministral-8b-combined-s2
ADAP_SFT=/workspace/adapters/ministral-8b-combined-sft

echo "" | tee -a $LOG
echo "=== Ministral-8B on Rilton-test-v2 (length-balanced) ===" | tee -a $LOG
do_eval "ministral/raw"          "$BASE"          ""     "$OUT/ministral__raw.csv"
do_eval "ministral/cjb-only"     "$ADAP_CJB"      "$BASE" "$OUT/ministral__cjb.csv"
do_eval "ministral/combined"     "$ADAP_COMB"     "$BASE" "$OUT/ministral__combined.csv"
do_eval "ministral/rilton-only"  "$ADAP_RIL_V1"   "$BASE" "$OUT/ministral__rilton_only.csv"
do_eval "ministral/rilton-v2"    "$ADAP_RIL_V2"   "$BASE" "$OUT/ministral__rilton_v2.csv"
do_eval "ministral/combined-v2"  "$ADAP_COMB_V2"  "$BASE" "$OUT/ministral__combined_v2.csv"
do_eval "ministral/combined-s1"  "$ADAP_COMB_S1"  "$BASE" "$OUT/ministral__combined_s1.csv"
do_eval "ministral/combined-s2"  "$ADAP_COMB_S2"  "$BASE" "$OUT/ministral__combined_s2.csv"
do_eval "ministral/sft-combined" "$ADAP_SFT"      "$BASE" "$OUT/ministral__sft.csv"

clear_cache "mistralai--Ministral-8B-Instruct-2410"

# Qwen Apache-2.0 models
Q9_BASE=Qwen/Qwen3.5-9B
Q9_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-9b-cjb/dpo_adapter
Q9_COM=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-9b/dpo_adapter
echo "" | tee -a $LOG
echo "=== Qwen3.5-9B on Rilton-test-v2 ===" | tee -a $LOG
do_eval "qwen9b/raw"      "$Q9_BASE"  ""        "$OUT/qwen9b__raw.csv"
do_eval "qwen9b/cjb"      "$Q9_CJB"   "$Q9_BASE" "$OUT/qwen9b__cjb.csv"
do_eval "qwen9b/combined" "$Q9_COM"   "$Q9_BASE" "$OUT/qwen9b__combined.csv"
clear_cache "Qwen--Qwen3.5-9B"

Q27_BASE=Qwen/Qwen3.5-27B
Q27_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-27b-cjb/dpo_adapter
Q27_COM=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-27b/dpo_adapter
echo "" | tee -a $LOG
echo "=== Qwen3.5-27B on Rilton-test-v2 (QLoRA) ===" | tee -a $LOG
do_eval "qwen27b/raw"      "$Q27_BASE"  ""         "$OUT/qwen27b__raw.csv"      "1"
do_eval "qwen27b/cjb"      "$Q27_CJB"   "$Q27_BASE" "$OUT/qwen27b__cjb.csv"      "1"
do_eval "qwen27b/combined" "$Q27_COM"   "$Q27_BASE" "$OUT/qwen27b__combined.csv" "1"
clear_cache "Qwen--Qwen3.5-27B"

echo "" | tee -a $LOG
echo "=== RILTON-TEST-V2 SUMMARY ===" | tee -a $LOG
printf "%-30s %-22s %-22s\n" "Adapter" "overall (CI)" "macro" | tee -a $LOG
for f in $OUT/*.csv; do
    [ -f "$f" ] || continue
    ovr=$(awk -F, '/^overall,/ {print $4}' "$f")
    lo=$(awk -F, '/^overall,/ {print $5}' "$f")
    hi=$(awk -F, '/^overall,/ {print $6}' "$f")
    mac=$(awk -F, '/^macro,/ {print $4}' "$f")
    tag=$(basename "$f" .csv)
    printf "%-30s %-22s %-22s\n" "$tag" "$ovr [$lo,$hi]" "${mac:-N/A}" | tee -a $LOG
done
echo "==== rilton-augtest follow-on finished: $(date) ====" | tee -a $LOG
