#!/bin/bash
# Housing QA follow-on: runs after the main overnight pipeline finishes.
# Adds Housing QA as a 4th eval benchmark across all ministral-8b conditions.
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
HOUSING=$WORK/data/housing_qa_test_pairs.jsonl
OUT=$WORK/pipeline/06_eval/outputs/housing_qa
mkdir -p $OUT

BASE=mistralai/Ministral-8B-Instruct-2410
PY=python3
LOG=/tmp/housing_followon.log

ADAP_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_RIL_V1=/workspace/adapters/ministral-8b-rilton-only
ADAP_RIL_V2=/workspace/adapters/ministral-8b-rilton-v2
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2
ADAP_COMB_S1=/workspace/adapters/ministral-8b-combined-s1
ADAP_COMB_S2=/workspace/adapters/ministral-8b-combined-s2
ADAP_SFT=/workspace/adapters/ministral-8b-combined-sft

echo "==== housing follow-on started: $(date) ====" | tee $LOG

# Wait for the main pipeline to finish + GPU to be free
echo "Waiting for main overnight pipeline to complete..." | tee -a $LOG
while ! grep -q 'overnight pipeline finished' /tmp/overnight_master.log 2>/dev/null; do
    sleep 60
done
echo "Main pipeline done. Waiting for GPU to be free..." | tee -a $LOG
while true; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${mb:-0}" -lt 8000 ] && break
    sleep 30
done

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 OUT_CSV=$4 PAIRS=${5:-$HOUSING} MAX=${6:-4096}
    if [ -f "$OUT_CSV" ]; then echo "  [skip] $TAG" | tee -a $LOG; return 0; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="--base-model $BASEARG"
    echo "  [run] $TAG" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source pipeline --pairs "$PAIRS" \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length $MAX $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        echo "    ok: $(awk -F, '/^overall,/ {print $4\" CI=[\"$5\",\"$6\"]\"}' $OUT_CSV)" | tee -a $LOG
    else
        echo "    FAILED" | tee -a $LOG
    fi
}

# First: re-run the 3 rilton-only transfer cells that were stale from the broken
# earlier adapter (Phase 3 skipped them because old CSVs existed). The new
# properly-trained rilton-only adapter must replace these.
CJB_TEST=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl
RIL_TEST=$WORK/data/rilton/pairs_test.jsonl
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl
T3X3=$WORK/pipeline/06_eval/outputs/transfer_3x3
echo "" | tee -a $LOG
echo "=== Re-running rilton-only on CJB/Rilton/BarExam (stale CSVs deleted) ===" | tee -a $LOG
do_eval "rilton-only/CJB"     "$ADAP_RIL_V1" "$BASE" "$T3X3/rilton__cjb.csv"     "$CJB_TEST" 4096
do_eval "rilton-only/Rilton"  "$ADAP_RIL_V1" "$BASE" "$T3X3/rilton__rilton.csv"  "$RIL_TEST" 8192
do_eval "rilton-only/BarExam" "$ADAP_RIL_V1" "$BASE" "$T3X3/rilton__barexam.csv" "$BAR_TEST" 4096

echo "" | tee -a $LOG
echo "=== Housing QA evals ===" | tee -a $LOG

# Run all available conditions
do_eval "raw"          "$BASE"          ""     "$OUT/raw.csv"
do_eval "cjb-only"     "$ADAP_CJB"      "$BASE" "$OUT/cjb_only.csv"
do_eval "combined"     "$ADAP_COMB"     "$BASE" "$OUT/combined.csv"
do_eval "rilton-only"  "$ADAP_RIL_V1"   "$BASE" "$OUT/rilton_only.csv"
do_eval "rilton-v2"    "$ADAP_RIL_V2"   "$BASE" "$OUT/rilton_v2.csv"
do_eval "combined-v2"  "$ADAP_COMB_V2"  "$BASE" "$OUT/combined_v2.csv"
do_eval "combined-s1"  "$ADAP_COMB_S1"  "$BASE" "$OUT/combined_s1.csv"
do_eval "combined-s2"  "$ADAP_COMB_S2"  "$BASE" "$OUT/combined_s2.csv"
do_eval "sft-combined" "$ADAP_SFT"      "$BASE" "$OUT/sft_combined.csv"

echo "" | tee -a $LOG
echo "=== HOUSING QA SUMMARY ===" | tee -a $LOG
for f in $OUT/*.csv; do
    [ -f "$f" ] || continue
    ovr=$(awk -F, '/^overall,/ {print $4}' "$f")
    lc=$(awk -F, '/^length_controlled,/ {print $4}' "$f")
    mar=$(awk -F, '/^overall,/ {print $7}' "$f")
    tag=$(basename "$f" .csv)
    printf "%-20s overall=%-8s LC=%-8s margin=%-8s\n" "$tag" "${ovr:-N/A}" "${lc:-N/A}" "${mar:-N/A}" | tee -a $LOG
done
echo "==== housing follow-on finished: $(date) ====" | tee -a $LOG
