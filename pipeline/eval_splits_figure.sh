#!/bin/bash
# Produce per-split CSVs for all 4 conditions needed for the AI4Law figures.
# Conditions: raw, CJB-only DPO, LRB-only DPO, combined-v2 DPO.
# Eval on CJB test set only (373 pairs, fast ~15 min per run).
# Run this AFTER overnight_pipeline.sh has finished (adapters in /workspace/adapters).
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
CJB_TEST=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl
BASE=mistralai/Ministral-8B-Instruct-2410
PY=python3
OUT=$WORK/pipeline/06_eval/outputs/splits_figure
mkdir -p $OUT
LOG=/tmp/eval_splits_figure.log
echo "==== eval_splits_figure started: $(date) ====" | tee $LOG

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 OUT_CSV=$4
    [ -f "$OUT_CSV" ] && { echo "  [skip] $TAG" | tee -a $LOG; return 0; }
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="--base-model $BASEARG"
    echo "  [eval] $TAG" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source contextual_judge_bench \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length 4096 $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        echo "    ok: $(awk -F, '/^overall,/{print "overall="$4}' $OUT_CSV)" | tee -a $LOG
    else
        echo "    FAILED — see ${OUT_CSV%.csv}.log" | tee -a $LOG
    fi
}

# Adapter paths
ADAP_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_LRB=/workspace/adapters/ministral-8b-rilton-only
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2

# Check which adapters exist
echo "Checking adapters..." | tee -a $LOG
for name in "CJB-only:$ADAP_CJB" "combined:$ADAP_COMB" "LRB-only:$ADAP_LRB" "combined-v2:$ADAP_COMB_V2"; do
    tag="${name%%:*}"; path="${name##*:}"
    [ -f "$path/adapter_config.json" ] && echo "  ✓ $tag" || echo "  ✗ $tag (not yet trained)" | tee -a $LOG
done

# Run evals (skip if adapter missing)
do_eval "raw"          "$BASE"        ""      "$OUT/ministral__raw__cjb.csv"
[ -f "$ADAP_CJB/adapter_config.json"     ] && do_eval "cjb-only"    "$ADAP_CJB"    "$BASE" "$OUT/ministral__cjb_only__cjb.csv"
[ -f "$ADAP_COMB/adapter_config.json"    ] && do_eval "combined"     "$ADAP_COMB"   "$BASE" "$OUT/ministral__combined__cjb.csv"
[ -f "$ADAP_LRB/adapter_config.json"     ] && do_eval "lrb-only"     "$ADAP_LRB"    "$BASE" "$OUT/ministral__lrb_only__cjb.csv"
[ -f "$ADAP_COMB_V2/adapter_config.json" ] && do_eval "combined-v2"  "$ADAP_COMB_V2" "$BASE" "$OUT/ministral__combined_v2__cjb.csv"

# Print per-split table
echo "" | tee -a $LOG
echo "=== PER-SPLIT RESULTS ===" | tee -a $LOG
printf "%-20s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n" \
    "condition" "overall" "comp_qa" "comp_su" "conc_qa" "conc_su" "faith_qa" "faith_su" "ref_ans" "ref_unans" | tee -a $LOG
echo "$(printf '%.0s-' {1..130})" | tee -a $LOG
for f in $OUT/*.csv; do
    [ -f "$f" ] || continue
    tag=$(basename "$f" .csv | sed 's/ministral__//;s/__cjb//')
    ov=$(awk   -F, '/^overall,/{print $4}'               "$f")
    cqa=$(awk  -F, '/^completeness_qa,/{print $4}'       "$f")
    csu=$(awk  -F, '/^completeness_summ,/{print $4}'     "$f")
    nqa=$(awk  -F, '/^conciseness_qa,/{print $4}'        "$f")
    nsu=$(awk  -F, '/^conciseness_summ,/{print $4}'      "$f")
    fqa=$(awk  -F, '/^faithfulness_qa,/{print $4}'       "$f")
    fsu=$(awk  -F, '/^faithfulness_summ,/{print $4}'     "$f")
    rans=$(awk -F, '/^refusal_answerable,/{print $4}'    "$f")
    run=$(awk  -F, '/^refusal_unanswerable,/{print $4}'  "$f")
    printf "%-20s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n" \
        "$tag" "$ov" "$cqa" "$csu" "$nqa" "$nsu" "$fqa" "$fsu" "$rans" "$run" | tee -a $LOG
done
echo "==== eval_splits_figure finished: $(date) ====" | tee -a $LOG
