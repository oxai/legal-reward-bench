#!/bin/bash
# 3x3 transfer matrix on Ministral-8B: {raw, CJB, Rilton, combined} x {CJB, Rilton, BarExam}
# All evals use eval_dpo.py with the new metrics (CI, mean_margin, macro, length_controlled).
set -e
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
OUT=$WORK/pipeline/06_eval/outputs/transfer_3x3
mkdir -p $OUT

BASE=mistralai/Ministral-8B-Instruct-2410
RIL_ADAPTER=/workspace/adapters/ministral-8b-rilton
CJB_ADAPTER=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
COMB_ADAPTER=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter

CJB_TEST=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl
RIL_TEST=$WORK/data/rilton/pairs_test.jsonl
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl
PY=python3

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 PAIRS=$4 OUT_CSV=$5 MAX=$6
    if [ -f "$OUT_CSV" ]; then echo "  [skip] $TAG"; return; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local LOG="${OUT_CSV%.csv}.log"
    local PJSON="${OUT_CSV%.csv}_pairs.jsonl"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="--base-model $BASEARG"
    echo "  [run] $TAG"
    $PY $EVAL --model "$MODEL" --source pipeline --pairs "$PAIRS" \
        --output "$OUT_CSV" --pairs-output "$PJSON" --max-length $MAX $EXTRA \
        > "$LOG" 2>&1 \
        && echo "    ok: $(grep -E '^overall|^macro|^length_controlled' $OUT_CSV | tr '\n' ' ')" \
        || echo "    FAILED: $LOG"
}

# Eval on CJB-test (max-length 4096)
do_eval "raw->CJB"        "$BASE"          ""     "$CJB_TEST" "$OUT/raw__cjb.csv"        4096
do_eval "cjb->CJB"        "$CJB_ADAPTER"   "$BASE" "$CJB_TEST" "$OUT/cjb__cjb.csv"        4096
do_eval "rilton->CJB"     "$RIL_ADAPTER"   "$BASE" "$CJB_TEST" "$OUT/rilton__cjb.csv"     4096
do_eval "combined->CJB"   "$COMB_ADAPTER"  "$BASE" "$CJB_TEST" "$OUT/combined__cjb.csv"   4096

# Eval on Rilton-test (max-length 8192 — long prompts)
do_eval "raw->Rilton"        "$BASE"         ""     "$RIL_TEST" "$OUT/raw__rilton.csv"      8192
do_eval "cjb->Rilton"        "$CJB_ADAPTER"  "$BASE" "$RIL_TEST" "$OUT/cjb__rilton.csv"      8192
do_eval "rilton->Rilton"     "$RIL_ADAPTER"  "$BASE" "$RIL_TEST" "$OUT/rilton__rilton.csv"   8192
do_eval "combined->Rilton"   "$COMB_ADAPTER" "$BASE" "$RIL_TEST" "$OUT/combined__rilton.csv" 8192

# Eval on Bar Exam (short answers, max-length 4096 is plenty)
do_eval "raw->BarExam"       "$BASE"         ""     "$BAR_TEST" "$OUT/raw__barexam.csv"      4096
do_eval "cjb->BarExam"       "$CJB_ADAPTER"  "$BASE" "$BAR_TEST" "$OUT/cjb__barexam.csv"      4096
do_eval "rilton->BarExam"    "$RIL_ADAPTER"  "$BASE" "$BAR_TEST" "$OUT/rilton__barexam.csv"   4096
do_eval "combined->BarExam"  "$COMB_ADAPTER" "$BASE" "$BAR_TEST" "$OUT/combined__barexam.csv" 4096

echo ""
echo "========================================"
echo "3x3 TRANSFER MATRIX DONE: $(date)"
echo "========================================"
printf "%-30s %10s %10s %10s\n" "Train -> Eval" "Overall" "Macro" "LC"
for tr in raw cjb rilton combined; do
  for ev in cjb rilton barexam; do
    f=$OUT/${tr}__${ev}.csv
    if [ -f "$f" ]; then
      ovr=$(awk -F, '$1=="overall"{print $4}' $f)
      mac=$(awk -F, '$1=="macro"{print $4}' $f)
      lc=$(awk -F, '$1=="length_controlled"{print $4}' $f)
      printf "%-30s %10s %10s %10s\n" "$tr -> $ev" "$ovr" "$mac" "${lc:-N/A}"
    fi
  done
done
