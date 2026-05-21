#!/bin/bash
# Consolidated post-pipeline follow-on. Runs ONCE after the main overnight
# pipeline finishes (Phases 5 + 6 complete). Order:
#   1. Augment Rilton TEST pairs so they are length-balanced (removes the
#      structural ~0.353 ceiling caused by 9-word refusal templates in the
#      original test set).
#   2. Eval every key Ministral adapter on the augmented Rilton test.
#   3. Eval every Ministral adapter on Housing QA (Stanford Reglab).
#   4. Eval Qwen3.5-9B (Apache 2.0) raw/cjb/combined on augmented Rilton, Bar
#      Exam, Housing.
#   5. Eval Qwen3.5-27B (Apache 2.0, QLoRA) raw/cjb/combined on same.
# All evals use eval_dpo.py so every CSV has Wilson 95% CI + mean margin +
# macro + length_controlled.
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
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl
HOU_TEST=$WORK/data/housing_qa_test_pairs.jsonl
CJB_TEST=$WORK/pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl

BASE_MIN=mistralai/Ministral-8B-Instruct-2410
Q9_BASE=Qwen/Qwen3.5-9B
Q27_BASE=Qwen/Qwen3.5-27B

ADAP_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b-cjb/dpo_adapter
ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_RIL_V1=/workspace/adapters/ministral-8b-rilton-only
ADAP_RIL_V2=/workspace/adapters/ministral-8b-rilton-v2
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2
ADAP_COMB_S1=/workspace/adapters/ministral-8b-combined-s1
ADAP_COMB_S2=/workspace/adapters/ministral-8b-combined-s2
ADAP_SFT=/workspace/adapters/ministral-8b-combined-sft

Q9_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-9b-cjb/dpo_adapter
Q9_COM=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-9b/dpo_adapter
Q27_CJB=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-27b-cjb/dpo_adapter
Q27_COM=$WORK/pipeline/06_eval/outputs/model_sweep/qwen3.5-27b/dpo_adapter

OUT_RIL_V2=$WORK/pipeline/06_eval/outputs/rilton_test_v2
OUT_HOU=$WORK/pipeline/06_eval/outputs/housing_qa
OUT_QWEN=$WORK/pipeline/06_eval/outputs/qwen_transfer
mkdir -p $OUT_RIL_V2 $OUT_HOU $OUT_QWEN

PY=python3
LOG=/tmp/post_pipeline.log
echo "==== post-pipeline started: $(date) ====" | tee $LOG

# Wait for the main pipeline
echo "Waiting for overnight pipeline to finish..." | tee -a $LOG
while ! grep -q 'overnight pipeline finished' /tmp/overnight_master.log 2>/dev/null; do
    sleep 60
done
echo "Overnight done. Waiting for GPU to be free..." | tee -a $LOG
while true; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${mb:-0}" -lt 8000 ] && break
    sleep 30
done

# 1) Augment Rilton test
if [ ! -f "$RIL_TEST_V2" ]; then
    echo "" | tee -a $LOG
    echo "=== Phase 1: augment Rilton test ===" | tee -a $LOG
    $PY $IMPROVE --input "$RIL_TEST" --output "$RIL_TEST_V2" > /tmp/augment_test.log 2>&1
    tail -3 /tmp/augment_test.log | tee -a $LOG
else
    echo "[skip] $RIL_TEST_V2 exists" | tee -a $LOG
fi

clear_cache() { rm -rf "$HF_HOME/hub/models--${1}" 2>/dev/null || true; }

do_eval() {
    local TAG=$1 MODEL=$2 BASEARG=$3 PAIRS=$4 OUT_CSV=$5 MAX=${6:-4096} QLORA=${7:-}
    if [ -f "$OUT_CSV" ]; then echo "  [skip] $TAG" | tee -a $LOG; return 0; fi
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASEARG" ] && EXTRA="$EXTRA --base-model $BASEARG"
    [ "$QLORA" = "1" ] && EXTRA="$EXTRA --qlora"
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

# 2A) SMOKE TEST FIRST: raw + rilton-v2 on augmented Rilton.
# If rilton-v2 isn't materially above raw, the augmentation doesn't translate
# at test time — we abort the expensive Qwen phases to save 75 min.
SMOKE_PASS=1
echo "" | tee -a $LOG
echo "=== Phase 2A: SMOKE TEST on augmented Rilton ===" | tee -a $LOG
do_eval "smoke/raw/RilV2"       "$BASE_MIN"     ""          "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__raw.csv"       8192
do_eval "smoke/rilton-v2/RilV2" "$ADAP_RIL_V2"  "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__rilton_v2.csv" 8192

raw_acc=$(awk -F, '/^overall,/ {print $4}' $OUT_RIL_V2/ministral__raw.csv 2>/dev/null)
rv2_acc=$(awk -F, '/^overall,/ {print $4}' $OUT_RIL_V2/ministral__rilton_v2.csv 2>/dev/null)
diff=$(awk -v a="$rv2_acc" -v b="$raw_acc" 'BEGIN{print a-b}')
echo "  smoke: raw=$raw_acc, rilton-v2=$rv2_acc, diff=$diff" | tee -a $LOG
# Need rilton-v2 at least +5pp above raw to consider augmentation worthwhile
if awk -v d="$diff" 'BEGIN{exit (d>=0.05)?0:1}'; then
    echo "  SMOKE PASS — proceeding with full evals" | tee -a $LOG
    SMOKE_PASS=1
else
    echo "  SMOKE FAIL — rilton-v2 not materially above raw. Aborting Qwen phases." | tee -a $LOG
    SMOKE_PASS=0
fi

# 2B) Remaining Ministral evals on augmented Rilton (cheap, always run)
echo "" | tee -a $LOG
echo "=== Phase 2B: remaining Ministral evals on augmented Rilton ===" | tee -a $LOG
do_eval "min/cjb/RilV2"          "$ADAP_CJB"      "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__cjb.csv"          8192
do_eval "min/combined/RilV2"     "$ADAP_COMB"     "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__combined.csv"     8192
do_eval "min/rilton-only/RilV2"  "$ADAP_RIL_V1"   "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__rilton_only.csv"  8192
do_eval "min/combined-v2/RilV2"  "$ADAP_COMB_V2"  "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__combined_v2.csv"  8192
do_eval "min/combined-s1/RilV2"  "$ADAP_COMB_S1"  "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__combined_s1.csv"  8192
do_eval "min/combined-s2/RilV2"  "$ADAP_COMB_S2"  "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__combined_s2.csv"  8192
do_eval "min/sft/RilV2"          "$ADAP_SFT"      "$BASE_MIN" "$RIL_TEST_V2" "$OUT_RIL_V2/ministral__sft.csv"          8192

# 3) Ministral on Housing QA
echo "" | tee -a $LOG
echo "=== Phase 3: Ministral on Housing QA ===" | tee -a $LOG
do_eval "min/raw/Hou"          "$BASE_MIN"      ""     "$HOU_TEST" "$OUT_HOU/ministral__raw.csv"
do_eval "min/cjb/Hou"          "$ADAP_CJB"      "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__cjb.csv"
do_eval "min/combined/Hou"     "$ADAP_COMB"     "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__combined.csv"
do_eval "min/rilton-only/Hou"  "$ADAP_RIL_V1"   "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__rilton_only.csv"
do_eval "min/rilton-v2/Hou"    "$ADAP_RIL_V2"   "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__rilton_v2.csv"
do_eval "min/combined-v2/Hou"  "$ADAP_COMB_V2"  "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__combined_v2.csv"
do_eval "min/combined-s1/Hou"  "$ADAP_COMB_S1"  "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__combined_s1.csv"
do_eval "min/combined-s2/Hou"  "$ADAP_COMB_S2"  "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__combined_s2.csv"
do_eval "min/sft/Hou"          "$ADAP_SFT"      "$BASE_MIN" "$HOU_TEST" "$OUT_HOU/ministral__sft.csv"

clear_cache "mistralai--Ministral-8B-Instruct-2410"

# 4) Qwen3.5-9B (Apache 2.0) on augmented Rilton, BarExam, Housing
# GATED by smoke test outcome — if augmentation didn't translate, skip.
if [ "$SMOKE_PASS" != "1" ]; then
    echo "" | tee -a $LOG
    echo "=== Phase 4: SKIPPED (smoke test failed) ===" | tee -a $LOG
    echo "=== Phase 5: SKIPPED (smoke test failed) ===" | tee -a $LOG
    echo "==== post-pipeline aborted early due to smoke fail: $(date) ====" | tee -a $LOG
    exit 0
fi

echo "" | tee -a $LOG
echo "=== Phase 4: Qwen3.5-9B transfer ===" | tee -a $LOG
do_eval "q9/raw/RilV2"      "$Q9_BASE"  ""        "$RIL_TEST_V2" "$OUT_QWEN/qwen9b__raw__rilV2.csv"      8192
do_eval "q9/raw/Bar"        "$Q9_BASE"  ""        "$BAR_TEST"    "$OUT_QWEN/qwen9b__raw__bar.csv"
do_eval "q9/raw/Hou"        "$Q9_BASE"  ""        "$HOU_TEST"    "$OUT_QWEN/qwen9b__raw__hou.csv"
do_eval "q9/cjb/RilV2"      "$Q9_CJB"   "$Q9_BASE" "$RIL_TEST_V2" "$OUT_QWEN/qwen9b__cjb__rilV2.csv"      8192
do_eval "q9/cjb/Bar"        "$Q9_CJB"   "$Q9_BASE" "$BAR_TEST"    "$OUT_QWEN/qwen9b__cjb__bar.csv"
do_eval "q9/cjb/Hou"        "$Q9_CJB"   "$Q9_BASE" "$HOU_TEST"    "$OUT_QWEN/qwen9b__cjb__hou.csv"
do_eval "q9/combined/RilV2" "$Q9_COM"   "$Q9_BASE" "$RIL_TEST_V2" "$OUT_QWEN/qwen9b__combined__rilV2.csv" 8192
do_eval "q9/combined/Bar"   "$Q9_COM"   "$Q9_BASE" "$BAR_TEST"    "$OUT_QWEN/qwen9b__combined__bar.csv"
do_eval "q9/combined/Hou"   "$Q9_COM"   "$Q9_BASE" "$HOU_TEST"    "$OUT_QWEN/qwen9b__combined__hou.csv"

clear_cache "Qwen--Qwen3.5-9B"

# Aggressive cleanup before Qwen3.5-27B (largest model)
echo "  [flush] dropping stale HF caches before Qwen-27B" | tee -a $LOG
for d in /dev/shm/hf_cache/hub/models--*; do
    case "$(basename $d)" in
        models--Qwen--Qwen3.5-27B) ;;
        *) rm -rf "$d" 2>/dev/null ;;
    esac
done
echo "  [flush] /dev/shm now $(df -h /dev/shm | awk 'NR==2{print $4}') free" | tee -a $LOG

# 5) Qwen3.5-27B (QLoRA)
echo "" | tee -a $LOG
echo "=== Phase 5: Qwen3.5-27B transfer (QLoRA) ===" | tee -a $LOG
do_eval "q27/raw/RilV2"      "$Q27_BASE"  ""         "$RIL_TEST_V2" "$OUT_QWEN/qwen27b__raw__rilV2.csv"      8192 "1"
do_eval "q27/raw/Bar"        "$Q27_BASE"  ""         "$BAR_TEST"    "$OUT_QWEN/qwen27b__raw__bar.csv"        4096 "1"
do_eval "q27/raw/Hou"        "$Q27_BASE"  ""         "$HOU_TEST"    "$OUT_QWEN/qwen27b__raw__hou.csv"        4096 "1"
do_eval "q27/cjb/RilV2"      "$Q27_CJB"   "$Q27_BASE" "$RIL_TEST_V2" "$OUT_QWEN/qwen27b__cjb__rilV2.csv"      8192 "1"
do_eval "q27/cjb/Bar"        "$Q27_CJB"   "$Q27_BASE" "$BAR_TEST"    "$OUT_QWEN/qwen27b__cjb__bar.csv"        4096 "1"
do_eval "q27/cjb/Hou"        "$Q27_CJB"   "$Q27_BASE" "$HOU_TEST"    "$OUT_QWEN/qwen27b__cjb__hou.csv"        4096 "1"
do_eval "q27/combined/RilV2" "$Q27_COM"   "$Q27_BASE" "$RIL_TEST_V2" "$OUT_QWEN/qwen27b__combined__rilV2.csv" 8192 "1"
do_eval "q27/combined/Bar"   "$Q27_COM"   "$Q27_BASE" "$BAR_TEST"    "$OUT_QWEN/qwen27b__combined__bar.csv"   4096 "1"
do_eval "q27/combined/Hou"   "$Q27_COM"   "$Q27_BASE" "$HOU_TEST"    "$OUT_QWEN/qwen27b__combined__hou.csv"   4096 "1"

clear_cache "Qwen--Qwen3.5-27B"

# Summary
echo "" | tee -a $LOG
echo "=== FINAL POST-PIPELINE SUMMARY ===" | tee -a $LOG
for d in "$OUT_RIL_V2" "$OUT_HOU" "$OUT_QWEN"; do
    echo "--- $(basename $d) ---" | tee -a $LOG
    for f in $d/*.csv; do
        [ -f "$f" ] || continue
        ovr=$(awk -F, '/^overall,/ {print $4}' "$f")
        lo=$(awk -F, '/^overall,/ {print $5}' "$f")
        hi=$(awk -F, '/^overall,/ {print $6}' "$f")
        mac=$(awk -F, '/^macro,/ {print $4}' "$f")
        tag=$(basename "$f" .csv)
        printf "%-32s %-22s macro=%s\n" "$tag" "$ovr [$lo,$hi]" "${mac:-N/A}" | tee -a $LOG
    done
done
echo "==== post-pipeline finished: $(date) ====" | tee -a $LOG
