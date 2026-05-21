#!/bin/bash
# EMNLP sprint: length-norm eval + completeness upsampling.
# Waits for GPU to be free after all prior jobs finish.
set -u
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export PYTHONPATH=/home/valentin/work/legal-reward-modelling
export PYTHONUTF8=1
mkdir -p /dev/shm/tmp /workspace/adapters

WORK=/home/valentin/work/legal-reward-modelling
EVAL=$WORK/pipeline/06_eval/eval_dpo.py
TRAIN=$WORK/pipeline/05_reward_model/train_dpo.py
COMBINED_V2=$WORK/pipeline/06_eval/outputs/model_sweep/combined_train_v2_truncated.jsonl
RIL_TEST_V2=$WORK/data/rilton/pairs_test_v2.jsonl
BAR_TEST=$WORK/data/barexam_test_pairs.jsonl
HOU_TEST=$WORK/data/housing_qa_test_pairs.jsonl

OUT=$WORK/pipeline/06_eval/outputs/emnlp_sprint
mkdir -p $OUT

BASE=mistralai/Ministral-8B-Instruct-2410
PY=python3
LOG=/tmp/emnlp_sprint.log
echo "==== emnlp-sprint queued: $(date) ====" | tee $LOG

# ── helpers ────────────────────────────────────────────────────────────────────
wait_gpu() {
    local thresh=${1:-8000}
    echo "  [wait-gpu] <${thresh}MB" | tee -a $LOG
    while true; do
        mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        [ "${mb:-9999}" -lt "$thresh" ] && break
        echo "    GPU: ${mb}MB $(date +%H:%M:%S)" | tee -a $LOG
        sleep 60
    done
}

free_status() {
    echo "  [mem] GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')MB shm=$(df -h /dev/shm | awk 'NR==2{print $4}')" | tee -a $LOG
}

cleanup() {
    find /workspace/adapters -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
}

do_eval() {
    local TAG=$1 MODEL=$2 BASE_ARG=$3 SRC=$4 PAIRS=$5 OUT_CSV=$6 MAX=${7:-4096} LNORM=${8:-}
    [ -f "$OUT_CSV" ] && { echo "  [skip] $TAG" | tee -a $LOG; return 0; }
    mkdir -p "$(dirname $OUT_CSV)"
    local EXTRA=""
    [ -n "$BASE_ARG" ] && EXTRA="$EXTRA --base-model $BASE_ARG"
    [ -n "$PAIRS" ]    && EXTRA="$EXTRA --pairs $PAIRS"
    [ -n "$LNORM" ]    && EXTRA="$EXTRA --length-normalize"
    echo "  [eval] $TAG" | tee -a $LOG
    $PY $EVAL --model "$MODEL" --source "$SRC" \
        --output "$OUT_CSV" --pairs-output "${OUT_CSV%.csv}_pairs.jsonl" \
        --max-length $MAX $EXTRA > "${OUT_CSV%.csv}.log" 2>&1
    if [ -f "$OUT_CSV" ]; then
        ov=$(awk -F, '/^overall,/ {print $4}' "$OUT_CSV")
        cqa=$(awk -F, '/^completeness_qa,/ {print $4}' "$OUT_CSV")
        csu=$(awk -F, '/^completeness_summ,/ {print $4}' "$OUT_CSV")
        echo "    ok: overall=$ov  c_qa=$cqa  c_summ=$csu" | tee -a $LOG
    else
        echo "    FAILED — see ${OUT_CSV%.csv}.log" | tee -a $LOG
    fi
}

do_train() {
    local TAG=$1 PAIRS=$2 OUT_DIR=$3
    [ -f "$OUT_DIR/adapter_config.json" ] && { echo "  [skip-train] $TAG" | tee -a $LOG; return 0; }
    cleanup; mkdir -p "$OUT_DIR"
    echo "  [train] $TAG -> $OUT_DIR" | tee -a $LOG
    free_status
    $PY $TRAIN --base-model "$BASE" --source pipeline --pairs "$PAIRS" \
        --output-dir "$OUT_DIR" --epochs 3 --batch-size 4 --grad-accum 4 \
        --learning-rate 2e-5 --lora-r 32 --lora-alpha 64 --seed 42 \
        > "$OUT_DIR/train.log" 2>&1
    cleanup
    if [ -f "$OUT_DIR/adapter_config.json" ]; then
        tl=$(tr '\r' '\n' < "$OUT_DIR/train.log" | grep -oP "'train_loss':\s*'[^']+'" | tail -1)
        echo "    done: $tl" | tee -a $LOG
    else
        echo "    TRAIN FAILED — see $OUT_DIR/train.log" | tee -a $LOG
    fi
    free_status
}

# ── wait for prior jobs ────────────────────────────────────────────────────────
echo "Waiting for prior jobs (v3, DeepSeek) to finish..." | tee -a $LOG
wait_gpu 8000
# Extra safety: wait until no train_dpo.py is running
while pgrep -f "train_dpo.py" > /dev/null 2>&1; do
    echo "  train_dpo.py still running, waiting 60s..." | tee -a $LOG
    sleep 60
done
wait_gpu 8000
echo "GPU clear, starting sprint. $(date)" | tee -a $LOG
free_status

# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — Length-norm eval on CJB
# Key question: does completeness jump from below-chance to above-chance?
# Raw + DPO-combined + DPO-combined-v2. ~25 min total. Decision point.
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a $LOG
echo "=== PHASE A: Length-norm eval (Ministral-8B on CJB) ===" | tee -a $LOG

ADAP_COMB=$WORK/pipeline/06_eval/outputs/model_sweep/ministral-8b/dpo_adapter
ADAP_COMB_V2=/workspace/adapters/ministral-8b-combined-v2

do_eval "A/raw/sum"   "$BASE"         ""      contextual_judge_bench "" "$OUT/lensum__raw__cjb.csv"    4096 ""
do_eval "A/raw/lnorm" "$BASE"         ""      contextual_judge_bench "" "$OUT/lnnorm__raw__cjb.csv"   4096 "1"
do_eval "A/dpo/sum"   "$ADAP_COMB"   "$BASE"  contextual_judge_bench "" "$OUT/lensum__dpo__cjb.csv"   4096 ""
do_eval "A/dpo/lnorm" "$ADAP_COMB"   "$BASE"  contextual_judge_bench "" "$OUT/lnnorm__dpo__cjb.csv"  4096 "1"
do_eval "A/v2/lnorm"  "$ADAP_COMB_V2" "$BASE" contextual_judge_bench "" "$OUT/lnnorm__dpov2__cjb.csv" 4096 "1"

echo "" | tee -a $LOG
echo "=== PHASE A: COMPLETENESS SNAPSHOT ===" | tee -a $LOG
for f in $OUT/len*.csv $OUT/lnn*.csv; do
    [ -f "$f" ] || continue
    ov=$(awk  -F, '/^overall,/           {print $4}' "$f")
    cqa=$(awk -F, '/^completeness_qa,/   {print $4}' "$f")
    csu=$(awk -F, '/^completeness_summ,/ {print $4}' "$f")
    printf "%-48s  overall=%-6s  c_qa=%-6s  c_summ=%-6s\n" \
        "$(basename $f .csv)" "$ov" "$cqa" "$csu" | tee -a $LOG
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE B — Completeness 10x upsampling
# Generates corpus, trains Ministral-8B, evals with sum + length-norm.
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a $LOG
echo "=== PHASE B: Completeness 10x upsampling ===" | tee -a $LOG

COMP10X=$OUT/combined_v2_comp10x.jsonl
if [ ! -f "$COMP10X" ]; then
    echo "  Generating completeness-10x corpus..." | tee -a $LOG
    $PY - <<PYEOF
import json, random
src = "$COMBINED_V2"
dst = "$COMP10X"
pairs, completeness = [], []
with open(src) as f:
    for line in f:
        r = json.loads(line)
        pairs.append(r)
        if "completeness" in r.get("split", ""):
            completeness.append(r)
upsampled = pairs + completeness * 9
random.seed(42)
random.shuffle(upsampled)
with open(dst, "w") as f:
    for r in upsampled:
        f.write(json.dumps(r) + "\n")
print(f"comp10x corpus: {len(upsampled)} pairs ({len(completeness)} completeness base x10)")
PYEOF
fi

ADAP_COMP10X=/workspace/adapters/ministral-8b-comp10x
do_train "ministral-comp10x" "$COMP10X" "$ADAP_COMP10X"

do_eval "B/comp10x/sum"      "$ADAP_COMP10X" "$BASE" contextual_judge_bench "" "$OUT/lensum__comp10x__cjb.csv"   4096 ""
do_eval "B/comp10x/lnorm"    "$ADAP_COMP10X" "$BASE" contextual_judge_bench "" "$OUT/lnnorm__comp10x__cjb.csv"  4096 "1"
do_eval "B/comp10x/ril"      "$ADAP_COMP10X" "$BASE" pipeline "$RIL_TEST_V2" "$OUT/lensum__comp10x__ril.csv"  8192 ""
do_eval "B/comp10x/bar"      "$ADAP_COMP10X" "$BASE" pipeline "$BAR_TEST"    "$OUT/lensum__comp10x__bar.csv"  4096 ""
do_eval "B/comp10x/hou"      "$ADAP_COMP10X" "$BASE" pipeline "$HOU_TEST"    "$OUT/lensum__comp10x__hou.csv"  4096 ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C — Best existing adapter (combined-v2) length-norm on all benchmarks
# Zero training cost. Shows if length-norm generalises across eval sets.
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a $LOG
echo "=== PHASE C: combined-v2 length-norm sweep ===" | tee -a $LOG

do_eval "C/v2/ril/lnorm" "$ADAP_COMB_V2" "$BASE" pipeline "$RIL_TEST_V2" "$OUT/lnnorm__dpov2__ril.csv"  8192 "1"
do_eval "C/v2/bar/lnorm" "$ADAP_COMB_V2" "$BASE" pipeline "$BAR_TEST"    "$OUT/lnnorm__dpov2__bar.csv"  4096 "1"
do_eval "C/v2/hou/lnorm" "$ADAP_COMB_V2" "$BASE" pipeline "$HOU_TEST"    "$OUT/lnnorm__dpov2__hou.csv"  4096 "1"

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a $LOG
echo "=== EMNLP SPRINT FINAL SUMMARY ===" | tee -a $LOG
printf "%-50s  %-8s  %-8s  %-8s\n" "file" "overall" "c_qa" "c_summ" | tee -a $LOG
for f in $OUT/*.csv; do
    [ -f "$f" ] || continue
    ov=$(awk  -F, '/^overall,/           {print $4}' "$f")
    cqa=$(awk -F, '/^completeness_qa,/   {print $4}' "$f")
    csu=$(awk -F, '/^completeness_summ,/ {print $4}' "$f")
    printf "%-50s  %-8s  %-8s  %-8s\n" "$(basename $f .csv)" \
        "${ov:-N/A}" "${cqa:--}" "${csu:--}" | tee -a $LOG
done
echo "==== emnlp-sprint finished: $(date) ====" | tee -a $LOG
