"""
Sequential GPU job queue:
  1. DPO+surgery (L04+L15) per-split eval on CJB 400-pair test set → paper Table 1
  2. Rilton cross-benchmark eval (max-length=8192)
  3. DPO training on Rilton's pairs
  4. Rilton-DPO eval on CJB 400-pair test set
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
PYTHON     = sys.executable
DPO_EVAL   = ROOT / "pipeline" / "06_eval" / "eval_dpo.py"
DPO_TRAIN  = ROOT / "pipeline" / "05_reward_model" / "train_dpo.py"
OUT        = ROOT / "pipeline" / "06_eval" / "outputs"
RILTON_OUT = OUT / "rilton_eval"

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
DPO_ADAPTER = str(ROOT / "pipeline/06_eval/outputs/model_sweep/llama3.1-8b/dpo_adapter")
RILTON_DPO  = str(ROOT / "pipeline/05_reward_model/outputs/dpo_rilton")
CJB_TEST    = str(ROOT / "pipeline/06_eval/outputs/model_sweep/cjb_test.jsonl")
RILTON_ALL  = str(ROOT / "data/rilton/pairs_all.jsonl")


def _run(cmd: list, log: Path) -> bool:
    env = {**os.environ, "PYTHONUTF8": "1"}
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {log.stem}", flush=True)
    with log.open("w", encoding="utf-8") as lf:
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=lf, stderr=lf).returncode
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"    {status} — {log}", flush=True)
    return rc == 0


def read_csv(path: Path) -> dict[str, float]:
    import csv
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["split"]] = float(row["reward_accuracy"])
    return result


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. DPO+surgery per-split on the 400-pair CJB test set              #
    # ------------------------------------------------------------------ #
    surgery_csv = OUT / "surgery_cjb_l04l15.csv"
    if not surgery_csv.exists():
        _run([
            PYTHON, DPO_EVAL,
            "--model", DPO_ADAPTER,
            "--base-model", BASE_MODEL,
            "--source", "pipeline",
            "--pairs", CJB_TEST,
            "--output", str(surgery_csv),
            "--qlora",
            "--spectral-layer", "4,15",
            "--spectral-alpha", "0.2,0.1",
        ], OUT / "surgery_cjb_l04l15.log")
    else:
        print(f"[skip] surgery_cjb_l04l15.csv", flush=True)

    if surgery_csv.exists():
        res = read_csv(surgery_csv)
        print("\nSurgery CJB results (400-pair test set):")
        for k, v in sorted(res.items(), key=lambda x: (x[0] != "overall", x[0])):
            print(f"  {k:<30} {v:.4f}")

    # ------------------------------------------------------------------ #
    # 2. Rilton cross-benchmark eval                                      #
    # ------------------------------------------------------------------ #
    RILTON_CONFIGS = [
        ("raw",             BASE_MODEL,  BASE_MODEL, None,   None),
        ("dpo",             DPO_ADAPTER, BASE_MODEL, None,   None),
        ("dpo_surg_l04l15", DPO_ADAPTER, BASE_MODEL, "4,15", "0.2,0.1"),
    ]
    RILTON_OUT.mkdir(parents=True, exist_ok=True)
    for slug, model, base, layers, alphas in RILTON_CONFIGS:
        out = RILTON_OUT / f"{slug}.csv"
        if out.exists():
            print(f"  [skip] rilton/{slug}", flush=True)
            continue
        cmd = [
            PYTHON, DPO_EVAL,
            "--model", model,
            "--source", "pipeline",
            "--pairs", RILTON_ALL,
            "--output", str(out),
            "--max-length", "6144",
            "--limit", "300",
            "--qlora",
        ]
        if model != base:
            cmd += ["--base-model", base]
        if layers:
            cmd += ["--spectral-layer", layers, "--spectral-alpha", alphas]
        _run(cmd, RILTON_OUT / f"{slug}.log")

    # Print Rilton summary
    print("\nRilton eval summary:")
    for slug, *_ in RILTON_CONFIGS:
        res = read_csv(RILTON_OUT / f"{slug}.csv")
        ov = res.get("overall", None)
        print(f"  {slug:<25} {ov:.4f}" if ov else f"  {slug:<25} FAILED")

    # ------------------------------------------------------------------ #
    # 3. DPO training on Rilton's pairs                                   #
    # ------------------------------------------------------------------ #
    rilton_adapter = Path(RILTON_DPO)
    if not (rilton_adapter / "adapter_config.json").exists():
        _run([
            PYTHON, DPO_TRAIN,
            "--base-model", BASE_MODEL,
            "--source", "pipeline",
            "--pairs", str(ROOT / "data/rilton/pairs_train.jsonl"),
            "--output-dir", RILTON_DPO,
            "--qlora",
            "--epochs", "3",
            "--learning-rate", "5e-5",
        ], OUT / "rilton_dpo_train.log")
    else:
        print(f"[skip] Rilton DPO already trained", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Rilton-DPO on CJB 400-pair test set (reverse transfer)          #
    # ------------------------------------------------------------------ #
    if (rilton_adapter / "adapter_config.json").exists():
        reverse_csv = OUT / "rilton_dpo_on_cjb.csv"
        if not reverse_csv.exists():
            _run([
                PYTHON, DPO_EVAL,
                "--model", RILTON_DPO,
                "--base-model", BASE_MODEL,
                "--source", "pipeline",
                "--pairs", CJB_TEST,
                "--output", str(reverse_csv),
                "--qlora",
            ], OUT / "rilton_dpo_on_cjb.log")

        if reverse_csv.exists():
            res = read_csv(reverse_csv)
            print(f"\nRilton-DPO on CJB: {res.get('overall', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
