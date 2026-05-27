"""
Compute statistical summaries for reward-model evaluation results.

Directory layout expected under --results-dir:
  <results-dir>/
    <model-name>/
      raw.csv                  # aggregate CSV (split, n, correct, reward_accuracy)
      dpo_cjb.csv
      dpo_combined.csv
      raw_pairs.jsonl          # optional per-pair JSONL {idx, split, correct}
      dpo_cjb_pairs.jsonl
      dpo_combined_pairs.jsonl

Outputs:
  - Formatted table printed to stdout
  - stats_summary.csv written to --results-dir

Usage:
  python pipeline/06_eval/compute_stats.py --results-dir pipeline/06_eval/outputs/model_sweep

  # With seed replicates for mean±std:
  python pipeline/06_eval/compute_stats.py \\
      --results-dir pipeline/06_eval/outputs/model_sweep \\
      --seed-dirs "llama3:outputs/seed0,outputs/seed1,outputs/seed2"
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2, norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS = ("raw", "dpo_cjb", "dpo_combined")
BOOTSTRAP_N = 10_000
BOOTSTRAP_RNG_SEED = 42
CI_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class AggStats(NamedTuple):
    n: int
    correct: int
    reward_accuracy: float


class BootstrapCI(NamedTuple):
    lower: float
    upper: float
    method: str  # "bootstrap" or "binomial-approx"


class PairwiseTest(NamedTuple):
    a_label: str
    b_label: str
    statistic: float
    p_value: float
    test: str  # "mcnemar" or "z-test"


class CohenH(NamedTuple):
    config_a: str
    config_b: str
    h: float


# ---------------------------------------------------------------------------
# CSV / JSONL loading
# ---------------------------------------------------------------------------


def _load_agg_csv(path: Path) -> dict[str, AggStats] | None:
    """Load aggregate CSV; return dict keyed by split, or None if missing."""
    if not path.exists():
        return None
    result: dict[str, AggStats] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                result[row["split"]] = AggStats(
                    n=int(row["n"]),
                    correct=int(row["correct"]),
                    reward_accuracy=float(row["reward_accuracy"]),
                )
            except (KeyError, ValueError):
                continue
    return result or None


def _load_pairs_jsonl(path: Path) -> list[int] | None:
    """
    Load per-pair JSONL; return list of 0/1 correct values (overall only),
    or None if the file is missing.
    """
    if not path.exists():
        return None
    corrects: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                corrects.append(int(obj["correct"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return corrects if corrects else None


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    corrects: list[int],
    *,
    n_iter: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_RNG_SEED,
    level: float = CI_LEVEL,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    arr = np.array(corrects, dtype=np.float32)
    n = len(arr)
    samples = rng.choice(arr, size=(n_iter, n), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - level
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return BootstrapCI(lower=lo, upper=hi, method="bootstrap")


def binomial_approx_ci(
    n: int,
    correct: int,
    *,
    level: float = CI_LEVEL,
) -> BootstrapCI:
    """Wilson score interval (more accurate than normal approx for small n)."""
    if n == 0:
        return BootstrapCI(lower=float("nan"), upper=float("nan"), method="binomial-approx")
    p = correct / n
    z = norm.ppf(1 - (1 - level) / 2)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return BootstrapCI(lower=max(0.0, centre - half), upper=min(1.0, centre + half), method="binomial-approx")


def compute_ci(
    corrects: list[int] | None,
    agg: AggStats | None,
) -> BootstrapCI | None:
    if corrects is not None:
        return bootstrap_ci(corrects)
    if agg is not None:
        return binomial_approx_ci(agg.n, agg.correct)
    return None


# ---------------------------------------------------------------------------
# Effect size: Cohen's h
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions. Sign: positive means p1 > p2."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(
    corrects_a: list[int],
    corrects_b: list[int],
    label_a: str,
    label_b: str,
) -> PairwiseTest | None:
    """
    McNemar's test (with continuity correction) on paired binary outcomes.
    Returns None if vectors differ in length or are empty.
    """
    if len(corrects_a) != len(corrects_b) or not corrects_a:
        return None
    a = np.array(corrects_a, dtype=np.int8)
    b = np.array(corrects_b, dtype=np.int8)
    # b=1, a=0  (b correct, a wrong)
    n_01 = int(np.sum((a == 0) & (b == 1)))
    # b=0, a=1
    n_10 = int(np.sum((a == 1) & (b == 0)))
    discordant = n_01 + n_10
    if discordant == 0:
        return PairwiseTest(
            a_label=label_a, b_label=label_b,
            statistic=0.0, p_value=1.0, test="mcnemar",
        )
    # McNemar with continuity correction
    stat = (abs(n_01 - n_10) - 1) ** 2 / discordant
    p = float(chi2.sf(stat, df=1))
    return PairwiseTest(
        a_label=label_a, b_label=label_b,
        statistic=float(stat), p_value=p, test="mcnemar",
    )


# ---------------------------------------------------------------------------
# Two-proportion z-test (fallback when pairs not available)
# ---------------------------------------------------------------------------


def two_prop_z_test(
    agg_a: AggStats,
    agg_b: AggStats,
    label_a: str,
    label_b: str,
) -> PairwiseTest:
    n1, k1 = agg_a.n, agg_a.correct
    n2, k2 = agg_b.n, agg_b.correct
    if n1 == 0 or n2 == 0:
        return PairwiseTest(label_a, label_b, float("nan"), float("nan"), "z-test")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return PairwiseTest(label_a, label_b, float("nan"), 1.0, "z-test")
    stat = (p1 - p2) / denom
    p_val = float(2 * norm.sf(abs(stat)))
    return PairwiseTest(label_a, label_b, float(stat), p_val, "z-test")


# ---------------------------------------------------------------------------
# Seed mean±std
# ---------------------------------------------------------------------------


def compute_seed_stats(
    model_label: str,
    seed_dirs: list[Path],
) -> dict[str, dict[str, float]]:
    """
    For each config in CONFIGS, load overall reward_accuracy from each seed dir
    and return {config: {mean, std}}.
    """
    out: dict[str, dict[str, float]] = {}
    for cfg in CONFIGS:
        accs: list[float] = []
        for sd in seed_dirs:
            csv_path = sd / f"{cfg}.csv"
            agg = _load_agg_csv(csv_path)
            if agg is None:
                continue
            overall = agg.get("overall")
            if overall is None:
                continue
            accs.append(overall.reward_accuracy)
        if accs:
            out[cfg] = {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)}
    return out


# ---------------------------------------------------------------------------
# Core analysis for a single model directory
# ---------------------------------------------------------------------------


class ModelResult(NamedTuple):
    model: str
    config: str
    split: str
    n: int
    correct: int
    reward_accuracy: float
    ci_lower: float | None
    ci_upper: float | None
    ci_method: str | None
    cohen_h_vs_raw: float | None


def analyse_model_dir(model_dir: Path) -> list[ModelResult]:
    """Return per-(config, split) ModelResult rows for one model directory."""
    results: list[ModelResult] = []
    model_name = model_dir.name

    # Load all agg CSVs and pair lists
    aggs: dict[str, dict[str, AggStats] | None] = {}
    pairs: dict[str, list[int] | None] = {}
    for cfg in CONFIGS:
        aggs[cfg] = _load_agg_csv(model_dir / f"{cfg}.csv")
        pairs[cfg] = _load_pairs_jsonl(model_dir / f"{cfg}_pairs.jsonl")

    raw_agg = aggs.get("raw")

    for cfg in CONFIGS:
        agg = aggs[cfg]
        if agg is None:
            continue
        pair_list = pairs[cfg]

        for split, stats in agg.items():
            # CI: use bootstrap from pairs if available (overall only), else Wilson
            if split == "overall" and pair_list is not None:
                ci = bootstrap_ci(pair_list)
            else:
                ci = binomial_approx_ci(stats.n, stats.correct)

            # Cohen's h vs raw baseline (overall split)
            h: float | None = None
            if cfg != "raw" and raw_agg is not None and split in raw_agg:
                h = cohens_h(stats.reward_accuracy, raw_agg[split].reward_accuracy)

            results.append(ModelResult(
                model=model_name,
                config=cfg,
                split=split,
                n=stats.n,
                correct=stats.correct,
                reward_accuracy=stats.reward_accuracy,
                ci_lower=ci.lower,
                ci_upper=ci.upper,
                ci_method=ci.method,
                cohen_h_vs_raw=h,
            ))

    return results


# ---------------------------------------------------------------------------
# Pairwise tests for one model directory
# ---------------------------------------------------------------------------


PAIRWISE_PAIRS = [
    ("raw", "dpo_cjb"),
    ("raw", "dpo_combined"),
    ("dpo_cjb", "dpo_combined"),
]


def pairwise_tests(model_dir: Path) -> list[PairwiseTest]:
    tests: list[PairwiseTest] = []
    aggs: dict[str, dict[str, AggStats] | None] = {}
    pair_lists: dict[str, list[int] | None] = {}
    for cfg in CONFIGS:
        aggs[cfg] = _load_agg_csv(model_dir / f"{cfg}.csv")
        pair_lists[cfg] = _load_pairs_jsonl(model_dir / f"{cfg}_pairs.jsonl")

    for cfg_a, cfg_b in PAIRWISE_PAIRS:
        agg_a, agg_b = aggs.get(cfg_a), aggs.get(cfg_b)
        pairs_a, pairs_b = pair_lists.get(cfg_a), pair_lists.get(cfg_b)
        if agg_a is None or agg_b is None:
            continue

        if pairs_a is not None and pairs_b is not None:
            test = mcnemar_test(pairs_a, pairs_b, cfg_a, cfg_b)
            if test is not None:
                tests.append(test)
        else:
            overall_a = agg_a.get("overall")
            overall_b = agg_b.get("overall")
            if overall_a is not None and overall_b is not None:
                tests.append(two_prop_z_test(overall_a, overall_b, cfg_a, cfg_b))

    return tests


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "       n/a      "
    return f"[{lo:.4f}, {hi:.4f}]"


def _fmt_h(h: float | None) -> str:
    if h is None or (isinstance(h, float) and math.isnan(h)):
        return "   n/a"
    return f"{h:+.4f}"


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "   n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def print_results_table(
    all_results: list[ModelResult],
    pairwise: dict[str, list[PairwiseTest]],
    seed_stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    # --- Main accuracy table ---
    header = (
        f"{'model':<30}  {'config':<14}  {'split':<12}  "
        f"{'n':>6}  {'acc':>7}  {'95% CI':^18}  {'h_vs_raw':>8}  {'CI method':<16}"
    )
    print("\n" + "=" * len(header))
    print("REWARD ACCURACY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(all_results, key=lambda x: (x.model, x.config, x[2] != "overall", x[2])):
        print(
            f"{r.model:<30}  {r.config:<14}  {r.split:<12}  "
            f"{r.n:>6}  {r.reward_accuracy:>7.4f}  {_fmt_ci(r.ci_lower, r.ci_upper):^18}  "
            f"{_fmt_h(r.cohen_h_vs_raw):>8}  {r.ci_method or '':16}"
        )

    # --- Pairwise tests ---
    if pairwise:
        print("\n" + "=" * 60)
        print("PAIRWISE SIGNIFICANCE TESTS")
        print("=" * 60)
        print(f"{'model':<30}  {'A vs B':<28}  {'test':<8}  {'stat':>8}  {'p':>7}")
        print("-" * 90)
        for model_name, tests in sorted(pairwise.items()):
            for t in tests:
                label = f"{t.a_label} vs {t.b_label}"
                stat_str = f"{t.statistic:.4f}" if not math.isnan(t.statistic) else "   n/a"
                print(
                    f"{model_name:<30}  {label:<28}  {t.test:<8}  "
                    f"{stat_str:>8}  {_fmt_p(t.p_value):>7}"
                )

    # --- Seed mean±std ---
    if seed_stats:
        print("\n" + "=" * 60)
        print("SEED REPLICATE STATISTICS (mean ± std)")
        print("=" * 60)
        print(f"{'model':<30}  {'config':<14}  {'mean':>7}  {'std':>7}")
        print("-" * 65)
        for model_name, cfg_map in sorted(seed_stats.items()):
            for cfg, stats in sorted(cfg_map.items()):
                print(
                    f"{model_name:<30}  {cfg:<14}  "
                    f"{stats['mean']:>7.4f}  {stats['std']:>7.4f}"
                )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_summary_csv(
    path: Path,
    all_results: list[ModelResult],
    pairwise: dict[str, list[PairwiseTest]],
    seed_stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Main results
        writer.writerow([
            "section", "model", "config", "split", "n", "correct",
            "reward_accuracy", "ci_lower", "ci_upper", "ci_method", "cohen_h_vs_raw",
        ])
        for r in sorted(all_results, key=lambda x: (x.model, x.config, x[2] != "overall", x[2])):
            writer.writerow([
                "accuracy",
                r.model, r.config, r.split, r.n, r.correct,
                f"{r.reward_accuracy:.6f}",
                f"{r.ci_lower:.6f}" if r.ci_lower is not None and not math.isnan(r.ci_lower) else "",
                f"{r.ci_upper:.6f}" if r.ci_upper is not None and not math.isnan(r.ci_upper) else "",
                r.ci_method or "",
                f"{r.cohen_h_vs_raw:.6f}" if r.cohen_h_vs_raw is not None and not math.isnan(r.cohen_h_vs_raw) else "",
            ])

        # Pairwise tests
        writer.writerow([])
        writer.writerow(["section", "model", "config_a", "config_b", "test", "statistic", "p_value"])
        for model_name, tests in sorted(pairwise.items()):
            for t in tests:
                writer.writerow([
                    "pairwise", model_name, t.a_label, t.b_label, t.test,
                    f"{t.statistic:.6f}" if not math.isnan(t.statistic) else "",
                    f"{t.p_value:.6f}" if not math.isnan(t.p_value) else "",
                ])

        # Seed stats
        writer.writerow([])
        writer.writerow(["section", "model", "config", "seed_mean", "seed_std"])
        for model_name, cfg_map in sorted(seed_stats.items()):
            for cfg, stats in sorted(cfg_map.items()):
                writer.writerow([
                    "seed_stats", model_name, cfg,
                    f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seed_dirs(specs: list[str]) -> dict[str, list[Path]]:
    """
    Parse --seed-dirs entries of the form  model_label:dir0,dir1,dir2
    Returns {model_label: [Path, ...]}.
    """
    out: dict[str, list[Path]] = {}
    for spec in specs:
        if ":" not in spec:
            print(f"[warn] ignoring malformed --seed-dirs entry (no ':'): {spec!r}", file=sys.stderr)
            continue
        label, dirs_str = spec.split(":", 1)
        dirs = [Path(d.strip()) for d in dirs_str.split(",") if d.strip()]
        out[label.strip()] = dirs
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CI, McNemar's test, Cohen's h for reward model evals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True, metavar="DIR",
        help="Directory of model result subdirs, each containing raw.csv, dpo_cjb.csv, etc.",
    )
    parser.add_argument(
        "--seed-dirs", nargs="+", default=[], metavar="MODEL:DIR0,DIR1,...",
        help="Per-model seed directories for mean±std. Format: model_label:seed0_dir,seed1_dir,...",
    )
    parser.add_argument(
        "--split", default="overall", metavar="SPLIT",
        help="Which split to highlight in summary (default: overall).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Output CSV path (default: <results-dir>/stats_summary.csv).",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"[error] --results-dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover model subdirectories (any subdir containing at least one recognised CSV)
    model_dirs: list[Path] = []
    for candidate in sorted(results_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if any((candidate / f"{cfg}.csv").exists() for cfg in CONFIGS):
            model_dirs.append(candidate)

    if not model_dirs:
        print(f"[warn] no model subdirectories with recognised CSVs found in {results_dir}", file=sys.stderr)

    # Analyse each model
    all_results: list[ModelResult] = []
    pairwise: dict[str, list[PairwiseTest]] = {}

    for model_dir in model_dirs:
        model_results = analyse_model_dir(model_dir)
        all_results.extend(model_results)

        tests = pairwise_tests(model_dir)
        if tests:
            pairwise[model_dir.name] = tests

    # Seed stats
    seed_dir_map = _parse_seed_dirs(args.seed_dirs)
    seed_stats: dict[str, dict[str, dict[str, float]]] = {}
    for model_label, dirs in seed_dir_map.items():
        stats = compute_seed_stats(model_label, dirs)
        if stats:
            seed_stats[model_label] = stats

    # Print
    print_results_table(all_results, pairwise, seed_stats)

    # Write CSV
    out_path = args.output or (results_dir / "stats_summary.csv")
    write_summary_csv(out_path, all_results, pairwise, seed_stats)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
