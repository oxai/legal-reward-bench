from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_jsonl(path) -> list[dict]:
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "06_eval"
DEFAULT_PAIRS = ROOT / "pipeline" / "04_pairs" / "outputs" / "pairs.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
CJB_STAGE = ROOT / "pipeline" / "05_reward_model"

_MIN_FREE_VRAM = 6 * 1024 ** 3


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPO alignment via reward accuracy on preference pairs.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or path to saved model/adapter.")
    parser.add_argument("--base-model", default=None, help="Base model id (required when --model is a LoRA adapter).")
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated ContextualJudgeBench splits.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--qlora", action="store_true", help="Load base model in 4-bit NF4 for large models.")
    parser.add_argument("--flash-attn2", action="store_true", help="Use Flash Attention 2 (requires flash-attn).")
    parser.add_argument(
        "--spectral-layer", default=None,
        help="Comma-separated layer indices for spectral surgery, e.g. '7' or '7,14,20'.",
    )
    parser.add_argument(
        "--spectral-alpha", default=None,
        help="Comma-separated alpha values (one per layer, or single value broadcast to all), e.g. '-0.19' or '-0.1,-0.05'.",
    )
    parser.add_argument(
        "--scan-snr", action="store_true",
        help="Fast Lanczos SNR scan across all layers (k=32). Prints table and exits — no eval.",
    )
    parser.add_argument(
        "--snr-k", type=int, default=32,
        help="Lanczos rank k for --scan-snr (default 32, paper uses 32).",
    )
    parser.add_argument(
        "--pairs-output", type=Path, default=None, metavar="PATH",
        help="If set, write per-pair JSONL {idx, split, correct} to this file.",
    )
    parser.add_argument(
        "--length-normalize", action="store_true",
        help="Use mean per-token log-prob instead of sum. Sum is default since it matches DPO's training objective; "
             "the length_controlled subset metric handles length bias more cleanly than this flag.",
    )
    args = parser.parse_args()

    _check_vram()

    path = Path(args.model)
    is_adapter = path.exists() and (path / "adapter_config.json").exists()
    base_id = args.base_model if is_adapter else args.model

    if is_adapter and base_id is None:
        raise ValueError("--base-model is required when --model is a LoRA adapter directory.")

    # Load base model first — surgery is applied here, before the LoRA adapter is attached.
    # This matches the spectral-steering-v2 pattern: surgery on raw base weights, then eval.
    base = _load_base(base_id, use_qlora=args.qlora, use_flash_attn2=args.flash_attn2)

    # --scan-snr: fast Lanczos sweep, print table, write CSV, exit.
    if args.scan_snr:
        snr_results = scan_snr_all_layers(base, k=args.snr_k)
        print_snr_table(snr_results)
        out = args.output or STAGE_DIR / "outputs" / f"snr_scan__{base_id.replace('/', '_')}.csv"
        write_snr_csv(out, snr_results)
        print(f"Wrote SNR scan to {out}")
        return

    # Parse spectral surgery spec.
    layers: list[int] = _parse_int_list(args.spectral_layer) if args.spectral_layer else []
    if layers and args.spectral_alpha is not None:
        alphas_raw = _parse_float_list(args.spectral_alpha)
        alphas = alphas_raw if len(alphas_raw) == len(layers) else [alphas_raw[0]] * len(layers)
    elif layers:
        alphas = [-0.5] * len(layers)
    else:
        alphas = []

    for layer_idx, alpha in zip(layers, alphas):
        apply_spectral_surgery(base, layer_idx=layer_idx, alpha=alpha)

    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        model = base
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(args)
    if args.limit is not None:
        records = _limit_balanced(records, args.limit)

    print(f"Evaluating {len(records)} pairs... (length_normalize={args.length_normalize})")
    results, pair_records = evaluate(model, tokenizer, records, max_length=args.max_length,
                                     length_normalize=args.length_normalize)

    print_table(results)

    output = args.output or default_output_path(args.model, layers, alphas)
    write_csv(output, results)
    print(f"Wrote results to {output}")

    if args.pairs_output is not None:
        write_pairs_jsonl(args.pairs_output, pair_records)
        print(f"Wrote per-pair output to {args.pairs_output}")


def _check_vram() -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024 ** 3
    total_gb = total / 1024 ** 3
    print(f"VRAM: {(total-free)/1024**3:.1f} GB used / {total_gb:.1f} GB total ({free_gb:.1f} GB free)", flush=True)
    if free < _MIN_FREE_VRAM:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB VRAM free — need at least {_MIN_FREE_VRAM/1024**3:.0f} GB. "
            "Kill lingering GPU processes first (check nvidia-smi)."
        )


def _load_base(model_path: str, *, use_qlora: bool, use_flash_attn2: bool = False) -> AutoModelForCausalLM:
    attn_impl = "flash_attention_2" if use_flash_attn2 else "sdpa"
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto",
            attn_implementation=attn_impl,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_impl,
    )


def _resolve_layers(model: Any) -> list:
    """Robust layer accessor — copied from spectral-steering-v2/scripts/steer.py."""
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"Cannot find transformer layers for {type(model).__name__}")


def _dequantize_weight(dp: Any) -> torch.Tensor:
    """Return down_proj weight as float32 CPU tensor, dequantizing if 4-bit."""
    if hasattr(dp.weight, "quant_state"):
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(dp.weight.data, dp.weight.quant_state).float().cpu()
    return dp.weight.data.float().cpu()


def scan_snr_all_layers(model: Any, k: int = 32) -> list[tuple[int, float]]:
    """
    Fast SNR scan across all transformer layers.
    σ₁: Lanczos top singular value (k steps, accurate to <1%).
    σ̃:  ||W||_F / √rank  — RMS estimate of the median (Hutchinson proxy).
         S[k//2] from svd_lowrank is the k/2-th LARGEST value, not the true
         median (rank/2-th), so it severely underestimates SNR for large matrices.
    Safe dose bound: |α| ≪ 1/SNR per the paper.
    """
    layers = _resolve_layers(model)
    results: list[tuple[int, float]] = []
    niter = max(4, k // 4)
    print(f"\nSNR scan ({len(layers)} layers, k={k}, niter={niter}):", flush=True)
    for i, layer in enumerate(layers):
        W = _dequantize_weight(layer.mlp.down_proj)
        _, S_top, _ = torch.svd_lowrank(W, q=1, niter=niter)
        sigma_1 = S_top[0].item()
        # Hutchinson proxy for median: ||W||_F / sqrt(rank)
        rank = min(W.shape)
        sigma_med = W.norm("fro").item() / (rank ** 0.5)
        del W
        snr = sigma_1 / sigma_med if sigma_med > 1e-12 else 0.0
        safe_alpha = 1.0 / snr if snr > 0 else float("inf")
        print(f"  L{i:2d}: SNR={snr:.3f}  safe|α|≪{safe_alpha:.3f}", flush=True)
        results.append((i, snr))
    return results


def apply_spectral_surgery(model: Any, layer_idx: int, alpha: float) -> None:
    """
    Compress/amplify the dominant singular subspace of mlp.down_proj at layer_idx.
    Formula: σᵢ' = σᵢ · (1 + α · (σᵢ − μ_σ) / s_σ)
    α < 0 compresses dominant directions (reduces sycophancy).
    α > 0 amplifies dominant directions (induces sycophancy).

    Uses full thin SVD (torch.linalg.svd) — matches spectral-steering-v2 weight_svd_full().
    Paper: safe dose is |α| ≪ 1/SNR_ℓ. For Llama-3.1-8B L7 (SNR=5.25), safe |α| ≪ 0.19.
    """
    layers = _resolve_layers(model)
    layer = layers[layer_idx]
    dp = layer.mlp.down_proj

    W_cpu = _dequantize_weight(dp)

    # Full thin SVD on CPU via LAPACK — correct SNR, matches spectral-steering-v2.
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    del W_cpu

    snr = (S[0] / S.median()).item()
    S_new = S * (1 + alpha * (S - S.mean()) / S.std())
    del S

    W_new = (U @ torch.diag(S_new) @ Vh).contiguous()
    del U, S_new, Vh

    ref_dtype = next(layer.parameters()).dtype
    if ref_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        ref_dtype = torch.float16

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    new_mod = torch.nn.Linear(W_new.shape[1], W_new.shape[0], bias=False).to(device).to(ref_dtype)
    new_mod.weight.data = W_new.to(ref_dtype).to(device)
    del W_new

    layer.mlp.down_proj = new_mod
    torch.cuda.empty_cache()
    safe_alpha = 1.0 / snr if snr > 0 else float("inf")
    print(
        f"Spectral surgery: L{layer_idx}, α={alpha}, SNR={snr:.2f}, safe|α|≪{safe_alpha:.3f} → modified down_proj",
        flush=True,
    )


def _limit_balanced(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Sample `limit` records evenly across splits so all splits are represented."""
    from collections import defaultdict
    by_split: dict[str, list] = defaultdict(list)
    for r in records:
        by_split[r.get("split", "all")].append(r)
    n_splits = len(by_split)
    per_split = max(1, limit // n_splits)
    out = []
    for split_records in by_split.values():
        out.extend(split_records[:per_split])
    return out


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
        sys.path.insert(0, str(CJB_STAGE))
        from sources.contextual_judge_bench import QA_SPLITS, load_pairs

        splits = (
            tuple(s.strip() for s in args.splits.split(",") if s.strip())
            if args.splits
            else QA_SPLITS
        )
        prompt_template = read_text(DEFAULT_PROMPT)
        return load_pairs(prompt_template=prompt_template, splits=splits)

    pairs_path = args.pairs or DEFAULT_PAIRS
    return [
        {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "split": r.get("split") or r.get("metadata", {}).get("context_variant", "pipeline"),
        }
        for r in read_jsonl(pairs_path)
        if "prompt" in r
    ]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion — what RewardBench reports per split."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * (((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    max_length: int,
    length_normalize: bool = False,
    length_ratio_threshold: float = 2.0,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Score pairs and return (per-split metrics, per-pair records).

    Per-split fields (RewardBench-comparable):
      - reward_accuracy:   pairwise accuracy (random=0.5)
      - ci_low, ci_high:   Wilson score 95% CI
      - mean_margin:       mean(logp_chosen − logp_rejected); DPO-style preference strength

    Synthetic rows:
      - "overall":           micro-average over all pairs
      - "macro":             unweighted mean across real splits (each sub-skill weighted equally)
      - "length_controlled": micro-average over pairs with len ratio ≤ threshold (no length artifact)
    """
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    split_margin_sum: dict[str, float] = defaultdict(float)
    pair_records: list[dict[str, Any]] = []
    lc_correct = 0
    lc_total = 0
    lc_margin_sum = 0.0

    total = len(records)
    skipped = 0
    for index, record in enumerate(records, start=1):
        split = record.get("split", "all")
        log_p_chosen = sequence_log_prob(model, tokenizer, record["prompt"], record["chosen"], max_length, length_normalize)
        torch.cuda.empty_cache()
        log_p_rejected = sequence_log_prob(model, tokenizer, record["prompt"], record["rejected"], max_length, length_normalize)
        torch.cuda.empty_cache()
        if log_p_chosen is None or log_p_rejected is None:
            skipped += 1
            continue
        margin = log_p_chosen - log_p_rejected
        correct = margin > 0
        split_correct[split] += int(correct)
        split_correct["overall"] += int(correct)
        split_total[split] += 1
        split_total["overall"] += 1
        split_margin_sum[split] += margin
        split_margin_sum["overall"] += margin

        len_c = len(record["chosen"])
        len_r = len(record["rejected"])
        ratio = max(len_c, len_r) / max(1, min(len_c, len_r))
        in_lc = ratio <= length_ratio_threshold
        if in_lc:
            lc_correct += int(correct)
            lc_total += 1
            lc_margin_sum += margin

        pair_records.append({
            "idx": index - 1, "split": split, "correct": int(correct),
            "margin": round(margin, 4),
            "len_chosen": len_c, "len_rejected": len_r, "in_length_controlled": int(in_lc),
        })
        if index % 50 == 0 or index == total:
            n_scored = split_total["overall"]
            overall_acc = split_correct["overall"] / n_scored if n_scored else 0.0
            print(f"[{index}/{total}] scored={n_scored} skipped={skipped} overall reward_accuracy={overall_acc:.3f}", flush=True)

    def _row(n: int, correct: int, margin_sum: float) -> dict[str, Any]:
        acc = correct / n if n else 0.0
        lo, hi = _wilson_ci(correct, n)
        return {
            "n": n,
            "correct": correct,
            "reward_accuracy": round(acc, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "mean_margin": round(margin_sum / n, 4) if n else 0.0,
        }

    results: dict[str, dict[str, Any]] = {
        split: _row(split_total[split], split_correct[split], split_margin_sum[split])
        for split in split_total
    }
    real_splits = [s for s in split_total if s != "overall"]
    if real_splits:
        macro_acc = sum(results[s]["reward_accuracy"] for s in real_splits) / len(real_splits)
        macro_margin = sum(results[s]["mean_margin"] for s in real_splits) / len(real_splits)
        results["macro"] = {
            "n": len(real_splits),
            "correct": "",
            "reward_accuracy": round(macro_acc, 4),
            "ci_low": "",
            "ci_high": "",
            "mean_margin": round(macro_margin, 4),
        }
    if lc_total:
        results["length_controlled"] = _row(lc_total, lc_correct, lc_margin_sum)
    return results, pair_records


def write_pairs_jsonl(path: Path, pair_records: list[dict[str, Any]]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in pair_records:
            f.write(json.dumps(rec) + "\n")


def sequence_log_prob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    length_normalize: bool = False,
) -> float | None:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]

    max_completion = max_length - prompt_ids.shape[1]
    if max_completion <= 0:
        return None  # prompt too long; caller should skip this pair
    completion_ids = completion_ids[:, :max_completion]

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    prompt_len = prompt_ids.shape[1]
    response_logits = logits[0, prompt_len - 1 : -1, :].clone()
    del logits
    response_labels = input_ids[0, prompt_len:]

    if response_labels.shape[0] == 0:
        return 0.0

    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(response_labels)), response_labels]
    return token_log_probs.mean().item() if length_normalize else token_log_probs.sum().item()


def print_snr_table(results: list[tuple[int, float]]) -> None:
    print(f"\n{'layer':>6}  {'SNR':>8}  {'safe |α|≪':>12}  {'class':>12}")
    print("-" * 48)
    for layer_idx, snr in results:
        safe = 1.0 / snr if snr > 0 else float("inf")
        cls = "COLLAPSE" if snr > 8.0 else ("localised" if snr >= 4.5 else ("low-SNR" if snr >= 2.5 else "noise"))
        print(f"{layer_idx:>6}  {snr:>8.3f}  {safe:>12.3f}  {cls:>12}")
    peak_layer, peak_snr = max(results, key=lambda x: x[1])
    print(f"\nPeak SNR: L{peak_layer} (SNR={peak_snr:.3f}), safe |α| ≪ {1/peak_snr:.3f}")


def write_snr_csv(path: Path, results: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "snr", "safe_alpha_bound"])
        writer.writeheader()
        for layer_idx, snr in results:
            writer.writerow({"layer": layer_idx, "snr": round(snr, 4), "safe_alpha_bound": round(1.0 / snr, 4) if snr > 0 else 999})


def print_table(results: dict[str, dict[str, Any]]) -> None:
    priority = {"overall": 0, "macro": 1, "length_controlled": 2}
    rows = sorted(results.items(), key=lambda x: (priority.get(x[0], 3), x[0]))
    col_w = max(len(split) for split, _ in rows)
    print(f"\n{'split':<{col_w}}  {'n':>6}  {'acc':>7}  {'95% CI':>15}  {'margin':>8}")
    print("-" * (col_w + 44))
    for split, stats in rows:
        ci = (
            f"[{stats['ci_low']:.3f},{stats['ci_high']:.3f}]"
            if isinstance(stats.get("ci_low"), (int, float))
            else "—"
        )
        print(f"{split:<{col_w}}  {stats['n']:>6}  {stats['reward_accuracy']:>7.4f}  {ci:>15}  {stats['mean_margin']:>8.3f}")
    print()


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    priority = {"overall": 0, "macro": 1, "length_controlled": 2}
    rows = sorted(results.items(), key=lambda x: (priority.get(x[0], 3), x[0]))
    fields = ["split", "n", "correct", "reward_accuracy", "ci_low", "ci_high", "mean_margin"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for split, stats in rows:
            writer.writerow({"split": split, **stats})


def default_output_path(model: str, layers: list[int], alphas: list[float]) -> Path:
    slug = model.replace("/", "_").replace("\\", "_").strip("_")
    if layers:
        spec = "__".join(f"L{l}_a{a}" for l, a in zip(layers, alphas))
        suffix = f"__spectral_{spec}"
    else:
        suffix = ""
    return STAGE_DIR / "outputs" / f"reward_accuracy__{slug}{suffix}.csv"


if __name__ == "__main__":
    main()
