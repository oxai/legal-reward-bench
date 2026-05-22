"""Compute per-layer spectral SNR of MLP down_proj for a causal LM.

Following Syco.pdf, the dominant spectral direction of a single MLP weight
matrix mediates sycophancy in 10/11 open-source 1B-14B models.  The per-layer
SNR (S[0] / S.median()) identifies the target layer from weights alone.
"""
from __future__ import annotations

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id or local path")
    parser.add_argument("--out", default=None, help="optional: TSV output file")
    args = parser.parse_args()

    print(f"Loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Locate transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.layers
    else:
        raise RuntimeError("Cannot locate transformer layers")

    rows = []
    for li, layer in enumerate(layers):
        dp = layer.mlp.down_proj
        W = dp.weight.data.float().cpu()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        snr = (S[0] / S.median()).item()
        s0 = S[0].item()
        smed = S.median().item()
        smean = S.mean().item()
        sstd = S.std().item()
        rows.append((li, snr, s0, smed, smean, sstd))
        del W, U, S, Vh
        torch.cuda.empty_cache()
        print(f"  L{li:2d}  SNR={snr:7.3f}  S[0]={s0:7.3f}  median={smed:7.3f}  mean={smean:6.3f}  std={sstd:6.3f}", flush=True)

    print("\nTop-5 layers by SNR:")
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    for li, snr, s0, smed, _, _ in rows_sorted[:5]:
        print(f"  L{li}: SNR={snr:.3f}  S[0]={s0:.3f}  median={smed:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("layer\tsnr\ts_0\ts_median\ts_mean\ts_std\n")
            for r in rows:
                f.write("\t".join(f"{x:.4f}" if i > 0 else str(x) for i, x in enumerate(r)) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
