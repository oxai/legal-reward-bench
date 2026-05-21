from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--spectral-layer", default=None,
        help="Comma-separated layer indices for spectral surgery before training, e.g. '1' or '1,14'.",
    )
    parser.add_argument(
        "--spectral-alpha", default=None,
        help="Comma-separated alpha values (one per layer, or single value broadcast), e.g. '-0.3' or '-0.3,-0.2'.",
    )
    parser.add_argument("--source", choices=["pipeline", "contextual_judge_bench"], default="pipeline")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated ContextualJudgeBench splits (default: all QA splits).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Load base model in 4-bit NF4 (QLoRA). Reduces VRAM from ~2GB/B to ~0.5GB/B.",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        help="Use Flash Attention 2 (requires flash-attn package). Cuts VRAM and is ~2x faster on H100.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM at the cost of ~20%% slower training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training.")


def load_model_and_tokenizer(
    model_path: str,
    *,
    use_qlora: bool = False,
    use_flash_attn2: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if use_flash_attn2 else "sdpa"

    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation=attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    return model, tokenizer


def make_lora_config(*, r: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_spectral_surgery_from_args(model: AutoModelForCausalLM, args: argparse.Namespace) -> None:
    """Apply spectral surgery to base model weights before LoRA is attached.

    With QLoRA the base is frozen, so surgery is preserved throughout training.
    No-op when --spectral-layer is not passed.
    """
    if not getattr(args, "spectral_layer", None):
        return

    layers_idx = [int(x.strip()) for x in args.spectral_layer.split(",") if x.strip()]
    if args.spectral_alpha is not None:
        raw = [float(x.strip()) for x in args.spectral_alpha.split(",") if x.strip()]
        alphas = raw if len(raw) == len(layers_idx) else [raw[0]] * len(layers_idx)
    else:
        alphas = [-0.3] * len(layers_idx)

    transformer_layers = _get_transformer_layers(model)
    for li, alpha in zip(layers_idx, alphas):
        _apply_surgery_one_layer(transformer_layers, li, alpha)


def _get_transformer_layers(model: Any) -> list:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise RuntimeError(f"Cannot locate transformer layers in {type(model).__name__}")


def _apply_surgery_one_layer(layers: list, layer_idx: int, alpha: float) -> None:
    dp = layers[layer_idx].mlp.down_proj
    if hasattr(dp.weight, "quant_state"):
        import bitsandbytes as bnb
        W = bnb.functional.dequantize_4bit(dp.weight.data, dp.weight.quant_state).float().cpu()
    else:
        W = dp.weight.data.float().cpu()

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    del W
    snr = (S[0] / S.median()).item()
    S_new = S * (1 + alpha * (S - S.mean()) / S.std())
    del S
    W_new = (U @ torch.diag(S_new) @ Vh).contiguous()
    del U, S_new, Vh

    # For QLoRA layers the weight is NF4-quantized; use the declared compute dtype (bfloat16).
    # For regular layers, infer from the existing weight dtype.
    if hasattr(dp.weight, "quant_state"):
        ref_dtype = torch.bfloat16
    else:
        ref_dtype = dp.weight.dtype if dp.weight.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    device = next(layers[layer_idx].parameters()).device
    new_dp = torch.nn.Linear(W_new.shape[1], W_new.shape[0], bias=False).to(device).to(ref_dtype)
    new_dp.weight.data = W_new.to(ref_dtype).to(device)
    del W_new
    layers[layer_idx].mlp.down_proj = new_dp
    torch.cuda.empty_cache()
    print(f"Spectral surgery: L{layer_idx} α={alpha:+.3f} SNR={snr:.2f} → base weights modified", flush=True)


def load_records(args: argparse.Namespace, *, default_pairs: Path, default_prompt: Path) -> list[dict[str, Any]]:
    if args.source == "contextual_judge_bench":
        from sources.contextual_judge_bench import QA_SPLITS, load_pairs
        from common.storage import read_text

        splits = (
            tuple(s.strip() for s in args.splits.split(",") if s.strip())
            if args.splits
            else QA_SPLITS
        )
        prompt_template = read_text(default_prompt)
        return load_pairs(prompt_template=prompt_template, splits=splits)

    pairs_path = args.pairs or default_pairs
    from common.storage import read_jsonl
    return [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in read_jsonl(pairs_path)
    ]
