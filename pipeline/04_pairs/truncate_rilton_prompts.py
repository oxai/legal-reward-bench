"""
Pre-truncate Rilton prompts so DPO training with max_length=1024 actually
learns. Original prompts are ~17k chars / ~5k tokens; if max_length truncates
to fit, the response gets masked out and gradients vanish.

We KEEP the last ~3000 chars (~750 tokens), which preserves: end of the
legal context, the full Question, and the "Answer:\\n" marker. That leaves
~250 tokens of budget for chosen/rejected responses inside max_length=1024.

Usage:
  python truncate_rilton_prompts.py --input pairs_train_v2.jsonl --output pairs_train_v2_t.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def truncate_prompt(prompt: str, max_chars: int = 3000) -> str:
    if len(prompt) <= max_chars:
        return prompt
    # Keep the tail; if the prompt begins with "Legal context:\n", preserve a tiny header.
    tail = prompt[-max_chars:]
    # Ensure we don't start mid-word; back up to the first newline.
    nl = tail.find("\n")
    if 0 < nl < 200:
        tail = tail[nl + 1 :]
    header = "[Legal context truncated for length; question follows]\n\n"
    return header + tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    n_kept = n_trunc = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as fi, args.output.open("w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            old_len = len(r["prompt"])
            r["prompt"] = truncate_prompt(r["prompt"], max_chars=args.max_chars)
            if len(r["prompt"]) < old_len:
                n_trunc += 1
            else:
                n_kept += 1
            fo.write(json.dumps(r) + "\n")
    print(f"Truncated {n_trunc} prompts, kept {n_kept} intact. Wrote {args.output}.")


if __name__ == "__main__":
    main()
