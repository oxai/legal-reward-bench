"""
Wrapper that runs eval_dpo with stable logging, bypassing all PowerShell
pipe encoding issues and Python's StreamReader UnicodeDecodeError.

Uses low-level os.read() on the raw pipe file descriptor so bitsandbytes'
non-UTF-8 progress bar bytes (e.g. 0x82) never touch a codec.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

LOG_TXT = Path(__file__).parent / "outputs" / "eval_run_gpu.txt"
LOG_TXT.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-u",
    str(Path(__file__).parent / "eval_dpo.py"),
    "--model", "pipeline/05_reward_model/outputs/dpo_model",
    "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
    "--qlora",
    "--source", "contextual_judge_bench",
    "--limit", "100",
    "--spectral-layer", "7",
    "--spectral-alpha", "-0.5",
]

print(f"Log: {LOG_TXT}", flush=True)
print(f"CMD: {' '.join(cmd)}\n", flush=True)

env = {**os.environ, "PYTHONUTF8": "1", "TRANSFORMERS_VERBOSITY": "warning"}
cwd = str(Path(__file__).parents[2])

proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=env, cwd=cwd,
)

SKIP = [b"Loading weights", b"it/s]", b"FutureWarning", b"_check_is_size", b"\r"]

buf = b""
with LOG_TXT.open("wb") as logf:
    fd = proc.stdout.fileno()
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        logf.write(chunk)
        logf.flush()
        buf += chunk
        # split on newlines, print clean lines to console
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line_s = line.decode("utf-8", errors="replace").rstrip()
            if line_s and not any(s in line for s in SKIP):
                print(line_s, flush=True)

proc.wait()
print(f"\nDone. Exit code: {proc.returncode}", flush=True)
