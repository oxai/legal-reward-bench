from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = ROOT / "pipeline" / "01_triples" / "cache"

RANDOM_SEED = 13
PASSAGES_PER_CONTEXT = 10

VARIANTS = (
    "gold_surrounding",
    "gold_plus_random_distractors",
    "gold_plus_bm25_distractors",
    "gold_plus_nomic_distractors",
    "gold_plus_glove_distractors",
    "gold_plus_contriever_distractors",
    "random_context",
    "bm25_hard_negative",
    "nomic_hard_negative",
    "glove_hard_negative",
    "contriever_hard_negative",
)
