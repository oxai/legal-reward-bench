from __future__ import annotations

from .build import build_variant_triples, parse_variant_selection
from .common.config import VARIANTS

__all__ = [
    "VARIANTS",
    "build_variant_triples",
    "parse_variant_selection",
]
