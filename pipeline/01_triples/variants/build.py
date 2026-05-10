from __future__ import annotations

from typing import Any

from common.records import Triple

from .bm25 import build_bm25_variants
from .contriever import build_contriever_variants
from .glove import build_glove_variants
from .nomic import build_nomic_variants
from .random import build_random_variants
from .common.config import VARIANTS
from .surrounding import build_surrounding_variants


def parse_variant_selection(value: str) -> set[str]:
    if value == "all":
        return set(VARIANTS)
    selected = {variant.strip() for variant in value.split(",") if variant.strip()}
    unknown = sorted(selected - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {list(VARIANTS)}")
    if not selected:
        raise ValueError("At least one variant must be selected.")
    return selected


def build_variant_triples(
    base_triples: list[Triple],
    *,
    corpus: list[dict[str, Any]],
    variants: set[str] | None = None,
) -> list[Triple]:
    selected_variants = set(VARIANTS) if variants is None else variants
    if not selected_variants:
        return []
    unknown = sorted(selected_variants - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {list(VARIANTS)}")

    base_order = {triple.id: index for index, triple in enumerate(base_triples)}
    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    output: list[Triple] = []
    for build_method_variants in METHOD_BUILDERS:
        output.extend(
            build_method_variants(
                base_triples,
                corpus=corpus,
                selected_variants=selected_variants,
            )
        )
    output.sort(
        key=lambda triple: (
            base_order[triple.metadata["base_triple_id"]],
            variant_order[triple.metadata["context_variant"]],
        )
    )
    return output


METHOD_BUILDERS = (
    build_surrounding_variants,
    build_random_variants,
    build_bm25_variants,
    build_nomic_variants,
    build_glove_variants,
    build_contriever_variants,
)
