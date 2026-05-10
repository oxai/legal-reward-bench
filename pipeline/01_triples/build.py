from __future__ import annotations

import argparse
from pathlib import Path

from common.records import Triple
from common.storage import write_jsonl
from sources.legal_rag_bench import load_source
from variants import VARIANTS, build_variant_triples, parse_variant_selection
from variants.common.context import format_passages

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "triples.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Legal RAG Bench triples.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="?",
        const="all",
        default=None,
        help="Include context variants. Omit the value for all variants, or pass comma-separated names.",
    )
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.list_variants:
        print("\n".join(VARIANTS))
        return

    print("Loading Legal RAG Bench...")
    source = load_source(limit=args.limit)
    print(f"Prepared {len(source.triples)} base triples.")

    triples = make_base_triples(source.triples)

    if args.variants is not None:
        selected_variants = parse_variant_selection(args.variants)
        print(f"Building {len(selected_variants)} context variants...")
        variant_triples = build_variant_triples(
            source.triples,
            corpus=source.corpus,
            variants=selected_variants,
        )
        triples = interleave_base_and_variants(triples, variant_triples)

    print(f"Writing triples to {args.output}...")
    count = write_jsonl(args.output, (triple.to_dict() for triple in triples))
    print(f"Wrote {count} triples to {args.output}")


def make_base_triples(triples: list[Triple]) -> list[Triple]:
    output: list[Triple] = []
    for triple in triples:
        gold_id = triple.metadata["context_id"]
        output.append(
            Triple(
                id=triple.id,
                question=triple.question,
                context=format_passages(
                    [
                        {
                            "id": gold_id,
                            "title": "",
                            "text": triple.context,
                            "footnotes": "",
                        }
                    ]
                ),
                answer=triple.answer,
                metadata={
                    **triple.metadata,
                    "base_triple_id": triple.id,
                    "context_variant": "base",
                    "answerability": "answerable",
                    "variant_context_ids": [gold_id],
                    "gold_context_ids": [gold_id],
                    "gold_slot": 1,
                },
            )
        )
    return output


def interleave_base_and_variants(
    base_triples: list[Triple],
    variant_triples: list[Triple],
) -> list[Triple]:
    order = {triple.id: index for index, triple in enumerate(base_triples)}
    return sorted(
        [*base_triples, *variant_triples],
        key=lambda triple: (
            order[triple.metadata["base_triple_id"]],
            0 if triple.metadata["context_variant"] == "base" else 1,
        ),
    )


if __name__ == "__main__":
    main()
