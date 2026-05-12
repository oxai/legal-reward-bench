from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.records import PreferencePair, utc_now
from common.storage import read_jsonl, read_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "04_pairs"
DEFAULT_TRIPLES = ROOT / "pipeline" / "01_triples" / "outputs" / "triples.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
DEFAULT_OUTPUT = STAGE_DIR / "outputs" / "pairs.jsonl"

_FAITHFULNESS: dict[str, float] = {
    "fully_supported": 4.0,
    "partially_supported": 2.0,
    "unsupported": -2.0,
    "contradicted": -4.0,
    "not_applicable": 0.0,
}
_CORRECTNESS: dict[str, float] = {
    "correct": 4.0,
    "incorrect": -4.0,
    "not_applicable": 0.0,
}
_COMPLETENESS: dict[str, float] = {
    "complete": 2.0,
    "incomplete": -1.0,
    "not_applicable": 0.0,
}
_CONCISENESS: dict[str, float] = {
    "concise": 1.0,
    "acceptable": 0.0,
    "not_concise": -1.0,
    "not_applicable": 0.0,
}


def score_label(labels: dict[str, Any], answerability: str) -> float:
    behavior = labels["answer_behavior"]
    if behavior == "unusable":
        return -100.0
    if answerability == "answerable":
        if behavior == "abstained":
            return -8.0
        return (
            _FAITHFULNESS[labels["faithfulness"]]
            + _CORRECTNESS[labels["correctness"]]
            + _COMPLETENESS[labels["completeness"]]
            + _CONCISENESS[labels["conciseness"]]
        )
    if behavior == "abstained":
        return 5.0
    return -5.0 + _FAITHFULNESS.get(labels["faithfulness"], 0.0)


def build_pairs(
    *,
    triples: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    prompt_template: str,
    min_score_gap: float,
) -> list[PreferencePair]:
    triple_scored: dict[str, list[tuple[str, float]]] = {}
    for label in labels:
        triple_id = label["triple_id"]
        response_id = label["response_id"]
        if triple_id not in triples or response_id not in responses:
            continue
        answerability = triples[triple_id].get("metadata", {}).get("answerability", "answerable")
        score = score_label(label["labels"], answerability)
        triple_scored.setdefault(triple_id, []).append((response_id, score))

    pairs: list[PreferencePair] = []
    for triple_id, scored in triple_scored.items():
        triple = triples[triple_id]
        prompt = prompt_template.format(context=triple["context"], question=triple["question"])
        meta = triple.get("metadata", {})

        for i, (id_a, score_a) in enumerate(scored):
            for id_b, score_b in scored[i + 1:]:
                gap = score_a - score_b
                if abs(gap) < min_score_gap:
                    continue
                if gap > 0:
                    chosen_id, rejected_id = id_a, id_b
                    chosen_score, rejected_score = score_a, score_b
                else:
                    chosen_id, rejected_id = id_b, id_a
                    chosen_score, rejected_score = score_b, score_a

                pairs.append(
                    PreferencePair(
                        id=f"{chosen_id}__vs__{rejected_id}",
                        triple_id=triple_id,
                        chosen_response_id=chosen_id,
                        rejected_response_id=rejected_id,
                        prompt=prompt,
                        chosen=responses[chosen_id]["response"],
                        rejected=responses[rejected_id]["response"],
                        metadata={
                            "chosen_score": chosen_score,
                            "rejected_score": rejected_score,
                            "score_gap": chosen_score - rejected_score,
                            "answerability": meta.get("answerability", "answerable"),
                            "context_variant": meta.get("context_variant", "base"),
                            "created_at": utc_now(),
                        },
                    )
                )

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference pairs from pointwise response labels.")
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    args = parser.parse_args()

    triples = {row["id"]: row for row in read_jsonl(args.triples)}
    labels = read_jsonl(args.labels)

    responses: dict[str, dict[str, Any]] = {}
    for resp_path in args.responses:
        for row in read_jsonl(resp_path):
            responses[row["id"]] = row

    prompt_template = read_text(args.prompt)

    pairs = build_pairs(
        triples=triples,
        labels=labels,
        responses=responses,
        prompt_template=prompt_template,
        min_score_gap=args.min_score_gap,
    )

    count = write_jsonl(args.output, (pair.to_dict() for pair in pairs))
    print(f"Wrote {count} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
