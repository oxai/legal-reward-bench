from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common.storage import read_jsonl, read_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "04_pairs"
DEFAULT_TRIPLES = ROOT / "pipeline" / "01_triples" / "outputs" / "triples.jsonl"
DEFAULT_PROMPT = ROOT / "pipeline" / "02_responses" / "prompts" / "generation_v1.txt"
DEFAULT_OUTPUT = STAGE_DIR / "outputs" / "pairs.jsonl"

POLICY_VERSION = "hierarchical_v1"
HIERARCHY = ("answer_behavior", "faithfulness", "correctness", "completeness")

ANSWERABLE_BEHAVIOR_RANK = {
    "attempted": 2,
    "abstained": 1,
    "unusable": 0,
}
UNANSWERABLE_BEHAVIOR_RANK = {
    "abstained": 2,
    "attempted": 1,
    "unusable": 0,
}
FAITHFULNESS_RANK = {
    "fully_supported": 4,
    "partially_supported": 3,
    "unsupported": 2,
    "contradicted": 1,
    "not_applicable": 0,
}
CORRECTNESS_RANK = {
    "correct": 2,
    "incorrect": 1,
    "not_applicable": 0,
}
COMPLETENESS_RANK = {
    "complete": 2,
    "incomplete": 1,
    "not_applicable": 0,
}
SEMANTIC_RANKS = {
    "faithfulness": FAITHFULNESS_RANK,
    "correctness": CORRECTNESS_RANK,
    "completeness": COMPLETENESS_RANK,
}


def build_pairs(
    *,
    triples: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    prompt_template: str,
) -> list[dict[str, Any]]:
    validate_inputs(triples=triples, labels=labels, responses=responses)

    labels_by_triple: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        labels_by_triple.setdefault(label["triple_id"], []).append(label)

    pairs: list[dict[str, Any]] = []
    for triple_id in sorted(labels_by_triple, key=lambda id_: triple_sort_key(triples[id_])):
        triple = triples[triple_id]
        triple_labels = sorted(labels_by_triple[triple_id], key=lambda row: row["response_id"])
        answerability = triple.get("metadata", {}).get("answerability", "answerable")

        for left_index, left in enumerate(triple_labels):
            for right in triple_labels[left_index + 1 :]:
                preference = preferred_label(left, right, answerability=answerability)
                if preference is None:
                    continue

                chosen_label, rejected_label, decisive_dimension = preference
                pairs.append(
                    materialized_pair(
                        triple=triple,
                        chosen_label=chosen_label,
                        rejected_label=rejected_label,
                        responses=responses,
                        prompt_template=prompt_template,
                        decisive_dimension=decisive_dimension,
                    )
                )

    return pairs


def preferred_label(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    answerability: str,
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    left_labels = left["labels"]
    right_labels = right["labels"]

    behavior_ranks = (
        ANSWERABLE_BEHAVIOR_RANK if answerability == "answerable" else UNANSWERABLE_BEHAVIOR_RANK
    )
    behavior_preference = compare_ranked(
        left,
        right,
        rank=behavior_ranks,
        label_name="answer_behavior",
    )
    if behavior_preference is not None:
        chosen, rejected = behavior_preference
        return chosen, rejected, "answer_behavior"

    if answerability != "answerable":
        return None
    if left_labels["answer_behavior"] != "attempted" or right_labels["answer_behavior"] != "attempted":
        return None

    for label_name in HIERARCHY[1:]:
        semantic_preference = compare_ranked(
            left,
            right,
            rank=SEMANTIC_RANKS[label_name],
            label_name=label_name,
        )
        if semantic_preference is not None:
            chosen, rejected = semantic_preference
            return chosen, rejected, label_name

    return None


def compare_ranked(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    rank: dict[str, int],
    label_name: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    left_value = rank[left["labels"][label_name]]
    right_value = rank[right["labels"][label_name]]
    if left_value == right_value:
        return None
    return (left, right) if left_value > right_value else (right, left)


def materialized_pair(
    *,
    triple: dict[str, Any],
    chosen_label: dict[str, Any],
    rejected_label: dict[str, Any],
    responses: dict[str, dict[str, Any]],
    prompt_template: str,
    decisive_dimension: str,
) -> dict[str, Any]:
    chosen_response = responses[chosen_label["response_id"]]
    rejected_response = responses[rejected_label["response_id"]]
    metadata = triple.get("metadata", {})
    chosen_id = chosen_response["id"]
    rejected_id = rejected_response["id"]
    preference_type = split_name(
        answerability=metadata.get("answerability", "answerable"),
        decisive_dimension=decisive_dimension,
    )

    return {
        "id": f"{chosen_id}__preferred_to__{rejected_id}",
        "triple_id": triple["id"],
        "chosen_response_id": chosen_id,
        "rejected_response_id": rejected_id,
        "chosen_label_id": chosen_label["id"],
        "rejected_label_id": rejected_label["id"],
        "chosen_labels": chosen_label["labels"],
        "rejected_labels": rejected_label["labels"],
        "prompt": prompt_template.format(context=triple["context"], question=triple["question"]),
        "chosen": chosen_response["response"],
        "rejected": rejected_response["response"],
        "split": preference_type,
        "metadata": {
            "policy_version": POLICY_VERSION,
            "decisive_dimension": decisive_dimension,
            "preference_type": preference_type,
            "answerability": metadata.get("answerability", "answerable"),
            "context_variant": metadata.get("context_variant", "base"),
            "base_triple_id": metadata.get("base_triple_id", triple["id"]),
        },
    }


def split_name(*, answerability: str, decisive_dimension: str) -> str:
    if decisive_dimension != "answer_behavior":
        return decisive_dimension
    return "refusal_answerable" if answerability == "answerable" else "refusal_unanswerable"


def triple_sort_key(triple: dict[str, Any]) -> tuple[str, str]:
    metadata = triple.get("metadata", {})
    return (
        metadata.get("base_triple_id", triple["id"]),
        metadata.get("context_variant", "base"),
    )


def validate_inputs(
    *,
    triples: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
) -> None:
    label_response_ids = [label["response_id"] for label in labels]
    duplicate_label_ids = [id_ for id_, count in Counter(label_response_ids).items() if count > 1]
    if duplicate_label_ids:
        raise RuntimeError(f"Duplicate labels for response ids: {duplicate_label_ids[:10]}")

    missing_responses = sorted(set(label_response_ids) - set(responses))
    if missing_responses:
        raise RuntimeError(f"Labels reference missing responses: {missing_responses[:10]}")

    missing_triples = sorted({label["triple_id"] for label in labels} - set(triples))
    if missing_triples:
        raise RuntimeError(f"Labels reference missing triples: {missing_triples[:10]}")

    mismatched_triples = [
        label["response_id"]
        for label in labels
        if responses[label["response_id"]]["triple_id"] != label["triple_id"]
    ]
    if mismatched_triples:
        raise RuntimeError(f"Label/response triple_id mismatches: {mismatched_triples[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference pairs from pointwise response labels.")
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    triples = {row["id"]: row for row in read_jsonl(args.triples)}
    labels = read_jsonl(args.labels)

    responses: dict[str, dict[str, Any]] = {}
    for response_path in args.responses:
        for row in read_jsonl(response_path):
            responses[row["id"]] = row

    pairs = build_pairs(
        triples=triples,
        labels=labels,
        responses=responses,
        prompt_template=read_text(args.prompt),
    )

    count = write_jsonl(args.output, pairs)
    print(f"Wrote {count} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
