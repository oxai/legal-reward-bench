from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.storage import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract aggregate metrics from response labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument(
        "--base-ids",
        default="",
        help="Optional comma-separated base triple ids to include, e.g. legal-rag-bench-0001,legal-rag-bench-0002.",
    )
    parser.add_argument(
        "--base-id-limit",
        type=int,
        default=None,
        help="Optional shorthand for legal-rag-bench-0001 through legal-rag-bench-N.",
    )
    args = parser.parse_args()

    rows = load_metric_rows(labels_path=args.labels, triples_path=args.triples, responses_path=args.responses)
    base_ids = selected_base_ids(args.base_ids, args.base_id_limit)
    if base_ids:
        rows = [row for row in rows if row["base_triple_id"] in base_ids]
        if not rows:
            raise RuntimeError(f"No metric rows matched base ids: {sorted(base_ids)}")

    output_prefix = args.output_prefix or default_output_prefix(args.labels)

    write_summary(output_prefix.with_name(output_prefix.name + "__by_variant.csv"), rows, keys=["variant", "answerability"])
    write_summary(output_prefix.with_name(output_prefix.name + "__by_model.csv"), rows, keys=["model"])
    write_summary(
        output_prefix.with_name(output_prefix.name + "__by_model_variant.csv"),
        rows,
        keys=["model", "variant", "answerability"],
    )


def load_metric_rows(*, labels_path: Path, triples_path: Path, responses_path: Path) -> list[dict[str, Any]]:
    labels = read_jsonl(labels_path)
    triples = {row["id"]: row for row in read_jsonl(triples_path)}
    responses = {row["id"]: row for row in read_jsonl(responses_path)}

    validate_inputs(labels=labels, triples=triples, responses=responses)

    rows: list[dict[str, Any]] = []
    for label in labels:
        response = responses[label["response_id"]]
        triple = triples[label["triple_id"]]
        rows.append(
            {
                "model": response["metadata"]["model"],
                "base_triple_id": triple["metadata"].get("base_triple_id", triple["id"]),
                "variant": triple["metadata"]["context_variant"],
                "answerability": triple["metadata"]["answerability"],
                **label["labels"],
            }
        )
    return rows


def validate_inputs(
    *,
    labels: list[dict[str, Any]],
    triples: dict[str, dict[str, Any]],
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


def write_summary(path: Path, rows: list[dict[str, Any]], *, keys: list[str]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        summary = dict(zip(keys, group_key, strict=True))
        summary.update(summarize_group(group_rows))
        summary_rows.append(summary)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {path}")


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    attempted = [row for row in rows if row["answer_behavior"] == "attempted"]
    attempted_total = len(attempted)
    answerabilities = {row["answerability"] for row in rows}

    ideal = sum(is_ideal(row) for row in rows)
    fully_supported = sum(row["faithfulness"] == "fully_supported" for row in attempted)
    correct = sum(row["correctness"] == "correct" for row in attempted)
    complete = sum(row["completeness"] == "complete" for row in attempted)

    return {
        "n": total,
        "answerability_scope": "+".join(sorted(answerabilities)),
        "ideal_count": ideal,
        "ideal_rate": rate(ideal, total),
        "attempted_count": sum(row["answer_behavior"] == "attempted" for row in rows),
        "attempted_rate": rate(sum(row["answer_behavior"] == "attempted" for row in rows), total),
        "abstained_count": sum(row["answer_behavior"] == "abstained" for row in rows),
        "abstained_rate": rate(sum(row["answer_behavior"] == "abstained" for row in rows), total),
        "unusable_count": sum(row["answer_behavior"] == "unusable" for row in rows),
        "unusable_rate": rate(sum(row["answer_behavior"] == "unusable" for row in rows), total),
        "fully_supported_attempted_count": fully_supported,
        "fully_supported_attempted_rate": rate(fully_supported, attempted_total),
        "correct_attempted_count": correct,
        "correct_attempted_rate": rate(correct, attempted_total),
        "complete_attempted_count": complete,
        "complete_attempted_rate": rate(complete, attempted_total),
    }


def is_ideal(row: dict[str, Any]) -> bool:
    if row["answerability"] == "answerable":
        return (
            row["answer_behavior"] == "attempted"
            and row["faithfulness"] == "fully_supported"
            and row["correctness"] == "correct"
            and row["completeness"] == "complete"
        )
    return row["answer_behavior"] == "abstained"

def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def default_output_prefix(labels_path: Path) -> Path:
    stem = labels_path.stem
    if stem.startswith("labels__"):
        stem = "metrics__" + stem.removeprefix("labels__")
    else:
        stem = "metrics__" + stem
    return labels_path.with_name(stem)


def selected_base_ids(base_ids: str, base_id_limit: int | None) -> set[str]:
    if base_ids and base_id_limit is not None:
        raise ValueError("Use either --base-ids or --base-id-limit, not both.")
    if base_ids:
        return {base_id.strip() for base_id in base_ids.split(",") if base_id.strip()}
    if base_id_limit is not None:
        return {f"legal-rag-bench-{index:04d}" for index in range(1, base_id_limit + 1)}
    return set()


if __name__ == "__main__":
    main()
