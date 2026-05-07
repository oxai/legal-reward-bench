from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

from common.llm import generate_json
from common.records import ResponseLabel, utc_now
from common.storage import read_jsonl, read_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "02_preference"
DEFAULT_TRIPLES = ROOT / "pipeline" / "01_dataset" / "outputs" / "triples" / "legal_rag_bench.jsonl"
DEFAULT_BEHAVIOR_PROMPT = STAGE_DIR / "prompts" / "answer_behavior_v1.txt"
DEFAULT_SEMANTIC_PROMPT = STAGE_DIR / "prompts" / "semantic_labels_v1.txt"
DEFAULT_BEHAVIOR_SCHEMA = STAGE_DIR / "schemas" / "answer_behavior_v1.json"
DEFAULT_SEMANTIC_SCHEMA = STAGE_DIR / "schemas" / "semantic_labels_v1.json"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "outputs"
RUBRIC_VERSION = "answer_behavior_v1+semantic_labels_v1"

FINAL_LABEL_VALUES = {
    "answer_behavior": {"attempted", "abstained", "unusable"},
    "faithfulness": {
        "fully_supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "not_applicable",
    },
    "legal_conclusion": {"correct", "incorrect", "not_applicable"},
    "completeness": {"complete", "incomplete", "not_applicable"},
    "conciseness": {"concise", "acceptable", "not_concise", "not_applicable"},
}
BEHAVIOR_VALUES = {"answer_behavior": FINAL_LABEL_VALUES["answer_behavior"]}
SEMANTIC_VALUES = {key: values - {"not_applicable"} for key, values in FINAL_LABEL_VALUES.items() if key != "answer_behavior"}
REQUIRED_FINAL_LABELS = tuple(FINAL_LABEL_VALUES)
DOWNSTREAM_LABELS = REQUIRED_FINAL_LABELS[1:]


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Label candidate responses for preference data.")
    parser.add_argument("--judge-model", default="ministral-3:14b")
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    behavior_schema = json.loads(DEFAULT_BEHAVIOR_SCHEMA.read_text(encoding="utf-8"))
    semantic_schema = json.loads(DEFAULT_SEMANTIC_SCHEMA.read_text(encoding="utf-8"))
    behavior_prompt_template = read_text(DEFAULT_BEHAVIOR_PROMPT)
    semantic_prompt_template = read_text(DEFAULT_SEMANTIC_PROMPT)
    triples = {row["id"]: row for row in read_jsonl(args.triples)}
    responses = read_response_file(args.responses)
    if args.limit is not None:
        responses = responses[: args.limit]

    output = args.output or default_output_path(args.judge_model)
    labels = await label_responses(
        triples=triples,
        responses=responses,
        behavior_prompt_template=behavior_prompt_template,
        semantic_prompt_template=semantic_prompt_template,
        behavior_schema=behavior_schema,
        semantic_schema=semantic_schema,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        concurrency=args.concurrency,
    )
    count = write_jsonl(output, (label.to_dict() for label in labels))
    print(f"Wrote {count} response labels to {output}")


def read_response_file(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"--responses must point to a single JSONL file, got: {path}")
    rows = read_jsonl(path)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        response_id = row["id"]
        if response_id in seen:
            duplicates.append(response_id)
        seen.add(response_id)
    if duplicates:
        raise ValueError(f"Duplicate response ids in {path}: {duplicates[:10]}")
    return rows


async def label_responses(
    *,
    triples: dict[str, dict[str, Any]],
    responses: list[dict[str, Any]],
    behavior_prompt_template: str,
    semantic_prompt_template: str,
    behavior_schema: dict[str, Any],
    semantic_schema: dict[str, Any],
    judge_model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
    concurrency: int,
) -> list[ResponseLabel]:
    total = len(responses)
    gold_contexts = build_gold_context_map(triples)
    semaphore = asyncio.Semaphore(concurrency)

    async def label_one(index: int, response: dict[str, Any]) -> ResponseLabel:
        async with semaphore:
            triple = triples.get(response["triple_id"])
            if triple is None:
                raise ValueError(f"Response {response['id']} references missing triple {response['triple_id']}")

            print(f"[{index}/{total}] Labeling {response['id']}...", flush=True)
            answerability = triple.get("metadata", {}).get("answerability", "answerable")
            context_variant = triple.get("metadata", {}).get("context_variant", "gold_context")
            gold_context = resolve_gold_context(triple, gold_contexts)
            behavior_prompt = behavior_prompt_template.format(
                question=triple["question"],
                candidate_response=response["response"],
            )
            behavior_payload = await generate_valid_payload(
                prompt=behavior_prompt,
                schema=behavior_schema,
                judge_model=judge_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                retries=retries,
                validate=validate_behavior_payload,
            )
            answer_behavior = behavior_payload["answer_behavior"]
            if answer_behavior == "attempted":
                semantic_prompt = semantic_prompt_template.format(
                    question=triple["question"],
                    gold_context=gold_context,
                    candidate_context=triple["context"],
                    reference_answer=triple["answer"],
                    candidate_response=response["response"],
                )
                semantic_payload = await generate_valid_payload(
                    prompt=semantic_prompt,
                    schema=semantic_schema,
                    judge_model=judge_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    validate=validate_semantic_payload,
                )
                label_payload = {"answer_behavior": answer_behavior, **semantic_payload}
            else:
                label_payload = {
                    "answer_behavior": answer_behavior,
                    **{key: "not_applicable" for key in DOWNSTREAM_LABELS},
                }
            final_validation_errors = validate_final_label_payload(label_payload)
            if final_validation_errors:
                raise RuntimeError(f"Internal final label error for {response['id']}: {final_validation_errors}")

            return ResponseLabel(
                id=f"{response['id']}__{slugify_model(judge_model)}__{RUBRIC_VERSION}",
                response_id=response["id"],
                triple_id=response["triple_id"],
                labels=label_payload,
                metadata={
                    "judge_model": judge_model,
                    "rubric_version": RUBRIC_VERSION,
                    "schema_version": RUBRIC_VERSION,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "concurrency": concurrency,
                    "answerability": answerability,
                    "context_variant": context_variant,
                    "created_at": utc_now(),
                },
            )

    tasks = [label_one(index, response) for index, response in enumerate(responses, start=1)]
    return list(await asyncio.gather(*tasks))


def build_gold_context_map(triples: dict[str, dict[str, Any]]) -> dict[str, str]:
    gold_contexts: dict[str, str] = {}
    for triple in triples.values():
        metadata = triple.get("metadata", {})
        context_variant = metadata.get("context_variant", "gold_context")
        base_triple_id = metadata.get("base_triple_id", triple["id"])
        if context_variant in {"gold_context", "gold_exact"}:
            gold_contexts[base_triple_id] = triple["context"]
    return gold_contexts


def resolve_gold_context(triple: dict[str, Any], gold_contexts: dict[str, str]) -> str:
    metadata = triple.get("metadata", {})
    base_triple_id = metadata.get("base_triple_id", triple["id"])
    if base_triple_id in gold_contexts:
        return gold_contexts[base_triple_id]
    return triple["context"]


async def generate_valid_payload(
    *,
    prompt: str,
    schema: dict[str, Any],
    judge_model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
    validate: Callable[[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(retries + 1):
        prompt_for_attempt = prompt
        if attempt:
            prompt_for_attempt += (
                "\n\nReturn only a valid JSON object that exactly follows the schema. "
                "Fix the consistency errors from the prior attempt: "
                + "; ".join(errors[-5:])
            )

        payload = await generate_json(
            model=judge_model,
            prompt=prompt_for_attempt,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        validation_errors = validate(payload)
        if not validation_errors:
            return payload
        errors.extend(validation_errors)
        print(
            f"Invalid label payload on attempt {attempt + 1}: {validation_errors}",
            file=sys.stderr,
            flush=True,
        )

    raise RuntimeError("Judge returned invalid label payload: " + "; ".join(errors))


def validate_behavior_payload(label: dict[str, Any]) -> list[str]:
    return validate_keys_and_values(label, BEHAVIOR_VALUES)


def validate_semantic_payload(label: dict[str, Any]) -> list[str]:
    return validate_keys_and_values(label, SEMANTIC_VALUES)


def validate_keys_and_values(label: dict[str, Any], allowed_values: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    if set(label) != set(allowed_values):
        errors.append(f"payload must contain exactly these keys: {list(allowed_values)}")

    for key, values in allowed_values.items():
        value = label.get(key)
        if value not in values:
            errors.append(f"{key} must be one of {sorted(values)}, got {value!r}")

    return errors


def validate_final_label_payload(label: dict[str, Any]) -> list[str]:
    errors = validate_keys_and_values(label, FINAL_LABEL_VALUES)

    answer_behavior = label.get("answer_behavior")
    if answer_behavior == "attempted":
        for key in DOWNSTREAM_LABELS:
            if label.get(key) == "not_applicable":
                errors.append(f"{key} must not be 'not_applicable' for attempted answers")
    elif answer_behavior in {"abstained", "unusable"}:
        for key in DOWNSTREAM_LABELS:
            if label.get(key) != "not_applicable":
                errors.append(f"{key} must be 'not_applicable' for {answer_behavior} responses")

    return errors


def default_output_path(judge_model: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"pointwise_labels__{RUBRIC_VERSION}__{slugify_model(judge_model)}.jsonl"


def slugify_model(model: str) -> str:
    return model.replace(":", "_").replace(".", "_").replace("/", "_").strip("_")


if __name__ == "__main__":
    main()
