from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from common.records import ResponseLabel, utc_now
from common.storage import read_jsonl, read_text
from judge import JsonJudge, JudgeConfig
from rubric import (
    DOWNSTREAM_LABELS,
    RUBRIC_VERSION,
    validate_behavior_payload,
    validate_final_label_payload,
    validate_semantic_payload,
)

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "03_labels"
DEFAULT_TRIPLES = ROOT / "pipeline" / "01_triples" / "outputs" / "triples.jsonl"
DEFAULT_BEHAVIOR_PROMPT = STAGE_DIR / "prompts" / "answer_behavior_v1.txt"
DEFAULT_SEMANTIC_PROMPT = STAGE_DIR / "prompts" / "semantic_labels_v1.txt"
DEFAULT_BEHAVIOR_SCHEMA = STAGE_DIR / "schemas" / "answer_behavior_v1.json"
DEFAULT_SEMANTIC_SCHEMA = STAGE_DIR / "schemas" / "semantic_labels_v1.json"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "outputs"
DEFAULT_FALLBACK_JUDGE_MODEL = "gemini/gemini-2.5-flash"

__all__ = ["build_gold_context_map", "resolve_gold_context"]


@dataclass(frozen=True)
class LabelConfig:
    judge_model: str
    fallback_judge_model: str | None
    temperature: float
    max_tokens: int
    timeout_seconds: int
    retries: int
    concurrency: int

    def judge_config(self) -> JudgeConfig:
        return JudgeConfig(
            model=self.judge_model,
            fallback_model=self.fallback_judge_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )


@dataclass(frozen=True)
class LabelResources:
    behavior_prompt_template: str
    semantic_prompt_template: str
    behavior_schema: dict[str, Any]
    semantic_schema: dict[str, Any]


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    args = parse_args()
    resources = load_resources()
    config = LabelConfig(
        judge_model=args.judge_model,
        fallback_judge_model=args.fallback_judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        concurrency=args.concurrency,
    )
    triples = {row["id"]: row for row in read_jsonl(args.triples)}
    responses = read_unique_jsonl(args.responses, id_key="id", label="responses")
    if args.limit is not None:
        responses = responses[: args.limit]

    output = args.output or default_output_path(config.judge_model)
    existing_labels = read_existing_labels(output)
    if existing_labels:
        labeled_response_ids = {label["response_id"] for label in existing_labels}
        responses = [response for response in responses if response["id"] not in labeled_response_ids]
        print(
            f"Found {len(existing_labels)} existing labels in {output}; "
            f"resuming with {len(responses)} remaining responses.",
            flush=True,
        )

    labels = await label_responses(
        triples=triples,
        responses=responses,
        resources=resources,
        config=config,
        on_label=lambda label: append_label(output, label),
    )
    print(f"Wrote {len(labels)} new response labels to {output}")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--fallback-judge-model", default=DEFAULT_FALLBACK_JUDGE_MODEL)
    return parser.parse_args()


def load_resources() -> LabelResources:
    return LabelResources(
        behavior_prompt_template=read_text(DEFAULT_BEHAVIOR_PROMPT),
        semantic_prompt_template=read_text(DEFAULT_SEMANTIC_PROMPT),
        behavior_schema=json.loads(DEFAULT_BEHAVIOR_SCHEMA.read_text(encoding="utf-8")),
        semantic_schema=json.loads(DEFAULT_SEMANTIC_SCHEMA.read_text(encoding="utf-8")),
    )


async def label_responses(
    *,
    triples: dict[str, dict[str, Any]],
    responses: list[dict[str, Any]],
    resources: LabelResources,
    config: LabelConfig,
    on_label: Callable[[ResponseLabel], None] | None = None,
) -> list[ResponseLabel]:
    response_labeler = ResponseLabeler(
        triples=triples,
        gold_contexts=build_gold_context_map(triples),
        resources=resources,
        config=config,
        judge=JsonJudge(config.judge_config()),
    )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_one(index: int, response: dict[str, Any]) -> ResponseLabel:
        async with semaphore:
            print(f"[{index}/{len(responses)}] Labeling {response['id']}...", flush=True)
            return await response_labeler.label(response)

    tasks = [asyncio.create_task(run_one(index, response)) for index, response in enumerate(responses, start=1)]
    labels: list[ResponseLabel] = []
    for task in asyncio.as_completed(tasks):
        label = await task
        labels.append(label)
        if on_label is not None:
            on_label(label)
    return labels


class ResponseLabeler:
    def __init__(
        self,
        *,
        triples: dict[str, dict[str, Any]],
        gold_contexts: dict[str, str],
        resources: LabelResources,
        config: LabelConfig,
        judge: JsonJudge,
    ) -> None:
        self.triples = triples
        self.gold_contexts = gold_contexts
        self.resources = resources
        self.config = config
        self.judge = judge

    async def label(self, response: dict[str, Any]) -> ResponseLabel:
        try:
            triple = self._triple_for_response(response)
            behavior = await self.judge.generate(
                prompt=self.resources.behavior_prompt_template.format(
                    question=triple["question"],
                    candidate_response=response["response"],
                ),
                schema=self.resources.behavior_schema,
                validate=validate_behavior_payload,
            )

            answer_behavior = behavior.payload["answer_behavior"]
            if answer_behavior == "attempted":
                semantic = await self.judge.generate(
                    prompt=self.resources.semantic_prompt_template.format(
                        question=triple["question"],
                        gold_context=resolve_gold_context(triple, self.gold_contexts),
                        candidate_context=triple["context"],
                        reference_answer=triple["answer"],
                        candidate_response=response["response"],
                    ),
                    schema=self.resources.semantic_schema,
                    validate=validate_semantic_payload,
                )
                semantic_judge_model = semantic.model
                label_payload = {"answer_behavior": answer_behavior, **semantic.payload}
            else:
                semantic_judge_model = None
                label_payload = {
                    "answer_behavior": answer_behavior,
                    **{key: "not_applicable" for key in DOWNSTREAM_LABELS},
                }

            validation_errors = validate_final_label_payload(label_payload)
            if validation_errors:
                raise RuntimeError(f"Internal final label error for {response['id']}: {validation_errors}")

            return self._response_label(
                response=response,
                triple=triple,
                labels=label_payload,
                behavior_judge_model=behavior.model,
                semantic_judge_model=semantic_judge_model,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to label {response['id']}") from exc

    def _triple_for_response(self, response: dict[str, Any]) -> dict[str, Any]:
        triple = self.triples.get(response["triple_id"])
        if triple is None:
            raise ValueError(f"Response {response['id']} references missing triple {response['triple_id']}")
        return triple

    def _response_label(
        self,
        *,
        response: dict[str, Any],
        triple: dict[str, Any],
        labels: dict[str, Any],
        behavior_judge_model: str,
        semantic_judge_model: str | None,
    ) -> ResponseLabel:
        metadata = triple.get("metadata", {})
        return ResponseLabel(
            id=f"{response['id']}__{slugify_model(self.config.judge_model)}__{RUBRIC_VERSION}",
            response_id=response["id"],
            triple_id=response["triple_id"],
            labels=labels,
            metadata={
                "judge_model": self.config.judge_model,
                "fallback_judge_model": self.config.fallback_judge_model,
                "behavior_judge_model": behavior_judge_model,
                "semantic_judge_model": semantic_judge_model,
                "rubric_version": RUBRIC_VERSION,
                "schema_version": RUBRIC_VERSION,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "concurrency": self.config.concurrency,
                "answerability": metadata.get("answerability", "answerable"),
                "context_variant": metadata.get("context_variant", "base"),
                "created_at": utc_now(),
            },
        )


def build_gold_context_map(triples: dict[str, dict[str, Any]]) -> dict[str, str]:
    gold_contexts: dict[str, str] = {}
    for triple in triples.values():
        metadata = triple.get("metadata", {})
        base_triple_id = metadata.get("base_triple_id", triple["id"])
        if metadata.get("context_variant") == "base":
            gold_contexts[base_triple_id] = triple["context"]

    expected_base_ids = {
        triple.get("metadata", {}).get("base_triple_id", triple["id"]) for triple in triples.values()
    }
    missing = sorted(expected_base_ids - set(gold_contexts))
    if missing:
        raise ValueError(
            "Could not resolve authoritative base context for "
            f"{len(missing)} base triples. Ensure --triples includes context_variant='base' rows. "
            f"Examples: {missing[:5]}"
        )

    return gold_contexts


def resolve_gold_context(triple: dict[str, Any], gold_contexts: dict[str, str]) -> str:
    metadata = triple.get("metadata", {})
    base_triple_id = metadata.get("base_triple_id", triple["id"])
    if base_triple_id in gold_contexts:
        return gold_contexts[base_triple_id]
    raise ValueError(f"Could not resolve authoritative gold context for {triple['id']}")


def read_unique_jsonl(path: Path, *, id_key: str, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"--{label} must point to a single JSONL file, got: {path}")
    rows = read_jsonl(path)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        row_id = row[id_key]
        if row_id in seen:
            duplicates.append(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"Duplicate {label} ids in {path}: {duplicates[:10]}")
    return rows


def read_existing_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_unique_jsonl(path, id_key="response_id", label="existing labels")


def append_label(path: Path, label: ResponseLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(label.to_dict(), ensure_ascii=False))
        file.write("\n")


def default_output_path(judge_model: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"pointwise_labels__{RUBRIC_VERSION}__{slugify_model(judge_model)}.jsonl"


def slugify_model(model: str) -> str:
    return model.replace(":", "_").replace(".", "_").replace("/", "_").strip("_")


if __name__ == "__main__":
    main()
