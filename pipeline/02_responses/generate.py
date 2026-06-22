from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from common.llm import generate_text_async
from common.records import CandidateResponse, Triple, utc_now
from common.storage import read_jsonl, read_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "02_responses"
DEFAULT_INPUT = ROOT / "pipeline" / "01_triples" / "outputs" / "triples.jsonl"
DEFAULT_PROMPT = STAGE_DIR / "prompts" / "generation_v1.txt"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "outputs"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CONCURRENCY = 48


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    prompt_template: str
    prompt_version: str
    temperature: float
    max_tokens: int
    think: bool
    concurrency: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate legal responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="LiteLLM model identifier to generate with.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum response tokens.")
    parser.add_argument("--no-think", action="store_true", help="Disable Ollama reasoning mode.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Maximum concurrent generation requests.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    triples = [Triple(**record) for record in records]
    prompt_template = read_text(args.prompt)
    responses = generate_responses(
        triples,
        model=args.model,
        prompt_template=prompt_template,
        prompt_version=args.prompt.stem,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        think=not args.no_think,
        concurrency=args.concurrency,
    )

    output = args.output or default_output_path(args.model, args.prompt.stem)
    count = write_jsonl(output, (response.to_dict() for response in responses))
    print(f"Wrote {count} candidate responses to {output}")


def generate_responses(
    triples: list[Triple],
    *,
    model: str,
    prompt_template: str,
    prompt_version: str,
    temperature: float,
    max_tokens: int,
    think: bool,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[CandidateResponse]:
    config = GenerationConfig(
        model=model,
        prompt_template=prompt_template,
        prompt_version=prompt_version,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        concurrency=concurrency,
    )
    total = len(triples)

    async def run() -> list[CandidateResponse]:
        semaphore = asyncio.Semaphore(config.concurrency)

        async def one(index: int, triple: Triple) -> CandidateResponse:
            async with semaphore:
                return await _generate_response(index=index, total=total, triple=triple, config=config)

        return list(await asyncio.gather(*(one(index, triple) for index, triple in enumerate(triples, start=1))))

    return asyncio.run(run())


async def _generate_response(
    *,
    index: int,
    total: int,
    triple: Triple,
    config: GenerationConfig,
) -> CandidateResponse:
    print(f"[{index}/{total}] Generating response for {triple.id}...", flush=True)
    prompt = config.prompt_template.format(context=triple.context, question=triple.question)
    generation = await generate_text_async(
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        think=config.think,
    )
    if generation.finish_reason == "length":
        print(f"[{index}/{total}] Warning: response hit max_tokens for {triple.id}.", flush=True)

    model_slug = slugify_model(config.model)
    return CandidateResponse(
        id=f"{triple.id}__{model_slug}__{config.prompt_version}",
        triple_id=triple.id,
        response=generation.text,
        metadata={
            "model": config.model,
            "prompt_version": config.prompt_version,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "think": config.think,
            "concurrency": config.concurrency,
            "finish_reason": generation.finish_reason,
            "created_at": utc_now(),
        },
    )


def default_output_path(model: str, prompt_version: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{slugify_model(model)}__{prompt_version}.jsonl"


def slugify_model(model: str) -> str:
    slug = model.replace(":", "_").replace(".", "_")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_")


if __name__ == "__main__":
    main()
