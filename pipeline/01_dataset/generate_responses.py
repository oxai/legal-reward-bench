from __future__ import annotations

import argparse
import re
from pathlib import Path

from common.ollama import generate as ollama_generate
from common.schema import CandidateResponse, Triple, utc_now
from common.storage import read_jsonl, read_text, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "pipeline" / "01_dataset"
DEFAULT_INPUT = STAGE_DIR / "outputs" / "triples" / "legal_rag_bench.jsonl"
DEFAULT_PROMPT = STAGE_DIR / "prompts" / "generation_v1.txt"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "outputs" / "candidate_responses"

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate legal responses.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--no-think", action="store_true", help="Disable Ollama reasoning mode.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

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
) -> list[CandidateResponse]:
    outputs: list[CandidateResponse] = []
    model_slug = slugify_model(model)

    total = len(triples)
    for index, triple in enumerate(triples, start=1):
        print(f"[{index}/{total}] Generating response for {triple.id}...", flush=True)
        prompt = prompt_template.format(context=triple.context, question=triple.question)
        answer = ollama_generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )
        outputs.append(
            CandidateResponse(
                id=f"{triple.id}__{model_slug}__{prompt_version}",
                triple_id=triple.id,
                response=answer,
                metadata={
                    "model": model,
                    "prompt_version": prompt_version,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "think": think,
                    "created_at": utc_now(),
                },
            )
        )
    return outputs

def default_output_path(model: str, prompt_version: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{slugify_model(model)}__{prompt_version}.jsonl"

def slugify_model(model: str) -> str:
    slug = model.replace(":", "_").replace(".", "_")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_")

if __name__ == "__main__":
    main()
