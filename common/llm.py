from __future__ import annotations

import json
import os
from typing import Any

from litellm import acompletion, completion, embedding
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def normalize_model_name(model: str) -> str:
    if "/" in model:
        return model
    return f"ollama_chat/{model}"


def normalize_embedding_model_name(model: str) -> str:
    if "/" in model:
        return model
    return f"ollama/{model}"


def completion_kwargs(model: str) -> dict[str, Any]:
    normalized = normalize_model_name(model)
    kwargs: dict[str, Any] = {"model": normalized}
    if normalized.startswith(("ollama/", "ollama_chat/")):
        kwargs["api_base"] = OLLAMA_HOST
    return kwargs


def embedding_kwargs(model: str) -> dict[str, Any]:
    normalized = normalize_embedding_model_name(model)
    kwargs: dict[str, Any] = {"model": normalized}
    if normalized.startswith("ollama/"):
        kwargs["api_base"] = OLLAMA_HOST
    return kwargs


def generate_text(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    think: bool = False,
    timeout_seconds: int = 600,
) -> str:
    kwargs = completion_kwargs(model)
    if kwargs["model"].startswith(("ollama/", "ollama_chat/")):
        kwargs["think"] = think

    response = completion(
        **kwargs,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError(f"LLM returned an empty response for {model}.")
    return text.strip()


async def generate_json(
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    response = await acompletion(
        **completion_kwargs(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name(schema),
                "schema": schema,
                "strict": True,
            },
        },
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError(f"LLM returned an empty JSON response for {model}.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON for {model}: {text[:500]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"LLM returned JSON that is not an object for {model}.")
    return parsed


def embed_texts(
    *,
    model: str,
    texts: list[str],
    batch_size: int = 64,
    timeout_seconds: int = 900,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = embedding(
            **embedding_kwargs(model),
            input=batch,
            timeout=timeout_seconds,
        )
        vectors.extend([item["embedding"] for item in response.data])
    return vectors


def schema_name(schema: dict[str, Any]) -> str:
    return str(schema.get("title") or "label_schema")
