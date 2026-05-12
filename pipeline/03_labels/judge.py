from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from common.llm import generate_json


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    fallback_model: str | None
    temperature: float
    max_tokens: int
    timeout_seconds: int
    retries: int


@dataclass(frozen=True)
class JudgeResult:
    payload: dict[str, Any]
    model: str


class JsonJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    async def generate(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        validate: Callable[[dict[str, Any]], list[str]],
    ) -> JudgeResult:
        errors: list[str] = []
        for model_index, model in enumerate(self._models()):
            if model_index:
                print(f"Falling back to judge model {model}.", file=sys.stderr, flush=True)

            result = await self._try_model(
                model=model,
                prompt=prompt,
                schema=schema,
                validate=validate,
                errors=errors,
            )
            if result is not None:
                return result

        raise RuntimeError("Judge returned invalid label payload: " + "; ".join(errors))

    def _models(self) -> list[str]:
        models = [self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            models.append(self.config.fallback_model)
        return models

    async def _try_model(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        validate: Callable[[dict[str, Any]], list[str]],
        errors: list[str],
    ) -> JudgeResult | None:
        for attempt in range(self.config.retries + 1):
            try:
                payload = await generate_json(
                    model=model,
                    prompt=self._prompt_for_attempt(prompt, attempt, errors),
                    schema=schema,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                )
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                print(
                    f"Judge call failed for {model} on attempt {attempt + 1}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                await self._sleep_before_retry(attempt)
                continue

            validation_errors = validate(payload)
            if not validation_errors:
                return JudgeResult(payload=payload, model=model)

            errors.extend(f"{model}: {error}" for error in validation_errors)
            print(
                f"Invalid label payload from {model} on attempt {attempt + 1}: {validation_errors}",
                file=sys.stderr,
                flush=True,
            )
            await self._sleep_before_retry(attempt)

        return None

    def _prompt_for_attempt(self, prompt: str, attempt: int, errors: list[str]) -> str:
        if not attempt:
            return prompt
        return (
            prompt
            + "\n\nReturn only a valid JSON object that exactly follows the schema. "
            "Fix the consistency errors from the prior attempt: "
            + "; ".join(errors[-5:])
        )

    async def _sleep_before_retry(self, attempt: int) -> None:
        if attempt < self.config.retries:
            await asyncio.sleep(min(2**attempt, 8))
