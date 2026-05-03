from __future__ import annotations

import ollama

OLLAMA_HOST = "http://localhost:11434"

def generate(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    think: bool,
    timeout_seconds: int = 600,
) -> str:
    try:
        response = ollama.Client(host=OLLAMA_HOST, timeout=timeout_seconds).generate(
            model=model,
            prompt=prompt,
            stream=False,
            think=think,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
    except ollama.RequestError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it with `brew services start ollama` "
            "or `ollama serve`."
        ) from exc
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama error: {exc}") from exc

    answer = response.response.strip()
    if not answer:
        done_reason = response.done_reason or "unknown"
        raise RuntimeError(
            f"Ollama returned an empty response for {model}. "
            f"done_reason={done_reason}. Reasoning tokens count against --max-tokens; "
            "try increasing it or pass --no-think."
        )
    return answer