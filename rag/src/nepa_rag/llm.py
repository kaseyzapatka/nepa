from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Iterator

import requests

from .config import RagConfig


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


def generate_text(config: RagConfig, prompt: str, *, model: str | None = None) -> LLMResponse:
    if config.llm_provider != "ollama":
        raise LLMError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")

    selected_model = model or config.llm_model
    try:
        return _ollama_generate(config, prompt, selected_model)
    except LLMError:
        if config.llm_fallback_model and config.llm_fallback_model != selected_model:
            return _ollama_generate(config, prompt, config.llm_fallback_model)
        raise


def stream_text(config: RagConfig, prompt: str, *, model: str | None = None) -> Iterator[str]:
    if config.llm_provider != "ollama":
        raise LLMError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")

    selected_model = model or config.llm_model
    emitted = False
    try:
        for token in _ollama_stream(config, prompt, selected_model):
            emitted = True
            yield token
    except LLMError:
        if (
            not emitted
            and config.llm_fallback_model
            and config.llm_fallback_model != selected_model
        ):
            yield from _ollama_stream(config, prompt, config.llm_fallback_model)
            return
        raise


def _ollama_payload(config: RagConfig, prompt: str, model: str, *, stream: bool) -> dict[str, object]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0,
            "num_ctx": config.llm_num_ctx,
            "num_predict": config.llm_num_predict,
        },
    }


def _ollama_generate(config: RagConfig, prompt: str, model: str) -> LLMResponse:
    url = f"{config.ollama_host}/api/generate"
    payload = _ollama_payload(config, prompt, model, stream=False)
    try:
        response = requests.post(url, json=payload, timeout=config.llm_timeout_seconds)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise LLMError(
            f"Ollama timed out after {config.llm_timeout_seconds} seconds for model {model}. "
            "The server is reachable, but the model did not return text in time. "
            "Try the fast model, ask a narrower question, or retry while the model is warm."
        ) from exc
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Could not connect to Ollama at {config.ollama_host}. "
            "Start Ollama, then try again."
        ) from exc
    except requests.HTTPError as exc:
        raise LLMError(
            f"Ollama returned HTTP {exc.response.status_code} for model {model}. "
            f"Confirm `ollama pull {model}` has completed."
        ) from exc
    except requests.RequestException as exc:
        raise LLMError(
            f"Ollama request failed for model {model}. "
            f"Confirm Ollama is running and `ollama pull {model}` has completed."
        ) from exc

    data = response.json()
    text = str(data.get("response", "")).strip()
    if not text:
        raise LLMError(f"Ollama returned an empty response for model {model}.")
    return LLMResponse(text=text, model=model)


def _ollama_stream(config: RagConfig, prompt: str, model: str) -> Iterator[str]:
    url = f"{config.ollama_host}/api/generate"
    payload = _ollama_payload(config, prompt, model, stream=True)
    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=config.llm_timeout_seconds,
        ) as response:
            response.raise_for_status()
            saw_text = False
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise LLMError(str(data["error"]))
                token = str(data.get("response", ""))
                if token:
                    saw_text = True
                    yield token
                if data.get("done"):
                    break
            if not saw_text:
                raise LLMError(f"Ollama returned an empty streamed response for model {model}.")
    except requests.Timeout as exc:
        raise LLMError(
            f"Ollama timed out after {config.llm_timeout_seconds} seconds for model {model}. "
            "The server is reachable, but no streamed text arrived in time. "
            "Try the fast model, ask a narrower question, or retry while the model is warm."
        ) from exc
    except requests.ConnectionError as exc:
        raise LLMError(
            f"Could not connect to Ollama at {config.ollama_host}. "
            "Start Ollama, then try again."
        ) from exc
    except requests.HTTPError as exc:
        raise LLMError(
            f"Ollama returned HTTP {exc.response.status_code} for model {model}. "
            f"Confirm `ollama pull {model}` has completed."
        ) from exc
    except requests.RequestException as exc:
        raise LLMError(
            f"Ollama request failed for model {model}. "
            f"Confirm Ollama is running and `ollama pull {model}` has completed."
        ) from exc
