from __future__ import annotations

import os
import time
from typing import Any

import requests

from .schema import PREDICTION_SCHEMA


API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def required_key(provider: str) -> str:
    try:
        return API_KEY_ENV[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc


def _key(provider: str) -> str:
    env_name = required_key(provider)
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"{env_name} is not set")
    return value


def _post(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Provider returned a non-object JSON response")
    return body


def call_openai(model_id: str, system: str, user: str, timeout: int = 90):
    payload = {
        "model": model_id,
        "store": False,
        "instructions": system,
        "input": user,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "revelio_job_posting_judgment",
                "strict": True,
                "schema": PREDICTION_SCHEMA,
            }
        },
    }
    body = _post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {_key('openai')}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
    )
    chunks = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    text = "".join(chunks)
    if not text:
        raise ValueError("OpenAI response contained no output text")
    return text, body.get("usage", {}), body


def call_mistral(model_id: str, system: str, user: str, timeout: int = 90):
    payload = {
        "model": model_id,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "revelio_job_posting_judgment", "schema": PREDICTION_SCHEMA},
        },
    }
    body = _post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {_key('mistral')}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
    )
    text = body["choices"][0]["message"]["content"]
    return text, body.get("usage", {}), body


def call_gemini(model_id: str, system: str, user: str, timeout: int = 90):
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "responseFormat": {"text": {"mimeType": "application/json", "schema": PREDICTION_SCHEMA}},
        },
    }
    body = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": _key("gemini")},
        payload=payload,
        timeout=timeout,
    )
    parts = body["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text:
        raise ValueError("Gemini response contained no output text")
    return text, body.get("usageMetadata", {}), body


def call(provider: str, model_id: str, system: str, user: str, timeout: int = 90):
    started = time.perf_counter()
    if provider == "openai":
        output = call_openai(model_id, system, user, timeout)
    elif provider == "mistral":
        output = call_mistral(model_id, system, user, timeout)
    elif provider == "gemini":
        output = call_gemini(model_id, system, user, timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return (*output, time.perf_counter() - started)
