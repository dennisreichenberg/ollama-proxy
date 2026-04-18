"""Backend HTTP clients for Ollama and vLLM."""

from __future__ import annotations

from typing import Any

import httpx


async def fetch_models_ollama(base_url: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Return OpenAI-style model objects from an Ollama backend."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()

    models = []
    for m in data.get("models", []):
        name = m.get("model") or m.get("name", "unknown")
        models.append({"id": name, "object": "model", "owned_by": "ollama"})
    return models


async def fetch_models_vllm(base_url: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Return OpenAI-style model objects from a vLLM backend."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()

    models = []
    for m in data.get("data", []):
        models.append({"id": m.get("id", "unknown"), "object": "model", "owned_by": "vllm"})
    return models


async def chat_completions(
    base_url: str,
    backend: str,
    payload: dict[str, Any],
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Forward a /v1/chat/completions request to the target backend and return the response JSON."""
    if backend == "ollama":
        url = f"{base_url}/v1/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
