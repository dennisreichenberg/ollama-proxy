"""FastAPI application exposing the OpenAI-compatible proxy endpoints."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .backends import chat_completions, fetch_models_ollama, fetch_models_vllm
from .config import ProxyConfig, load_config

app = FastAPI(title="ollama-proxy", version="0.1.0")

_config: ProxyConfig | None = None


def get_config() -> ProxyConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: ProxyConfig) -> None:
    global _config
    _config = cfg


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    cfg = get_config()
    all_models: list[dict[str, Any]] = []

    try:
        all_models += await fetch_models_ollama(cfg.ollama.url)
    except Exception:
        pass

    try:
        all_models += await fetch_models_vllm(cfg.vllm.url)
    except Exception:
        pass

    return JSONResponse({"object": "list", "data": all_models})


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request) -> JSONResponse:
    cfg = get_config()
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = payload.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    backend = cfg.backend_for_model(model)
    base_url = cfg.url_for_backend(backend)

    try:
        result = await chat_completions(base_url, backend, payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {exc}")

    return JSONResponse(result)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
