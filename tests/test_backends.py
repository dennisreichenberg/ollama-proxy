"""Tests for backend HTTP client functions."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from ollama_proxy.backends import chat_completions, fetch_models_ollama, fetch_models_vllm


OLLAMA_TAGS_RESPONSE = {
    "models": [
        {"model": "llama3:latest", "name": "llama3:latest"},
        {"name": "phi3"},
    ]
}

VLLM_MODELS_RESPONSE = {
    "data": [
        {"id": "mistral-7b"},
        {"id": "qwen2"},
    ]
}

CHAT_RESPONSE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
}


def make_mock_response(json_data, status_code=200):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.mark.asyncio
async def test_fetch_models_ollama():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=make_mock_response(OLLAMA_TAGS_RESPONSE))

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        models = await fetch_models_ollama("http://localhost:11434")

    assert len(models) == 2
    assert models[0]["id"] == "llama3:latest"
    assert models[0]["owned_by"] == "ollama"
    assert models[1]["id"] == "phi3"


@pytest.mark.asyncio
async def test_fetch_models_ollama_empty():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=make_mock_response({}))

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        models = await fetch_models_ollama("http://localhost:11434")

    assert models == []


@pytest.mark.asyncio
async def test_fetch_models_vllm():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=make_mock_response(VLLM_MODELS_RESPONSE))

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        models = await fetch_models_vllm("http://localhost:8000")

    assert len(models) == 2
    assert models[0]["id"] == "mistral-7b"
    assert models[0]["owned_by"] == "vllm"


@pytest.mark.asyncio
async def test_fetch_models_vllm_empty():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=make_mock_response({}))

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        models = await fetch_models_vllm("http://localhost:8000")

    assert models == []


@pytest.mark.asyncio
async def test_chat_completions_ollama():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=make_mock_response(CHAT_RESPONSE))

    payload = {"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]}

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        result = await chat_completions("http://localhost:11434", "ollama", payload)

    assert result["id"] == "chatcmpl-abc"
    mock_client.post.assert_called_once_with(
        "http://localhost:11434/v1/chat/completions", json=payload
    )


@pytest.mark.asyncio
async def test_chat_completions_vllm():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=make_mock_response(CHAT_RESPONSE))

    payload = {"model": "mistral", "messages": [{"role": "user", "content": "Hi"}]}

    with patch("ollama_proxy.backends.httpx.AsyncClient", return_value=mock_client):
        result = await chat_completions("http://localhost:8000", "vllm", payload)

    assert result["id"] == "chatcmpl-abc"
