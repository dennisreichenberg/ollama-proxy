"""Tests for the FastAPI application endpoints."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ollama_proxy.app import app, set_config
from ollama_proxy.config import BackendConfig, ProxyConfig

OLLAMA_MODELS = [{"id": "llama3", "object": "model", "owned_by": "ollama"}]
VLLM_MODELS = [{"id": "mistral", "object": "model", "owned_by": "vllm"}]
CHAT_RESP = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
}


@pytest.fixture(autouse=True)
def reset_config():
    cfg = ProxyConfig(
        ollama=BackendConfig(url="http://ollama:11434", models=["llama3"]),
        vllm=BackendConfig(url="http://vllm:8000", models=["mistral"]),
        default_backend="ollama",
    )
    set_config(cfg)
    yield
    set_config(cfg)


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models_both_backends(client: TestClient):
    with (
        patch("ollama_proxy.app.fetch_models_ollama", new=AsyncMock(return_value=OLLAMA_MODELS)),
        patch("ollama_proxy.app.fetch_models_vllm", new=AsyncMock(return_value=VLLM_MODELS)),
    ):
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "llama3" in ids
    assert "mistral" in ids


def test_list_models_ollama_fails(client: TestClient):
    async def boom(*a, **kw):
        raise Exception("Ollama down")

    with (
        patch("ollama_proxy.app.fetch_models_ollama", new=boom),
        patch("ollama_proxy.app.fetch_models_vllm", new=AsyncMock(return_value=VLLM_MODELS)),
    ):
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "mistral" in ids
    assert "llama3" not in ids


def test_list_models_vllm_fails(client: TestClient):
    async def boom(*a, **kw):
        raise Exception("vLLM down")

    with (
        patch("ollama_proxy.app.fetch_models_ollama", new=AsyncMock(return_value=OLLAMA_MODELS)),
        patch("ollama_proxy.app.fetch_models_vllm", new=boom),
    ):
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "llama3" in ids


def test_chat_completions_routes_to_ollama(client: TestClient):
    with patch(
        "ollama_proxy.app.chat_completions", new=AsyncMock(return_value=CHAT_RESP)
    ) as mock_cc:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 200
    mock_cc.assert_called_once()
    args = mock_cc.call_args
    assert args[0][0] == "http://ollama:11434"
    assert args[0][1] == "ollama"


def test_chat_completions_routes_to_vllm(client: TestClient):
    with patch(
        "ollama_proxy.app.chat_completions", new=AsyncMock(return_value=CHAT_RESP)
    ) as mock_cc:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "mistral", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 200
    mock_cc.assert_called_once()
    args = mock_cc.call_args
    assert args[0][0] == "http://vllm:8000"
    assert args[0][1] == "vllm"


def test_chat_completions_missing_model(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 400
    assert "model" in resp.json()["detail"]


def test_chat_completions_invalid_json(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_chat_completions_backend_http_error(client: TestClient):
    mock_response = httpx.Response(503, text="Service Unavailable")
    exc = httpx.HTTPStatusError("503", request=httpx.Request("POST", "/"), response=mock_response)

    async def raise_http(*a, **kw):
        raise exc

    with patch("ollama_proxy.app.chat_completions", new=raise_http):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 503


def test_chat_completions_backend_request_error(client: TestClient):
    async def raise_req(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    with patch("ollama_proxy.app.chat_completions", new=raise_req):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 502


def test_chat_completions_unknown_model_uses_default(client: TestClient):
    with patch(
        "ollama_proxy.app.chat_completions", new=AsyncMock(return_value=CHAT_RESP)
    ) as mock_cc:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "unknown-model", "messages": []},
        )
    assert resp.status_code == 200
    args = mock_cc.call_args
    assert args[0][1] == "ollama"
