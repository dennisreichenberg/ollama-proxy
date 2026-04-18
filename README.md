# ollama-proxy

A local API proxy/router that forwards requests between **Ollama** and **vLLM** backends behind a single, unified **OpenAI-compatible** REST API.

## Why

Apps that speak the OpenAI protocol (`/v1/chat/completions`, `/v1/models`) can now target a single endpoint regardless of which local LLM backend serves a given model.

## Installation

```bash
pip install -e .
```

## Quick start

```bash
# Start with defaults (Ollama on :11434, vLLM on :8000, proxy on :4117)
ollama-proxy

# Custom ports / hosts
ollama-proxy --port 8080 --host 0.0.0.0

# Override backend URLs
ollama-proxy --ollama-url http://my-server:11434 --vllm-url http://my-vllm:8001

# Change default backend (used when a model is not explicitly mapped)
ollama-proxy --default-backend vllm

# Use a YAML config file
ollama-proxy --config proxy.yaml
```

## YAML configuration

```yaml
# proxy.yaml
ollama:
  url: http://localhost:11434
  models:
    - llama3
    - phi3

vllm:
  url: http://localhost:8000
  models:
    - mistral-7b
    - qwen2

default_backend: ollama
```

Models listed under `ollama.models` are always forwarded to the Ollama backend.
Models listed under `vllm.models` are forwarded to vLLM.
Any other model falls back to `default_backend`.

You can also set `OLLAMA_PROXY_CONFIG` (path), `OLLAMA_URL`, and `VLLM_URL` as environment variables.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | Aggregated model list from all backends |
| POST | `/v1/chat/completions` | Routed chat completion |
| GET | `/health` | Liveness probe |

## Routing logic

1. If `model` is in `vllm.models` -- forward to vLLM.
2. If `model` is in `ollama.models` -- forward to Ollama.
3. Otherwise -- forward to `default_backend`.

## Development

```bash
pip install -e ".[dev]"
pytest --cov=ollama_proxy
```

Coverage is >= 80% (currently 99%).
