"""Configuration loading for ollama-proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendConfig:
    url: str
    models: list[str] = field(default_factory=list)


@dataclass
class ProxyConfig:
    ollama: BackendConfig = field(
        default_factory=lambda: BackendConfig(url="http://localhost:11434")
    )
    vllm: BackendConfig = field(
        default_factory=lambda: BackendConfig(url="http://localhost:8000")
    )
    default_backend: str = "ollama"

    def backend_for_model(self, model: str) -> str:
        """Return 'vllm' or 'ollama' depending on which backend lists the model."""
        if model in self.vllm.models:
            return "vllm"
        if model in self.ollama.models:
            return "ollama"
        return self.default_backend

    def url_for_backend(self, backend: str) -> str:
        if backend == "vllm":
            return self.vllm.url.rstrip("/")
        return self.ollama.url.rstrip("/")


def load_config(path: str | Path | None = None) -> ProxyConfig:
    """Load config from YAML file or return defaults."""
    data: dict[str, Any] = {}

    if path is None:
        env_path = os.environ.get("OLLAMA_PROXY_CONFIG")
        if env_path:
            path = Path(env_path)

    if path and Path(path).exists():
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

    ollama_cfg = data.get("ollama", {})
    vllm_cfg = data.get("vllm", {})

    return ProxyConfig(
        ollama=BackendConfig(
            url=ollama_cfg.get("url", os.environ.get("OLLAMA_URL", "http://localhost:11434")),
            models=ollama_cfg.get("models", []),
        ),
        vllm=BackendConfig(
            url=vllm_cfg.get("url", os.environ.get("VLLM_URL", "http://localhost:8000")),
            models=vllm_cfg.get("models", []),
        ),
        default_backend=data.get("default_backend", "ollama"),
    )
