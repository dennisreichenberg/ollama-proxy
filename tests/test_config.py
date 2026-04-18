"""Tests for config loading."""

import os
import textwrap
from pathlib import Path

import pytest

from ollama_proxy.config import BackendConfig, ProxyConfig, load_config


def test_default_config():
    cfg = load_config()
    assert cfg.ollama.url == "http://localhost:11434"
    assert cfg.vllm.url == "http://localhost:8000"
    assert cfg.default_backend == "ollama"


def test_backend_for_model_explicit_vllm():
    cfg = ProxyConfig(
        vllm=BackendConfig(url="http://vllm:8000", models=["mistral"]),
        ollama=BackendConfig(url="http://ollama:11434", models=["llama3"]),
    )
    assert cfg.backend_for_model("mistral") == "vllm"
    assert cfg.backend_for_model("llama3") == "ollama"


def test_backend_for_model_falls_back_to_default():
    cfg = ProxyConfig()
    assert cfg.backend_for_model("unknown-model") == "ollama"


def test_backend_for_model_default_vllm():
    cfg = ProxyConfig(default_backend="vllm")
    assert cfg.backend_for_model("unknown") == "vllm"


def test_url_for_backend():
    cfg = ProxyConfig(
        ollama=BackendConfig(url="http://ollama:11434/"),
        vllm=BackendConfig(url="http://vllm:8000/"),
    )
    assert cfg.url_for_backend("ollama") == "http://ollama:11434"
    assert cfg.url_for_backend("vllm") == "http://vllm:8000"


def test_load_config_from_yaml(tmp_path: Path):
    cfg_file = tmp_path / "proxy.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        ollama:
          url: http://my-ollama:11434
          models:
            - llama3
        vllm:
          url: http://my-vllm:8001
          models:
            - mistral
        default_backend: vllm
        """)
    )
    cfg = load_config(cfg_file)
    assert cfg.ollama.url == "http://my-ollama:11434"
    assert cfg.ollama.models == ["llama3"]
    assert cfg.vllm.url == "http://my-vllm:8001"
    assert cfg.vllm.models == ["mistral"]
    assert cfg.default_backend == "vllm"


def test_load_config_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_file = tmp_path / "proxy.yaml"
    cfg_file.write_text("ollama:\n  url: http://env-ollama:11434\n")
    monkeypatch.setenv("OLLAMA_PROXY_CONFIG", str(cfg_file))
    cfg = load_config()
    assert cfg.ollama.url == "http://env-ollama:11434"


def test_load_config_env_url_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_URL", "http://custom-ollama:11434")
    monkeypatch.setenv("VLLM_URL", "http://custom-vllm:9000")
    # Clear OLLAMA_PROXY_CONFIG if set
    monkeypatch.delenv("OLLAMA_PROXY_CONFIG", raising=False)
    cfg = load_config()
    assert cfg.ollama.url == "http://custom-ollama:11434"
    assert cfg.vllm.url == "http://custom-vllm:9000"


def test_load_config_missing_file():
    cfg = load_config("/nonexistent/path/proxy.yaml")
    assert cfg.ollama.url == "http://localhost:11434"


def test_load_config_empty_yaml(tmp_path: Path):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    cfg = load_config(cfg_file)
    assert cfg.default_backend == "ollama"
