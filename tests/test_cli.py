"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ollama_proxy.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


def test_cli_defaults(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config") as mock_set,
        patch("ollama_proxy.cli.uvicorn.run") as mock_run,
    ):
        result = runner.invoke(main, [])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs[1]["port"] == 4117
    assert call_kwargs[1]["host"] == "127.0.0.1"


def test_cli_custom_port(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config"),
        patch("ollama_proxy.cli.uvicorn.run") as mock_run,
    ):
        result = runner.invoke(main, ["--port", "9000"])
    assert result.exit_code == 0
    assert mock_run.call_args[1]["port"] == 9000


def test_cli_custom_host(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config"),
        patch("ollama_proxy.cli.uvicorn.run") as mock_run,
    ):
        result = runner.invoke(main, ["--host", "0.0.0.0"])
    assert result.exit_code == 0
    assert mock_run.call_args[1]["host"] == "0.0.0.0"


def test_cli_overrides_urls(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config") as mock_set,
        patch("ollama_proxy.cli.uvicorn.run"),
    ):
        result = runner.invoke(
            main,
            ["--ollama-url", "http://my-ollama:11434", "--vllm-url", "http://my-vllm:9000"],
        )
    assert result.exit_code == 0
    cfg = mock_set.call_args[0][0]
    assert cfg.ollama.url == "http://my-ollama:11434"
    assert cfg.vllm.url == "http://my-vllm:9000"


def test_cli_default_backend_vllm(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config") as mock_set,
        patch("ollama_proxy.cli.uvicorn.run"),
    ):
        result = runner.invoke(main, ["--default-backend", "vllm"])
    assert result.exit_code == 0
    cfg = mock_set.call_args[0][0]
    assert cfg.default_backend == "vllm"


def test_cli_with_config_file(runner: CliRunner, tmp_path):
    cfg_file = tmp_path / "proxy.yaml"
    cfg_file.write_text("ollama:\n  url: http://file-ollama:11434\n")
    with (
        patch("ollama_proxy.cli.set_config") as mock_set,
        patch("ollama_proxy.cli.uvicorn.run"),
    ):
        result = runner.invoke(main, ["--config", str(cfg_file)])
    assert result.exit_code == 0
    cfg = mock_set.call_args[0][0]
    assert cfg.ollama.url == "http://file-ollama:11434"


def test_cli_output_shows_urls(runner: CliRunner):
    with (
        patch("ollama_proxy.cli.set_config"),
        patch("ollama_proxy.cli.uvicorn.run"),
    ):
        result = runner.invoke(main, [])
    assert "ollama-proxy starting" in result.output
    assert "Ollama backend" in result.output
    assert "vLLM backend" in result.output
