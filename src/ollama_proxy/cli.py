"""CLI entry point for ollama-proxy."""

from __future__ import annotations

import click
import uvicorn

from .app import set_config
from .config import load_config


@click.command()
@click.option("--port", default=4117, show_default=True, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--config", "config_path", default=None, help="Path to YAML config file.")
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama backend URL (overrides config).",
)
@click.option(
    "--vllm-url",
    default=None,
    help="vLLM backend URL (overrides config).",
)
@click.option(
    "--default-backend",
    type=click.Choice(["ollama", "vllm"]),
    default=None,
    help="Default backend when model is not explicitly mapped.",
)
def main(
    port: int,
    host: str,
    config_path: str | None,
    ollama_url: str | None,
    vllm_url: str | None,
    default_backend: str | None,
) -> None:
    """Start the ollama-proxy server."""
    cfg = load_config(config_path)

    if ollama_url:
        cfg.ollama.url = ollama_url
    if vllm_url:
        cfg.vllm.url = vllm_url
    if default_backend:
        cfg.default_backend = default_backend

    set_config(cfg)

    click.echo(f"ollama-proxy starting on http://{host}:{port}")
    click.echo(f"  Ollama backend : {cfg.ollama.url}")
    click.echo(f"  vLLM backend   : {cfg.vllm.url}")
    click.echo(f"  Default backend: {cfg.default_backend}")

    uvicorn.run("ollama_proxy.app:app", host=host, port=port)
