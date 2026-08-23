from __future__ import annotations

import json
from pathlib import Path

import typer

from sentiment_agent.config import AppConfig
from sentiment_agent.workflow import run_from_config

app = typer.Typer(no_args_is_help=True, help="Run reproducible ASEAN sentiment-agent experiments.")


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Validate an experiment configuration without contacting an API."""
    parsed = AppConfig.load(config)
    typer.echo(f"config_hash={parsed.config_hash}")


@app.command("run")
def run_experiment(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    output_root: Path | None = typer.Option(None, "--output-root"),
) -> None:
    """Run an experiment using the configured OpenAI-compatible API."""
    summary = run_from_config(config, output_root=output_root)
    typer.echo(f"output_dir={summary.output_dir}")
    typer.echo(f"config_hash={summary.config_hash}")


@app.command("summarize")
def summarize(
    output_dir: Path = typer.Option(..., "--output-dir", exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Print the saved metrics for one experiment run."""
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise typer.BadParameter(f"metrics file does not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
