from pathlib import Path

from typer.testing import CliRunner

from sentiment_agent.cli import app


def test_cli_help_lists_research_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "validate-config" in result.stdout
    assert "summarize" in result.stdout


def test_validate_config_prints_hash_without_api_key(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  name: fixture\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate-config", "--config", str(config)])

    assert result.exit_code == 0
    assert "config_hash=" in result.stdout
    assert "api_key" not in result.stdout
