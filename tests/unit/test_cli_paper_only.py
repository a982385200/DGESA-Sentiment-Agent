from typer.testing import CliRunner

from sentiment_agent.cli import app


def test_cli_exposes_only_current_paper_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "validate-config" in result.stdout
    assert "run-paper" in result.stdout
    assert "summarize" in result.stdout
    assert "experience" not in result.stdout
    assert " run " not in result.stdout
