from pathlib import Path

from typer.testing import CliRunner

from sentiment_agent.cli import app


def test_validate_config_does_not_require_api_key(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("""
model:
  name: qwen-plus
  base_url: https://example.invalid/v1
  api_key_env: MISSING_KEY
embedding:
  model_id: BAAI/bge-m3
experiment:
  train_paths: []
  dev_paths: []
  test_paths: []
  output_root: outputs
  train_batch_size: 2
""", encoding="utf-8")
    result = CliRunner().invoke(app, ["validate-config", "--config", str(config)])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()


def test_experience_stats_reads_sqlite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    from sentiment_agent.experience.repository import ExperienceRepository
    with ExperienceRepository(run_dir / "experience_store" / "experiences.sqlite3"):
        pass
    result = CliRunner().invoke(app, ["experience", "stats", "--run", str(run_dir)])
    assert result.exit_code == 0
    assert "0" in result.stdout


def test_run_help_exposes_no_progress_switch() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--no-progress" in result.stdout
