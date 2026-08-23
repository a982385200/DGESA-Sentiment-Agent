from pathlib import Path

import pytest

from sentiment_agent.config import config_hash, load_config, redacted_config


PUBLIC_CONFIG = Path("configs/dgesa_paper.example.yaml")


def _write_config(path: Path, *, batch_size: int = 2) -> None:
    path.write_text(f"""
model:
  name: qwen-plus
  base_url: https://example.invalid/v1
  api_key_env: TEST_KEY
embedding:
  model_id: models/embeddings/bge-m3
experiment:
  train_paths: []
  test_paths: []
  output_root: outputs
  train_batch_size: {batch_size}
""", encoding="utf-8")


def test_load_config_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    _write_config(path, batch_size=0)
    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_rejects_unknown_legacy_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    _write_config(path)
    path.write_text(path.read_text(encoding="utf-8") + "  dev_paths: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_redacted_config_contains_key_name_but_never_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path)
    monkeypatch.setenv("TEST_KEY", "secret-value")
    payload = str(redacted_config(load_config(path)))
    assert "TEST_KEY" in payload
    assert "secret-value" not in payload


def test_config_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path)
    loaded = load_config(path)
    assert config_hash(loaded) == config_hash(loaded)


def test_public_config_is_valid_and_paper_aligned() -> None:
    config = load_config(PUBLIC_CONFIG)
    assert config.dgesa.enabled is True
    assert config.model.name == "qwen-plus"
    assert config.model.api_key_env == "OPENAI_API_KEY"
    assert config.embedding.model_id == "/path/to/your/embedding/model"
    assert config.experiment.train_paths == [
        Path("datasets/mini_dataset/vietnamese/train.json")]
    assert config.experiment.test_paths == [
        Path("datasets/mini_dataset/vietnamese/test.json")]


def test_readme_documents_public_config() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert f"--config {PUBLIC_CONFIG.as_posix()}" in readme
