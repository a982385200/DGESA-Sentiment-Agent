import json
from pathlib import Path

import pytest

from sentiment_agent.config import config_hash, load_config, redacted_config


def _write_config(path: Path, *, batch_size: int = 2) -> None:
    path.write_text(
        f"""
model:
  name: qwen-plus
  base_url: https://example.invalid/v1
  api_key_env: QWEN_API_KEY
embedding:
  model_id: BAAI/bge-m3
experiment:
  train_paths: [train.json]
  dev_paths: [dev.json]
  test_paths: [test.json]
  output_root: outputs
  train_batch_size: {batch_size}
""",
        encoding="utf-8",
    )


def test_load_config_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    _write_config(path, batch_size=0)

    with pytest.raises(ValueError, match="train_batch_size"):
        load_config(path)


def test_redacted_config_never_reads_or_contains_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "valid.yaml"
    _write_config(path)
    monkeypatch.setenv("QWEN_API_KEY", "super-secret")

    rendered = json.dumps(redacted_config(load_config(path)))

    assert "super-secret" not in rendered
    assert "QWEN_API_KEY" in rendered


def test_config_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "valid.yaml"
    _write_config(path)
    config = load_config(path)

    assert config_hash(config) == config_hash(config)


def test_shipped_qwen_configs_use_project_api_key_name() -> None:
    for path in (
        Path("configs/experiments/baseline_zero_shot.yaml"),
        Path("configs/experiments/evolution.yaml"),
    ):
        assert load_config(path).model.api_key_env == "OPENAI_API_KEY"


def test_shipped_configs_use_local_embedding_path() -> None:
    for path in (
        Path("configs/default.yaml"),
        Path("configs/experiments/baseline_zero_shot.yaml"),
        Path("configs/experiments/evolution.yaml"),
    ):
        assert load_config(path).embedding.model_id == "models/embeddings/bge-m3"


def test_mini_evolution_config_uses_mini_data_and_learning() -> None:
    config = load_config(Path("configs/experiments/evolution_mini.yaml"))
    assert config.retrieval.enabled is True
    assert config.retrieval.k == 5
    assert config.experiment.train_batch_size == 8
    assert config.experiment.checkpoints == [50, 100]
    assert config.experiment.train_paths == [Path("datasets/mini_dataset/vietnamese/train.json")]
    assert config.experiment.dev_paths == [Path("datasets/mini_dataset/vietnamese/dev.json")]
    assert config.experiment.test_paths == [Path("datasets/mini_dataset/vietnamese/test.json")]
    assert config.attribution.enabled is True
    assert config.attribution.max_retries == 2
    assert config.generalization.enabled is True
    assert config.generalization.minimum_support == 2
    assert config.generalization.minimum_batches == 2
    assert config.generalization.merge_similarity == pytest.approx(0.85)
