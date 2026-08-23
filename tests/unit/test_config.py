from pathlib import Path

import pytest

from sentiment_agent.config import AppConfig


def test_config_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 42\n"
        "model:\n"
        "  name: test-model\n"
        "  base_url: https://example.invalid/v1\n",
        encoding="utf-8",
    )

    assert AppConfig.load(path).config_hash == AppConfig.load(path).config_hash


def test_config_hash_changes_with_model(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("model:\n  name: model-a\n", encoding="utf-8")
    second.write_text("model:\n  name: model-b\n", encoding="utf-8")

    assert AppConfig.load(first).config_hash != AppConfig.load(second).config_hash


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("unknown_option: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_option"):
        AppConfig.load(path)
