import json
from pathlib import Path

import pytest

from sentiment_agent.evaluation.metrics import classification_metrics
from sentiment_agent.experiments.artifacts import ArtifactWriter


def test_macro_f1_includes_all_three_labels() -> None:
    result = classification_metrics(["positive"], ["positive"])
    assert set(result["per_class"]) == {"positive", "neutral", "negative"}
    assert result["macro_f1"] == 1 / 3


def test_classification_metrics_include_micro_and_weighted_f1() -> None:
    result = classification_metrics(
        ["positive", "positive", "neutral", "negative"],
        ["positive", "negative", "neutral", "negative"],
    )
    assert result["micro_f1"] == 0.75
    assert result["weighted_f1"] == pytest.approx(0.75)


def test_artifact_writer_writes_json_and_jsonl(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    writer.write_json("metrics.json", {"accuracy": 1.0})
    writer.append_jsonl("predictions.jsonl", {"id": "x"})
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"accuracy": 1.0}
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == {"id": "x"}
