import json
from pathlib import Path

from sentiment_agent.evaluation.metrics import classification_metrics
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import partition_batches


def test_macro_f1_includes_all_three_labels() -> None:
    result = classification_metrics(["positive"], ["positive"])
    assert set(result["per_class"]) == {"positive", "neutral", "negative"}
    assert result["macro_f1"] == 1 / 3


def test_partition_batches_respects_hard_checkpoints() -> None:
    assert [len(batch) for batch in partition_batches(list(range(5)), 4, [3, 5])] == [3, 2]


def test_artifact_writer_writes_json_and_jsonl(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    writer.write_json("metrics.json", {"accuracy": 1.0})
    writer.append_jsonl("predictions.jsonl", {"id": "x"})
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"accuracy": 1.0}
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == {"id": "x"}
