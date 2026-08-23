import json
from pathlib import Path

from sentiment_agent.evaluation.artifacts import ArtifactWriter


def test_artifact_writer_emits_valid_jsonl(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)

    writer.append_prediction({"sample_id": "vi-1", "label": "positive"})

    row = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["sample_id"] == "vi-1"


def test_artifact_writer_replaces_json_atomically(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)

    writer.write_json("metrics.json", {"accuracy": 0.5})
    writer.write_json("metrics.json", {"accuracy": 1.0})

    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == {"accuracy": 1.0}
    assert list(tmp_path.glob("*.tmp")) == []
