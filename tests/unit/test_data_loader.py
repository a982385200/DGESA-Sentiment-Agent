import json
from collections import Counter
from pathlib import Path

import pytest

from sentiment_agent.data.loader import load_examples, without_labels

ROW = {"id": "vi-1", "text": "tốt", "label": "positive", "language": "vi", "source": "tiny"}


def test_load_examples_supports_json_array(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([ROW]), encoding="utf-8")
    assert load_examples(path)[0].label == "positive"


def test_load_examples_supports_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    assert load_examples(path)[0].id == "vi-1"


def test_load_examples_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([ROW, ROW]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample id"):
        load_examples(path)


def test_without_labels_removes_gold_label(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([ROW]), encoding="utf-8")
    assert "label" not in without_labels(load_examples(path))[0].model_dump()


def test_medium_dataset_has_stratified_15000_train_and_2500_test() -> None:
    root = Path("datasets/medium_dataset")
    languages = ("indonesian", "khmer", "malay", "thai", "vietnamese")
    for language in languages:
        train = load_examples(root / language / "train.json")
        test = load_examples(root / language / "test.json")
        assert len(train) == 3000
        assert len(test) == 500
        assert set(Counter(item.label for item in train)) == {"positive", "neutral", "negative"}
        assert set(Counter(item.label for item in test)) == {"positive", "neutral", "negative"}
        assert {item.id for item in train}.isdisjoint(item.id for item in test)
