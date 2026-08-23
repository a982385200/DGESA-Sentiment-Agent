import json
from pathlib import Path

import pytest

from sentiment_agent.data.loader import load_jsonl, prediction_input
from sentiment_agent.data.stream import EvaluationStream, TrainingStream
from sentiment_agent.schemas import SentimentExample


@pytest.fixture
def example() -> SentimentExample:
    return SentimentExample(
        id="th-1",
        text="บริการดี",
        language="th",
        source="fixture",
        split="train",
        label="positive",
    )


def test_load_jsonl_assigns_requested_split(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "id-1",
                "text": "bagus",
                "language": "id",
                "source": "fixture",
                "label": "positive",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl(path, split="train")

    assert rows[0].split == "train"
    assert rows[0].label == "positive"


def test_prediction_input_removes_gold_label(example: SentimentExample) -> None:
    item = prediction_input(example)

    assert "label" not in item.model_dump()
    assert item.id == example.id


def test_evaluation_stream_has_no_feedback_method(example: SentimentExample) -> None:
    stream = EvaluationStream([example])

    assert next(iter(stream)).id == example.id
    assert not hasattr(stream, "feedback")


def test_training_feedback_rejects_mismatched_sample(example: SentimentExample) -> None:
    stream = TrainingStream([example])
    item = next(iter(stream))

    with pytest.raises(ValueError, match="sample id"):
        stream.feedback(item, predicted="positive", sample_id="wrong")


def test_training_feedback_uses_hidden_gold_label(example: SentimentExample) -> None:
    stream = TrainingStream([example])
    item = next(iter(stream))

    feedback = stream.feedback(item, predicted="negative", sample_id=example.id)

    assert feedback.gold_label == "positive"
    assert feedback.correct is False
