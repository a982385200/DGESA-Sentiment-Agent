import pytest
from pydantic import ValidationError

from sentiment_agent.schemas import Feedback, PredictionInput, SentimentExample


def test_prediction_input_rejects_label() -> None:
    with pytest.raises(ValidationError):
        PredictionInput(
            id="th-1",
            text="ดี",
            language="th",
            source="tiny",
            label="positive",
        )


def test_sentiment_example_converts_to_unlabelled_input() -> None:
    example = SentimentExample(
        id="vi-1",
        text="dịch vụ tốt",
        label="positive",
        language="vi",
        source="tiny",
    )

    item = example.to_prediction_input()

    assert item.model_dump() == {
        "id": "vi-1",
        "text": "dịch vụ tốt",
        "language": "vi",
        "source": "tiny",
    }


def test_feedback_validates_correct_flag() -> None:
    with pytest.raises(ValidationError, match="correct"):
        Feedback(
            sample_id="x",
            predicted_label="positive",
            gold_label="negative",
            correct=True,
        )
