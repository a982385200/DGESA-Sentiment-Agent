import pytest
from pydantic import ValidationError

from sentiment_agent.schemas import Feedback, PredictionInput


def test_prediction_input_rejects_gold_label() -> None:
    with pytest.raises(ValidationError, match="label"):
        PredictionInput.model_validate(
            {
                "id": "vi-1",
                "text": "dịch vụ tốt",
                "language": "vi",
                "source": "fixture",
                "label": "positive",
            }
        )


def test_feedback_derives_correctness() -> None:
    feedback = Feedback(
        sample_id="vi-1",
        predicted_label="positive",
        gold_label="positive",
    )

    assert feedback.correct is True


def test_prediction_input_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        PredictionInput(id="vi-1", text="   ", language="vi", source="fixture")
