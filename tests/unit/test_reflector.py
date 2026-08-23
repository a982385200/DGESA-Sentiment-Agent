from types import SimpleNamespace

from sentiment_agent.reflection.reflector import ReflectionPayload, Reflector
from sentiment_agent.schemas import Feedback, Prediction, PredictionInput


class FakeReflectionClient:
    def __init__(self) -> None:
        self.messages = None

    def chat_json(self, messages, response_model):
        self.messages = messages
        assert response_model is ReflectionPayload
        return SimpleNamespace(
            payload=ReflectionPayload(
                error_type="sarcasm",
                corrected_reason="The phrase is sarcastic.",
                generalized_rule="Check pragmatic cues before trusting positive words.",
                scope="social media",
            )
        )


def make_prediction() -> Prediction:
    return Prediction(
        sample_id="id-1",
        label="positive",
        confidence=0.8,
        reason="positive word",
        strategy="direct",
        model_name="fixture",
    )


def test_disabled_reflector_does_not_call_client() -> None:
    client = FakeReflectionClient()
    reflector = Reflector(client, enabled=False)

    result = reflector.reflect(
        PredictionInput(id="id-1", text="great...", language="id", source="social"),
        make_prediction(),
        Feedback(sample_id="id-1", predicted_label="positive", gold_label="negative"),
        [],
    )

    assert result is None
    assert client.messages is None


def test_reflector_returns_structured_rule() -> None:
    client = FakeReflectionClient()
    reflector = Reflector(client, enabled=True)

    result = reflector.reflect(
        PredictionInput(id="id-1", text="great...", language="id", source="social"),
        make_prediction(),
        Feedback(sample_id="id-1", predicted_label="positive", gold_label="negative"),
        [],
    )

    assert result is not None
    assert result.error_type == "sarcasm"
    assert client.messages is not None
