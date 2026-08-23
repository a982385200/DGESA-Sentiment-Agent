import pytest
from pydantic import ValidationError

from sentiment_agent.llm.parsing import PredictionPayload, parse_model_json


def test_parser_extracts_fenced_json() -> None:
    parsed = parse_model_json(
        '```json\n{"label":"positive","confidence":0.9,"reason":"clear praise"}\n```',
        PredictionPayload,
    )

    assert parsed.label == "positive"
    assert parsed.confidence == 0.9


def test_parser_extracts_json_surrounded_by_text() -> None:
    parsed = parse_model_json(
        'Result follows: {"label":"neutral","confidence":0.5,"reason":"mixed"} End.',
        PredictionPayload,
    )

    assert parsed.label == "neutral"


def test_parser_rejects_invalid_sentiment_label() -> None:
    with pytest.raises(ValidationError):
        parse_model_json(
            '{"label":"mixed","confidence":0.5,"reason":"unsupported"}',
            PredictionPayload,
        )


def test_parser_rejects_missing_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_model_json("positive sentiment", PredictionPayload)
