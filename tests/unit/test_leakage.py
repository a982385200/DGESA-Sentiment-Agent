from pathlib import Path

from sentiment_agent.data.stream import EvaluationStream
from sentiment_agent.schemas import SentimentExample


def test_evaluation_items_do_not_expose_gold_label(tmp_path: Path) -> None:
    example = SentimentExample(
        id="km-test-1",
        text="ល្អ",
        language="km",
        source="fixture",
        split="test",
        label="positive",
    )

    item = next(iter(EvaluationStream([example])))

    assert item.model_dump() == {
        "id": "km-test-1",
        "text": "ល្អ",
        "language": "km",
        "source": "fixture",
    }
