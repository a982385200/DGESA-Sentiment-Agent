import pytest

from sentiment_agent.evaluation.evaluator import CostTracker, Evaluator
from sentiment_agent.schemas import Usage


def test_evaluator_accumulates_predictions() -> None:
    evaluator = Evaluator()
    evaluator.add("positive", "positive")
    evaluator.add("negative", "positive")

    report = evaluator.report()

    assert report.accuracy == 0.5


def test_cost_tracker_uses_per_million_prices() -> None:
    tracker = CostTracker(input_price_per_million=2.0, output_price_per_million=8.0)

    tracker.record(Usage(prompt_tokens=500_000, completion_tokens=250_000), latency_seconds=1.5)

    assert tracker.estimated_cost == pytest.approx(3.0)
    assert tracker.total_tokens == 750_000
    assert tracker.average_latency_seconds == 1.5
