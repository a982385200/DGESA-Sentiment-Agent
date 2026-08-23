import pytest

from sentiment_agent.evaluation.metrics import ALL_LABELS, compute_metrics


def test_macro_f1_uses_all_three_labels() -> None:
    report = compute_metrics(["positive", "negative"], ["positive", "positive"])

    assert report.labels == list(ALL_LABELS)
    assert report.accuracy == 0.5
    assert report.macro_f1 == pytest.approx((2 / 3) / 3)


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(["positive"], [])


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_metrics([], [])
