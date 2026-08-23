from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sentiment_agent.schemas import SentimentLabel, StrictModel

ALL_LABELS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")


class ClassMetrics(StrictModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)


class MetricsReport(StrictModel):
    labels: list[SentimentLabel]
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_class: dict[SentimentLabel, ClassMetrics]
    sample_count: int = Field(gt=0)


def compute_metrics(
    gold: Sequence[SentimentLabel],
    predicted: Sequence[SentimentLabel],
) -> MetricsReport:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must have the same length")
    if not gold:
        raise ValueError("metric inputs must not be empty")

    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=list(ALL_LABELS),
        zero_division=0,
    )
    per_class = {
        label: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(ALL_LABELS)
    }
    return MetricsReport(
        labels=list(ALL_LABELS),
        accuracy=float(accuracy_score(gold, predicted)),
        macro_f1=float(sum(item.f1 for item in per_class.values()) / len(ALL_LABELS)),
        per_class=per_class,
        sample_count=len(gold),
    )
