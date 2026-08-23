from __future__ import annotations

from collections.abc import Sequence

from sentiment_agent.attribution.models import ErrorType
from sentiment_agent.evidence.models import CaseEvidence


def _value(item, name: str):
    value = getattr(item, name, None)
    if value is None and hasattr(item, "experience"):
        value = getattr(item.experience, name, None)
    return value


def infer_error_type(case: CaseEvidence, retrieved: Sequence) -> ErrorType:
    if not retrieved:
        return "missing_knowledge"
    if any(_value(item, "language") != case.language and
           _value(item, "sentiment") == case.predicted_label for item in retrieved):
        return "negative_transfer"
    if all(_value(item, "sentiment") != case.gold_label for item in retrieved):
        return "wrong_experience"
    if any(_value(item, "sentiment") == case.gold_label for item in retrieved):
        return "retrieval_failure"
    return "reasoning_error"
