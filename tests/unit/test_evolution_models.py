from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sentiment_agent.evidence.models import CaseEvidence
from sentiment_agent.attribution.models import Attribution
from sentiment_agent.generalization.models import GeneralizedExperience


def test_case_evidence_is_immutable() -> None:
    case = CaseEvidence(
        id="case-1", sample_id="vi-1", text="không nhận được tiền",
        language="vi", source="tiny", predicted_label="neutral",
        gold_label="negative", prediction_reason="status statement",
        confidence=0.7, batch_id=1, correct=False,
    )
    with pytest.raises(ValidationError):
        case.batch_id = 2


def test_attribution_rejects_unknown_error_type() -> None:
    with pytest.raises(ValidationError):
        Attribution(
            id="attr-1", case_id="case-1", error_type="unknown",
            root_cause="cause", corrected_reason="reason", candidate_rule="rule",
            confidence=0.8, created_batch=1,
        )


def test_generalized_experience_computes_quality() -> None:
    rule = GeneralizedExperience(
        id="rule-1", semantic="payment not received", sentiment="negative",
        rule="Expected payment not received is negative.", corrected_reason="Unmet expectation.",
        support_count=3, contradiction_count=1, supporting_batches=(1, 2),
        created_batch=1, last_updated_batch=2,
    )
    assert rule.reliability == pytest.approx(4 / 6)
    assert rule.contradiction_ratio == pytest.approx(0.25)


def test_generalized_experience_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        GeneralizedExperience(
            id="rule-1", status="deleted", semantic="x", sentiment="negative",
            rule="r", corrected_reason="c", created_batch=1, last_updated_batch=1,
        )
