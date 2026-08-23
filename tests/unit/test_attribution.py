import json

import pytest

from sentiment_agent.attribution.deterministic import infer_error_type
from sentiment_agent.attribution.llm_attributor import LLMAttributor
from sentiment_agent.evidence.models import CaseEvidence


def make_case() -> CaseEvidence:
    return CaseEvidence(
        id="case-1", sample_id="vi-1", text="không nhận tiền", language="vi",
        source="tiny", predicted_label="neutral", gold_label="negative",
        prediction_reason="status", confidence=0.7, batch_id=1, correct=False,
    )


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return next(self.responses)


def test_no_retrieved_experience_is_missing_knowledge() -> None:
    assert infer_error_type(make_case(), []) == "missing_knowledge"


@pytest.mark.anyio
async def test_attributor_retries_invalid_json_then_parses() -> None:
    payload = {
        "error_type": "missing_knowledge", "root_cause": "No payment rule",
        "corrected_reason": "Unmet payment expectation is negative",
        "candidate_rule": "Payment not received normally expresses negative sentiment",
        "scope_languages": ["vi"], "scope_sources": ["tiny"],
        "phenomena": ["uncompleted_event"], "confidence": 0.9,
    }
    client = SequenceClient(["not json", json.dumps(payload)])
    result = await LLMAttributor(client, max_retries=1).attribute(make_case(), [])
    assert result.used_fallback is False
    assert result.attribution.candidate_rule == payload["candidate_rule"]
    assert client.calls == 2


@pytest.mark.anyio
async def test_attributor_falls_back_after_invalid_json() -> None:
    client = SequenceClient(["bad", "still bad"])
    result = await LLMAttributor(client, max_retries=1).attribute(make_case(), [])
    assert result.used_fallback is True
    assert result.attribution.error_type == "missing_knowledge"
    assert result.attribution.candidate_rule
    assert result.raw_responses == ("bad", "still bad")
