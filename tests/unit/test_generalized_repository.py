from pathlib import Path

import pytest

from sentiment_agent.attribution.models import Attribution
from sentiment_agent.evidence.models import CaseEvidence
from sentiment_agent.generalization.models import GeneralizedExperience
from sentiment_agent.generalization.repository import EvolutionRepository


def make_case() -> CaseEvidence:
    return CaseEvidence(
        id="case-1", sample_id="vi-1", text="không nhận tiền", language="vi",
        source="tiny", predicted_label="neutral", gold_label="negative",
        prediction_reason="status", confidence=0.7, batch_id=1, correct=False,
    )


def make_rule() -> GeneralizedExperience:
    return GeneralizedExperience(
        id="rule-1", semantic="payment not received", sentiment="negative",
        rule="Missing an expected payment is negative.", corrected_reason="Unmet expectation.",
        created_batch=1, last_updated_batch=1,
    )


def test_repository_persists_case_attribution_and_rule(tmp_path: Path) -> None:
    path = tmp_path / "evolution.sqlite3"
    case = make_case()
    attribution = Attribution(
        id="attr-1", case_id=case.id, error_type="missing_knowledge",
        root_cause="missing rule", corrected_reason="unmet expectation",
        candidate_rule="Missing expected payment is negative.", confidence=0.9,
        created_batch=1,
    )
    with EvolutionRepository(path) as repo:
        repo.create_case(case)
        repo.create_attribution(attribution)
        repo.create_rule(make_rule())
    with EvolutionRepository(path) as repo:
        assert repo.get_case(case.id) == case
        assert repo.get_attribution(attribution.id) == attribution
        assert repo.get_rule("rule-1").sentiment == "negative"


def test_case_is_unique_per_sample_and_batch(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        repo.create_case(make_case())
        duplicate = make_case().model_copy(update={"id": "case-2"})
        with pytest.raises(ValueError, match="case evidence already exists"):
            repo.create_case(duplicate)


def test_evidence_relation_is_idempotent_and_updates_rule(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        repo.create_case(make_case())
        repo.create_rule(make_rule())
        assert repo.add_evidence("rule-1", "case-1", relation="support", batch_id=1)
        assert not repo.add_evidence("rule-1", "case-1", relation="support", batch_id=1)
        rule = repo.get_rule("rule-1")
        assert rule.support_count == 1
        assert rule.supporting_batches == (1,)
        assert repo.stats()["support_count"] == 1
        assert repo.history("rule-1")[-1]["event_type"] == "reinforced"
