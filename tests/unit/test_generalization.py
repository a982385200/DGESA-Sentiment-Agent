import hashlib
from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.attribution.llm_attributor import AttributionResult
from sentiment_agent.attribution.models import Attribution
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.generalization.lifecycle import LifecyclePolicy
from sentiment_agent.generalization.matcher import RuleMatcher
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.generalization.service import ExperienceEvolutionService
from sentiment_agent.schemas import Feedback, Prediction, PredictionInput


class FakeEmbedding:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class StaticAttributor:
    def __init__(self, used_fallback=False):
        self.calls = 0
        self.used_fallback = used_fallback
    async def attribute(self, case, retrieved):
        self.calls += 1
        attribution = Attribution(
            id=hashlib.sha256(case.id.encode()).hexdigest()[:24], case_id=case.id,
            error_type="missing_knowledge", root_cause="missing rule",
            corrected_reason="Unmet payment expectation is negative.",
            candidate_rule="Payment not received normally expresses negative sentiment.",
            scope_languages=(case.language,), scope_sources=(case.source,),
            phenomena=("uncompleted_event",), confidence=0.9, created_batch=case.batch_id,
        )
        return AttributionResult(attribution=attribution, used_fallback=self.used_fallback,
                                 raw_responses=("invalid",) if self.used_fallback else ())


def item(sample_id):
    return PredictionInput(id=sample_id, text="không nhận tiền", language="vi", source="tiny")


def wrong(sample_id):
    return Prediction(sample_id=sample_id, label="neutral", confidence=.7,
                      reason="status", model_name="fake")


def feedback(sample_id):
    return Feedback(sample_id=sample_id, predicted_label="neutral",
                    gold_label="negative", correct=False)


@pytest.mark.anyio
async def test_similar_rules_activate_only_after_two_batches(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "rules")
        attributor = StaticAttributor()
        service = ExperienceEvolutionService(
            repository=repo, embedding=FakeEmbedding(), attributor=attributor,
            matcher=RuleMatcher(repo, index, merge_similarity=.85), vector_index=index,
            lifecycle=LifecyclePolicy(minimum_support=2, minimum_batches=2,
                                      maximum_contradiction_ratio=.2,
                                      minimum_active_reliability=.6),
        )
        await service.learn_batch([item("a")], [wrong("a")], [feedback("a")], [[]], batch_id=1)
        assert repo.list_rules()[0].status == "candidate"
        await service.learn_batch([item("b")], [wrong("b")], [feedback("b")], [[]], batch_id=2)
        rules = repo.list_rules()
        assert len(rules) == 1
        assert rules[0].status == "active"
        assert rules[0].support_count == 2
        assert attributor.calls == 2


@pytest.mark.anyio
async def test_correct_sample_does_not_call_attributor(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "rules")
        attributor = StaticAttributor()
        service = ExperienceEvolutionService(
            repository=repo, embedding=FakeEmbedding(), attributor=attributor,
            matcher=RuleMatcher(repo, index, merge_similarity=.85), vector_index=index,
            lifecycle=LifecyclePolicy(),
        )
        prediction = Prediction(sample_id="ok", label="positive", confidence=.9,
                                reason="positive", model_name="fake")
        outcome = Feedback(sample_id="ok", predicted_label="positive",
                           gold_label="positive", correct=True)
        await service.learn_batch([item("ok")], [prediction], [outcome], [[]], batch_id=1)
        assert attributor.calls == 0
        assert repo.stats()["case_count"] == 1


@pytest.mark.anyio
async def test_attribution_fallback_is_retained_for_audit(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "rules")
        service = ExperienceEvolutionService(
            repository=repo, embedding=FakeEmbedding(), attributor=StaticAttributor(True),
            matcher=RuleMatcher(repo, index, merge_similarity=.85), vector_index=index,
            lifecycle=LifecyclePolicy())
        await service.learn_batch([item("fallback")], [wrong("fallback")],
                                  [feedback("fallback")], [[]], batch_id=1)
        assert service.attribution_failures == [{
            "case_id": repo.list_attributions()[0].case_id,
            "raw_responses": ["invalid"],
        }]
