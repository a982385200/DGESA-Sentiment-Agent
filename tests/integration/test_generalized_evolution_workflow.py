import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.attribution.llm_attributor import AttributionResult
from sentiment_agent.attribution.models import Attribution
from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.retrieval import RetrievalWeights
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import ExperimentRunner
from sentiment_agent.generalization.lifecycle import LifecyclePolicy
from sentiment_agent.generalization.matcher import RuleMatcher
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.generalization.retrieval import GeneralizedExperienceRetriever
from sentiment_agent.generalization.service import ExperienceEvolutionService
from sentiment_agent.llm.base import LLMResult, PredictionPayload
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import SentimentExample


class FakeEmbedding:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class AlwaysNeutralLLM:
    async def classify(self, messages):
        return LLMResult(payload=PredictionPayload(
            label="neutral", confidence=.8, reason="offline fake"))


class StaticAttributor:
    async def attribute(self, case, retrieved):
        attribution = Attribution(
            id=hashlib.sha256(case.id.encode()).hexdigest()[:24], case_id=case.id,
            error_type="missing_knowledge", root_cause="missing payment rule",
            corrected_reason="A missing expected payment is negative.",
            candidate_rule="Missing an expected payment expresses negative sentiment.",
            scope_languages=(case.language,), scope_sources=(case.source,),
            phenomena=("uncompleted_event",), confidence=.9, created_batch=case.batch_id)
        return AttributionResult(attribution=attribution, used_fallback=False)


def example(id_: str) -> SentimentExample:
    return SentimentExample(id=id_, text="payment not received", label="negative",
                            language="vi", source="tiny")


@pytest.mark.anyio
async def test_generalized_experience_activates_and_exports_research_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = run_dir / "experience_store"
    legacy = ExperienceRepository(store / "experiences.sqlite3")
    evolution = EvolutionRepository(store / "experiences.sqlite3")
    try:
        index = VectorIndex(store / "generalized_index")
        embedding = FakeEmbedding()
        service = ExperienceEvolutionService(
            repository=evolution, embedding=embedding, attributor=StaticAttributor(),
            matcher=RuleMatcher(evolution, index, merge_similarity=.85), vector_index=index,
            lifecycle=LifecyclePolicy(minimum_support=2, minimum_batches=2))
        agent = SentimentAgent(
            embedding=embedding, llm=AlwaysNeutralLLM(),
            retriever=GeneralizedExperienceRetriever(evolution, index, RetrievalWeights()),
            updater=ExperienceUpdater(legacy, index), vector_index=index,
            prompt_builder=PredictionPromptBuilder(), model_name="fake", retrieval_k=2,
            evolution_service=service)
        runner = ExperimentRunner(agent=agent, writer=ArtifactWriter(run_dir),
                                  batch_size=1, concurrency=1)
        await runner.run([example("one"), example("two"), example("three")], [], [example("test")])

        rule = evolution.list_rules()[0]
        assert rule.status == "active"
        rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
        third = next(row for row in rows if row["sample_id"] == "three")
        assert third["retrieved_experience_ids"] == [rule.id]
        assert (run_dir / "generalized_experiences.jsonl").exists()
        assert (run_dir / "attributions.jsonl").exists()
        metrics = json.loads((run_dir / "experience_evolution_metrics.json").read_text())
        assert metrics["case_count"] == 3
        assert metrics["active_count"] == 1
        assert metrics["compression_ratio"] > 0
    finally:
        evolution.close()
        legacy.close()
