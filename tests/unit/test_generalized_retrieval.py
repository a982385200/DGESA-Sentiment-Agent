from pathlib import Path

import numpy as np

from sentiment_agent.experience.retrieval import RetrievalWeights
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.generalization.models import GeneralizedExperience
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.generalization.retrieval import GeneralizedExperienceRetriever
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import PredictionInput


def rule(id_: str, status: str) -> GeneralizedExperience:
    return GeneralizedExperience(
        id=id_, status=status, semantic="payment not received", sentiment="negative",
        rule="Missing expected payment is negative.", corrected_reason="Unmet expectation.",
        scope_languages=("vi",), scope_sources=("tiny",), support_count=2,
        supporting_batches=(1, 2), created_batch=1, last_updated_batch=2,
    )


def test_retrieval_returns_only_active_rules(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        repo.create_rule(rule("candidate", "candidate"))
        repo.create_rule(rule("active", "active"))
        index.upsert("candidate", np.array([1., 0.], dtype=np.float32))
        index.upsert("active", np.array([1., 0.], dtype=np.float32))
        retriever = GeneralizedExperienceRetriever(repo, index, RetrievalWeights())
        results = retriever.search(np.array([1., 0.], dtype=np.float32), index.snapshot(),
                                   language="vi", source="tiny", k=5)
        assert [item.experience.id for item in results] == ["active"]


def test_prompt_injects_rule_but_not_provenance(tmp_path: Path) -> None:
    with EvolutionRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        repo.create_rule(rule("active", "active"))
        index.upsert("active", np.array([1., 0.], dtype=np.float32))
        result = GeneralizedExperienceRetriever(repo, index, RetrievalWeights()).search(
            np.array([1., 0.], dtype=np.float32), index.snapshot(),
            language="vi", source="tiny", k=1)
        messages = PredictionPromptBuilder().build(
            PredictionInput(id="x", text="text", language="vi", source="tiny"), result)
        content = messages[-1].content
        assert "Missing expected payment is negative" in content
        assert "supporting_batches" not in content
        assert "created_batch" not in content
