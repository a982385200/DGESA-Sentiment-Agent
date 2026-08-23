import json
from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.retrieval import ExperienceRetriever, RetrievalWeights
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.llm.base import LLMResult, PredictionPayload
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import Feedback, PredictionInput


class FakeEmbedding:
    def __init__(self): self.calls = 0
    def embed(self, texts):
        self.calls += 1
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeLLM:
    async def classify(self, messages):
        item = json.loads(messages[-1].content)
        return LLMResult(payload=PredictionPayload(label="positive", confidence=.8, reason=item["text"]))


class FakeEvolutionService:
    def __init__(self):
        self.calls = []
        self.repository = type("Repository", (), {"stats": lambda self: {"active_count": 3}})()

    async def learn_batch(self, items, predictions, feedback, retrieved_contexts, *, batch_id):
        self.calls.append((items, predictions, feedback, retrieved_contexts, batch_id))
        return ["generalized"]


@pytest.mark.anyio
async def test_batch_embeds_once_and_preserves_order(tmp_path: Path) -> None:
    embedding = FakeEmbedding()
    with ExperienceRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        agent = SentimentAgent(embedding=embedding, llm=FakeLLM(),
            retriever=ExperienceRetriever(repo, RetrievalWeights()),
            updater=ExperienceUpdater(repo, index), vector_index=index,
            prompt_builder=PredictionPromptBuilder(), model_name="fake", retrieval_k=3)
        items = [PredictionInput(id=str(i), text=str(i), language="vi", source="tiny") for i in range(2)]
        batch = await agent.predict_batch(items, max_concurrency=2)
        assert embedding.calls == 1
        assert [prediction.sample_id for prediction in batch] == ["0", "1"]


@pytest.mark.anyio
async def test_experience_is_visible_only_to_next_batch(tmp_path: Path) -> None:
    with ExperienceRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        agent = SentimentAgent(embedding=FakeEmbedding(), llm=FakeLLM(),
            retriever=ExperienceRetriever(repo, RetrievalWeights()), updater=ExperienceUpdater(repo, index),
            vector_index=index, prompt_builder=PredictionPromptBuilder(), model_name="fake", retrieval_k=3)
        first_items = [PredictionInput(id=str(i), text=str(i), language="vi", source="tiny") for i in range(2)]
        first = await agent.predict_batch(first_items, max_concurrency=2)
        assert all(not prediction.retrieved_experience_ids for prediction in first)
        feedback = [Feedback(sample_id=p.sample_id, predicted_label=p.label,
                             gold_label="positive", correct=True) for p in first]
        agent.learn_batch(first_items, first, feedback, batch_id=1)
        second = await agent.predict_batch(
            [PredictionInput(id="2", text="2", language="vi", source="tiny")], max_concurrency=1)
        assert second[0].retrieved_experience_ids


@pytest.mark.anyio
async def test_evolve_batch_delegates_prediction_context_and_reports_active_count(tmp_path: Path) -> None:
    evolution = FakeEvolutionService()
    with ExperienceRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        agent = SentimentAgent(embedding=FakeEmbedding(), llm=FakeLLM(),
            retriever=ExperienceRetriever(repo, RetrievalWeights()),
            updater=ExperienceUpdater(repo, index), vector_index=index,
            prompt_builder=PredictionPromptBuilder(), model_name="fake", retrieval_k=3,
            evolution_service=evolution)
        items = [PredictionInput(id="x", text="text", language="vi", source="tiny")]
        predictions = await agent.predict_batch(items, max_concurrency=1)
        feedback = [Feedback(sample_id="x", predicted_label="positive",
                             gold_label="negative", correct=False)]

        learned = await agent.evolve_batch(items, predictions, feedback, batch_id=2)

        assert learned == ["generalized"]
        assert evolution.calls[0][3][0] == ()
        assert agent.experience_count() == 3
        with pytest.raises(ValueError, match="matching prior predictions"):
            await agent.evolve_batch(items, predictions, feedback, batch_id=2)
