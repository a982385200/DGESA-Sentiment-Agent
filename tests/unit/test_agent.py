from pathlib import Path
from types import SimpleNamespace

import pytest

from sentiment_agent.agent.agent import SentimentAgent
from sentiment_agent.config import MemoryConfig
from sentiment_agent.llm.parsing import PredictionPayload
from sentiment_agent.memory.retrieval import ExperienceRetriever
from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.reflection.reflector import ReflectionPayload
from sentiment_agent.schemas import Feedback, PredictionInput, Usage
from sentiment_agent.strategies.prompts import PromptBuilder
from sentiment_agent.strategies.selector import EpsilonGreedySelector


class FakeClient:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def chat_json(self, messages, response_model):
        if response_model is PredictionPayload:
            return SimpleNamespace(
                payload=PredictionPayload(label="positive", confidence=0.8, reason="fixture"),
                usage=Usage(prompt_tokens=5, completion_tokens=3),
                cache_key="cache-1",
            )
        if response_model is ReflectionPayload:
            return SimpleNamespace(
                payload=ReflectionPayload(
                    error_type="none",
                    corrected_reason="The positive meaning is explicit.",
                    generalized_rule="Explicit praise is usually positive.",
                    scope="reviews",
                )
            )
        raise AssertionError(response_model)


@pytest.fixture
def store(tmp_path: Path) -> ExperienceStore:
    return ExperienceStore(tmp_path / "memory.sqlite3")


@pytest.fixture
def agent(store: ExperienceStore) -> SentimentAgent:
    return SentimentAgent(
        client=FakeClient(),
        retriever=ExperienceRetriever(store, MemoryConfig(min_reliability=0.0)),
        store=store,
        prompt_builder=PromptBuilder(),
        selector=EpsilonGreedySelector(["direct", "memory"], epsilon=0.0),
        model_name="fixture-model",
        reflection_enabled=True,
        cross_lingual=True,
        retrieval_k=3,
        seed=42,
    )


@pytest.fixture
def item() -> PredictionInput:
    return PredictionInput(id="vi-1", text="dịch vụ tốt", language="vi", source="reviews")


def test_predict_does_not_persist_experience(agent: SentimentAgent, store: ExperienceStore, item: PredictionInput) -> None:
    prediction = agent.predict(item)

    assert prediction.label == "positive"
    assert prediction.sample_id == item.id
    assert store.count() == 0


def test_learn_persists_case_and_rule(agent: SentimentAgent, store: ExperienceStore, item: PredictionInput) -> None:
    prediction = agent.predict(item)
    feedback = Feedback(sample_id=item.id, predicted_label=prediction.label, gold_label="positive")

    result = agent.learn(item, prediction, feedback)

    assert len(result.experience_ids) == 2
    assert store.count() == 2
    assert agent.selector.stats("vi")[prediction.strategy].mean_reward == 1.0


def test_learn_rejects_mismatched_feedback(agent: SentimentAgent, item: PredictionInput) -> None:
    prediction = agent.predict(item)
    feedback = Feedback(sample_id="wrong", predicted_label=prediction.label, gold_label="negative")

    with pytest.raises(ValueError, match="sample id"):
        agent.learn(item, prediction, feedback)


def test_learn_requires_prior_prediction(agent: SentimentAgent, item: PredictionInput) -> None:
    feedback = Feedback(sample_id=item.id, predicted_label="positive", gold_label="positive")
    prediction = SimpleNamespace(sample_id=item.id, label="positive", strategy="direct")

    with pytest.raises(ValueError, match="prior prediction"):
        agent.learn(item, prediction, feedback)
