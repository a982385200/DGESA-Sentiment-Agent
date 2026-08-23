import json

import pytest

from sentiment_agent.memory.retrieval import RetrievedExperience
from sentiment_agent.schemas import Experience, PredictionInput
from sentiment_agent.strategies.prompts import PromptBuilder


@pytest.fixture
def item() -> PredictionInput:
    return PredictionInput(id="th-1", text="บริการดี", language="th", source="reviews")


@pytest.fixture
def retrieved() -> list[RetrievedExperience]:
    experience = Experience(
        id="exp-1",
        text="dịch vụ tốt",
        language="vi",
        source="reviews",
        semantic_meaning="good service",
        sentiment="positive",
        reason="explicit praise",
        experience_type="successful_case",
        reliability=0.8,
    )
    return [RetrievedExperience(experience=experience, score=0.9, semantic_similarity=0.95)]


def test_direct_prompt_contains_no_experience(item: PredictionInput, retrieved: list[RetrievedExperience]) -> None:
    messages = PromptBuilder().build("direct", item, retrieved)

    rendered = json.dumps(messages, ensure_ascii=False)
    assert item.text in rendered
    assert "explicit praise" not in rendered


def test_memory_prompt_contains_retrieved_reason(item: PredictionInput, retrieved: list[RetrievedExperience]) -> None:
    messages = PromptBuilder().build("memory", item, retrieved)

    rendered = json.dumps(messages, ensure_ascii=False)
    assert "explicit praise" in rendered
    assert "0.800" in rendered


def test_translation_prompt_requests_pivot_translation(item: PredictionInput) -> None:
    rendered = json.dumps(PromptBuilder().build("translation", item, []), ensure_ascii=False)

    assert "English" in rendered


def test_prompt_builder_rejects_unknown_strategy(item: PredictionInput) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        PromptBuilder().build("unsupported", item, [])
