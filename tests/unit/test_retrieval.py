from pathlib import Path

import pytest

from sentiment_agent.config import MemoryConfig
from sentiment_agent.memory.retrieval import ExperienceRetriever
from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.schemas import Experience


def make_experience(text: str, language: str, source: str, reliability: float = 0.8) -> Experience:
    return Experience(
        text=text,
        language=language,
        source=source,
        semantic_meaning=text,
        sentiment="positive",
        reason="fixture",
        experience_type="successful_case",
        reliability=reliability,
    )


@pytest.fixture
def store(tmp_path: Path) -> ExperienceStore:
    result = ExperienceStore(tmp_path / "memory.sqlite3")
    result.add_or_update(make_experience("same vi", "vi", "reviews"), [1.0, 0.0])
    result.add_or_update(make_experience("same th", "th", "reviews"), [0.9, 0.1])
    result.add_or_update(make_experience("different vi", "vi", "social"), [0.0, 1.0])
    return result


@pytest.fixture
def retriever(store: ExperienceStore) -> ExperienceRetriever:
    config = MemoryConfig(
        min_reliability=0.0,
        semantic_weight=0.7,
        language_weight=0.1,
        domain_weight=0.1,
        reliability_weight=0.1,
        recency_weight=0.0,
    )
    return ExperienceRetriever(store, config)


def test_same_language_search_excludes_cross_language(retriever: ExperienceRetriever) -> None:
    results = retriever.search([1.0, 0.0], language="vi", source="reviews", cross_lingual=False, k=10)

    assert results
    assert all(item.experience.language == "vi" for item in results)


def test_cross_lingual_search_includes_other_languages(retriever: ExperienceRetriever) -> None:
    results = retriever.search([1.0, 0.0], language="vi", source="reviews", cross_lingual=True, k=10)

    assert any(item.experience.language == "th" for item in results)


def test_semantically_closest_experience_ranks_first(retriever: ExperienceRetriever) -> None:
    results = retriever.search([1.0, 0.0], language="vi", source="reviews", cross_lingual=False, k=2)

    assert results[0].experience.text == "same vi"
    assert results[0].semantic_similarity == pytest.approx(1.0)


def test_retrieval_respects_k(retriever: ExperienceRetriever) -> None:
    results = retriever.search([1.0, 0.0], language="vi", source="reviews", cross_lingual=True, k=1)

    assert len(results) == 1


def test_retrieval_rejects_vector_dimension_mismatch(retriever: ExperienceRetriever) -> None:
    with pytest.raises(ValueError, match="dimension"):
        retriever.search([1.0], language="vi", source="reviews", cross_lingual=True, k=5)
