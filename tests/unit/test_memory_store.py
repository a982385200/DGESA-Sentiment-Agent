from pathlib import Path

import pytest

from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.schemas import Experience


@pytest.fixture
def experience() -> Experience:
    return Experience(
        text="dịch vụ rất tốt",
        language="vi",
        source="reviews",
        semantic_meaning="service is very good",
        sentiment="positive",
        reason="explicit praise",
        experience_type="successful_case",
    )


@pytest.fixture
def store(tmp_path: Path) -> ExperienceStore:
    return ExperienceStore(tmp_path / "memory.sqlite3")


def test_duplicate_experience_updates_success_count(store: ExperienceStore, experience: Experience) -> None:
    first = store.add_or_update(experience, [1.0, 0.0])
    second = store.add_or_update(experience, [1.0, 0.0])

    assert first == second
    assert store.get(first).success_count == 2


def test_record_failure_reduces_reliability(store: ExperienceStore, experience: Experience) -> None:
    experience_id = store.add_or_update(experience, [1.0, 0.0])
    before = store.get(experience_id).reliability

    updated = store.record_outcome(experience_id, correct=False)

    assert updated.failure_count == 1
    assert updated.reliability < before


def test_store_returns_vector_with_experience(store: ExperienceStore, experience: Experience) -> None:
    experience_id = store.add_or_update(experience, [0.25, 0.75])

    stored, vector = store.get_with_vector(experience_id)

    assert stored.id == experience_id
    assert vector == [0.25, 0.75]


def test_store_rejects_empty_vector(store: ExperienceStore, experience: Experience) -> None:
    with pytest.raises(ValueError, match="vector"):
        store.add_or_update(experience, [])
