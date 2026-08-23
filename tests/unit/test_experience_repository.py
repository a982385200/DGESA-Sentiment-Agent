from pathlib import Path

import pytest

from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.schemas import Experience


@pytest.fixture
def experience() -> Experience:
    return Experience(
        id="exp-1", type="successful_case", language="vi", source="tiny",
        text="tốt", semantic_summary="good", sentiment="positive", reason="positive word",
        source_sample_id="vi-1", created_batch=1, last_used_batch=1,
    )


def test_repository_persists_experience_and_creation_event(tmp_path: Path, experience: Experience) -> None:
    path = tmp_path / "experiences.sqlite3"
    with ExperienceRepository(path) as repo:
        repo.create(experience, batch_id=1)
        assert repo.get(experience.id) == experience
        assert [event.event_type for event in repo.history(experience.id)] == ["created"]
    with ExperienceRepository(path) as repo:
        assert repo.count() == 1


def test_repository_updates_counts_and_reliability(tmp_path: Path, experience: Experience) -> None:
    with ExperienceRepository(tmp_path / "experiences.sqlite3") as repo:
        repo.create(experience, batch_id=1)
        updated = repo.merge_counts("exp-1", success_delta=1, failure_delta=0, batch_id=2)
        assert updated.success_count == 1
        assert updated.reliability == pytest.approx(2 / 3)
        assert repo.history("exp-1")[-1].event_type == "reinforced"


def test_repository_lists_experiences(tmp_path: Path, experience: Experience) -> None:
    with ExperienceRepository(tmp_path / "experiences.sqlite3") as repo:
        repo.create(experience, batch_id=1)
        assert repo.list() == [experience]
