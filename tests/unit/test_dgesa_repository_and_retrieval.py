from pathlib import Path

import numpy as np

from sentiment_agent.dgesa.models import PatternExperience, SampleExperience
from sentiment_agent.dgesa.repository import DGESARepository
from sentiment_agent.dgesa.retrieval import PatternRetriever, SampleRetriever


def sample(id_: str, text: str = "local correction") -> SampleExperience:
    return SampleExperience(
        id=id_, text="input", experience=text, sentiment="negative", language="vi",
        source="reviews", source_sample_id=id_, created_batch=1,
    )


def pattern(id_: str, *, scope="language", language="vi", status="active",
            evidence=("evidence",), support=5, contradiction=0) -> PatternExperience:
    return PatternExperience(
        id=id_, text=f"rule {id_}", sentiment="negative", source_language=language,
        status=status, scope=scope, evidence_texts=evidence,
        support_count=support, contradiction_count=contradiction,
        support_by_language={language: support},
        contradiction_by_language={language: contradiction},
        created_batch=1, last_updated_batch=1,
    )


def test_repository_round_trips_and_updates_both_granularities(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "dgesa.sqlite3") as repo:
        repo.save_sample(sample("s1"), np.array([1.0, 0.0]))
        repo.save_pattern(pattern("p1"), np.array([0.0, 1.0]))
        assert repo.get_sample("s1").experience == "local correction"
        assert repo.get_pattern("p1").text == "rule p1"
        assert [item.id for item in repo.list_samples()] == ["s1"]
        updated = pattern("p1", support=6)
        repo.save_pattern(updated, np.array([1.0, 1.0]))
        assert repo.get_pattern("p1").support_count == 6


def test_sample_retrieval_applies_similarity_threshold_and_top_k(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "dgesa.sqlite3") as repo:
        repo.save_sample(sample("near"), np.array([1.0, 0.0]))
        repo.save_sample(sample("far"), np.array([0.0, 1.0]))
        found = SampleRetriever(repo, minimum_similarity=.8).search(
            np.array([1.0, 0.0]), k=3)
        assert [item.experience.id for item in found] == ["near"]


def test_pattern_retrieval_enforces_scope_and_paper_score(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "dgesa.sqlite3") as repo:
        repo.save_pattern(pattern("same-language"), np.array([1.0, 0.0]))
        repo.save_pattern(pattern("foreign", language="th"), np.array([1.0, 0.0]))
        repo.save_pattern(pattern("global", scope="global", language="th"), np.array([.9, .1]))
        repo.save_pattern(pattern("local", scope="local", evidence=("exact",)), np.array([1.0, 0.0]))
        repo.save_pattern(pattern("suppressed", status="suppressed"), np.array([1.0, 0.0]))
        found = PatternRetriever(repo, semantic_weight=.6, reliability_weight=.3,
                                 conflict_weight=.1, local_similarity=.95).search(
            np.array([1.0, 0.0]), language="vi", evidence_vector=np.array([1.0, 0.0]), k=5)
        assert [item.experience.id for item in found] == ["local", "same-language", "global"]
        assert found[0].score == pytest.approx(.6 + .3 * (6 / 7))


def test_local_pattern_gate_compares_query_with_evidence_not_rule_vector(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "dgesa.sqlite3") as repo:
        repo.save_pattern(
            pattern("local", scope="local"), np.array([1.0, 0.0]),
            evidence_vectors=[np.array([0.0, 1.0])],
        )
        found = PatternRetriever(repo, local_similarity=.95).search(
            np.array([1.0, 0.0]), language="vi", k=3)
        assert found == []


def test_language_scope_uses_languages_with_sufficient_support(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "dgesa.sqlite3") as repo:
        value = pattern("multi", scope="language").model_copy(update={
            "support_count": 6,
            "support_by_language": {"vi": 1, "th": 5},
        })
        repo.save_pattern(value, np.array([1.0, 0.0]))
        found = PatternRetriever(repo, minimum_language_support=5).search(
            np.array([1.0, 0.0]), language="th", k=3)
        assert [item.experience.id for item in found] == ["multi"]


import pytest
