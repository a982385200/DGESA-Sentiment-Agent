import numpy as np

from sentiment_agent.dgesa.models import PatternExperience
from sentiment_agent.dgesa.policies import pattern_scope, pattern_status, weighted_coverage


def pattern(**updates) -> PatternExperience:
    values = dict(
        id="p1", text="contrast means negative", sentiment="negative",
        source_language="vi", evidence_texts=("bad despite praise",),
        support_count=3, contradiction_count=1,
        support_by_language={"vi": 3}, contradiction_by_language={"vi": 1},
        created_batch=1, last_updated_batch=2,
    )
    values.update(updates)
    return PatternExperience(**values)


def test_pattern_experience_uses_beta_prior_and_conflict_ratio() -> None:
    value = pattern()
    assert value.reliability == 4 / 6
    assert value.conflict_ratio == 1 / 4


def test_weighted_coverage_uses_softmax_candidate_weights() -> None:
    result = weighted_coverage(np.array([1.0, 0.0]), [
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ], temperature=1.0)
    assert result == pytest.approx(0.7310585786)


def test_pattern_status_matches_paper_three_state_policy() -> None:
    assert pattern_status(pattern(support_count=2, contradiction_count=0), .6, .2) == "active"
    assert pattern_status(pattern(support_count=2, contradiction_count=1), .6, .2) == "suppressed"
    assert pattern_status(pattern(support_count=0, contradiction_count=0), .6, .2) == "candidate"


def test_pattern_scope_promotes_from_local_to_language_to_global() -> None:
    assert pattern_scope(pattern(support_by_language={"vi": 4}), 5, 3) == "local"
    assert pattern_scope(pattern(support_by_language={"vi": 5}), 5, 3) == "language"
    assert pattern_scope(pattern(support_by_language={"vi": 1, "th": 1, "id": 1}), 5, 3) == "global"


import pytest
