from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sentiment_agent.dgesa.models import PatternExperience, PatternScope, PatternStatus


def weighted_coverage(query: np.ndarray, candidates: Sequence[np.ndarray], *,
                      temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not candidates:
        return 0.0
    query_vector = _normalized(query)
    similarities = np.asarray([
        float(_normalized(candidate) @ query_vector) for candidate in candidates
    ])
    scaled = similarities / temperature
    weights = np.exp(scaled - scaled.max())
    weights /= weights.sum()
    return float(weights @ similarities)


def pattern_status(pattern: PatternExperience, minimum_reliability: float,
                   maximum_conflict_ratio: float) -> PatternStatus:
    if pattern.conflict_ratio > maximum_conflict_ratio:
        return "suppressed"
    if pattern.reliability >= minimum_reliability:
        return "active"
    return "candidate"


def pattern_scope(pattern: PatternExperience, minimum_language_support: int,
                  minimum_global_languages: int) -> PatternScope:
    supported_languages = sum(value > 0 for value in pattern.support_by_language.values())
    if supported_languages >= minimum_global_languages:
        return "global"
    if max(pattern.support_by_language.values(), default=0) >= minimum_language_support:
        return "language"
    return "local"


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("vectors must be non-zero")
    return value / norm
