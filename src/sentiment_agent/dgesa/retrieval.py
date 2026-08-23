from __future__ import annotations

import numpy as np

from sentiment_agent.dgesa.models import RetrievedPattern, RetrievedSample
from sentiment_agent.dgesa.repository import DGESARepository


class SampleRetriever:
    def __init__(self, repository: DGESARepository, *, minimum_similarity: float = .8) -> None:
        self.repository = repository
        self.minimum_similarity = minimum_similarity

    def search(self, vector: np.ndarray, *, k: int) -> list[RetrievedSample]:
        query = _normalized(vector)
        rows = []
        for experience, candidate in self.repository.sample_vectors():
            similarity = float(query @ _normalized(candidate))
            if similarity >= self.minimum_similarity:
                rows.append((similarity, experience.id, experience))
        rows.sort(key=lambda value: (-value[0], value[1]))
        return [RetrievedSample(experience=row[2], similarity=row[0], rank=rank)
                for rank, row in enumerate(rows[:k], start=1)]


class PatternRetriever:
    def __init__(self, repository: DGESARepository, *, semantic_weight: float = .6,
                 reliability_weight: float = .3, conflict_weight: float = .1,
                 minimum_reliability: float = .6, maximum_conflict_ratio: float = .2,
                 local_similarity: float = .95,
                 minimum_language_support: int = 5) -> None:
        self.repository = repository
        self.semantic_weight = semantic_weight
        self.reliability_weight = reliability_weight
        self.conflict_weight = conflict_weight
        self.minimum_reliability = minimum_reliability
        self.maximum_conflict_ratio = maximum_conflict_ratio
        self.local_similarity = local_similarity
        self.minimum_language_support = minimum_language_support

    def search(self, vector: np.ndarray, *, language: str,
               evidence_vector: np.ndarray | None = None, k: int) -> list[RetrievedPattern]:
        query = _normalized(vector)
        evidence = query if evidence_vector is None else _normalized(evidence_vector)
        rows = []
        for experience, candidate, evidence_vectors in self.repository.pattern_records():
            candidate_vector = _normalized(candidate)
            similarity = float(query @ candidate_vector)
            if experience.status != "active":
                continue
            if experience.reliability < self.minimum_reliability:
                continue
            if experience.conflict_ratio > self.maximum_conflict_ratio:
                continue
            if (experience.scope == "language" and
                    experience.support_by_language.get(language, 0)
                    < self.minimum_language_support):
                continue
            if experience.scope == "local":
                evidence_similarity = max(
                    (float(evidence @ _normalized(value)) for value in evidence_vectors),
                    default=-1.0,
                )
                if evidence_similarity < self.local_similarity:
                    continue
            score = (self.semantic_weight * similarity
                     + self.reliability_weight * experience.reliability
                     - self.conflict_weight * experience.conflict_ratio)
            rows.append((score, similarity, experience.id, experience))
        rows.sort(key=lambda value: (-value[0], value[2]))
        return [RetrievedPattern(experience=row[3], score=row[0], similarity=row[1], rank=rank)
                for rank, row in enumerate(rows[:k], start=1)]


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("vectors must be non-zero")
    return value / norm
