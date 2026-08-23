from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.vector_index import VectorSnapshot
from sentiment_agent.schemas import Language, RetrievedExperience


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 1.0
    language: float = 0.1
    source: float = 0.05
    reliability: float = 0.1


class ExperienceRetriever:
    def __init__(self, repository: ExperienceRepository, weights: RetrievalWeights,
                 *, minimum_reliability: float = 0.0, cross_lingual: bool = True) -> None:
        self.repository = repository
        self.weights = weights
        self.minimum_reliability = minimum_reliability
        self.cross_lingual = cross_lingual

    def search(self, vector: np.ndarray, snapshot: VectorSnapshot, *, language: Language,
               source: str, k: int) -> list[RetrievedExperience]:
        if not snapshot.ids or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query = query / np.linalg.norm(query)
        rows = []
        for index, experience_id in enumerate(snapshot.ids):
            experience = self.repository.get(experience_id)
            if experience.reliability < self.minimum_reliability:
                continue
            if not self.cross_lingual and experience.language != language:
                continue
            components = {
                "semantic": float(snapshot.vectors[index] @ query) * self.weights.semantic,
                "language": float(experience.language == language) * self.weights.language,
                "source": float(experience.source == source) * self.weights.source,
                "reliability": experience.reliability * self.weights.reliability,
            }
            rows.append((sum(components.values()), experience.id, experience, components))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return [RetrievedExperience(experience=row[2], score=row[0], score_components=row[3], rank=rank)
                for rank, row in enumerate(rows[:k], start=1)]
