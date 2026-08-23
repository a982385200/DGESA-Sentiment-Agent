from __future__ import annotations

import numpy as np

from sentiment_agent.experience.retrieval import RetrievalWeights
from sentiment_agent.experience.vector_index import VectorIndex, VectorSnapshot
from sentiment_agent.generalization.models import GeneralizedExperience
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.schemas import Language, StrictModel


class RetrievedGeneralizedExperience(StrictModel):
    experience: GeneralizedExperience
    score: float
    score_components: dict[str, float]
    rank: int


class GeneralizedExperienceRetriever:
    def __init__(self, repository: EvolutionRepository, vector_index: VectorIndex,
                 weights: RetrievalWeights, *, minimum_reliability: float = 0.0,
                 cross_lingual: bool = True) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.weights = weights
        self.minimum_reliability = minimum_reliability
        self.cross_lingual = cross_lingual

    def search(self, vector: np.ndarray, snapshot: VectorSnapshot, *, language: Language,
               source: str, k: int) -> list[RetrievedGeneralizedExperience]:
        if not snapshot.ids or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query = query / np.linalg.norm(query)
        rows = []
        for index, rule_id in enumerate(snapshot.ids):
            rule = self.repository.get_rule(rule_id)
            if rule.status != "active" or rule.reliability < self.minimum_reliability:
                continue
            language_match = not rule.scope_languages or language in rule.scope_languages
            if not self.cross_lingual and not language_match:
                continue
            source_match = not rule.scope_sources or source in rule.scope_sources
            components = {
                "semantic": float(snapshot.vectors[index] @ query) * self.weights.semantic,
                "language": float(language_match) * self.weights.language,
                "source": float(source_match) * self.weights.source,
                "reliability": rule.reliability * self.weights.reliability,
            }
            rows.append((sum(components.values()), rule.id, rule, components))
        rows.sort(key=lambda value: (-value[0], value[1]))
        return [RetrievedGeneralizedExperience(
            experience=row[2], score=row[0], score_components=row[3], rank=rank,
        ) for rank, row in enumerate(rows[:k], start=1)]
