from __future__ import annotations

import numpy as np

from sentiment_agent.experience.vector_index import VectorIndex, VectorSnapshot
from sentiment_agent.generalization.models import GeneralizedExperience
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.schemas import SentimentLabel


class RuleMatcher:
    def __init__(self, repository: EvolutionRepository, vector_index: VectorIndex,
                 *, merge_similarity: float) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.merge_similarity = merge_similarity

    def find_match(self, vector: np.ndarray, *, sentiment: SentimentLabel,
                   snapshot: VectorSnapshot | None = None) -> GeneralizedExperience | None:
        current = snapshot or self.vector_index.snapshot()
        if not current.ids:
            return None
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query = query / np.linalg.norm(query)
        candidates = []
        for index, rule_id in enumerate(current.ids):
            rule = self.repository.get_rule(rule_id)
            if rule.sentiment != sentiment or rule.status == "suppressed":
                continue
            similarity = float(current.vectors[index] @ query)
            if similarity >= self.merge_similarity:
                candidates.append((similarity, rule.id, rule))
        if not candidates:
            return None
        candidates.sort(key=lambda value: (-value[0], value[1]))
        return candidates[0][2]
