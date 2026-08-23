from __future__ import annotations

from pydantic import Field

from sentiment_agent.config import MemoryConfig
from sentiment_agent.memory.scoring import cosine_similarity
from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.schemas import Experience, LanguageCode, StrictModel


class RetrievedExperience(StrictModel):
    experience: Experience
    score: float
    semantic_similarity: float = Field(ge=-1.0, le=1.0)


class ExperienceRetriever:
    def __init__(self, store: ExperienceStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config

    def search(
        self,
        query_vector: list[float],
        *,
        language: LanguageCode,
        source: str,
        cross_lingual: bool,
        k: int,
        current_round: int = 0,
    ) -> list[RetrievedExperience]:
        if k < 0:
            raise ValueError("k must not be negative")
        if k == 0:
            return []
        results: list[RetrievedExperience] = []
        for experience, vector in self.store.all_with_vectors():
            if not cross_lingual and experience.language != language:
                continue
            if experience.reliability < self.config.min_reliability:
                continue
            semantic_similarity = cosine_similarity(query_vector, vector)
            language_match = float(experience.language == language)
            domain_match = float(experience.source == source)
            age = max(0, current_round - experience.last_used_round)
            recency = 1.0 / (1.0 + age)
            score = (
                self.config.semantic_weight * semantic_similarity
                + self.config.language_weight * language_match
                + self.config.domain_weight * domain_match
                + self.config.reliability_weight * experience.reliability
                + self.config.recency_weight * recency
            )
            results.append(
                RetrievedExperience(
                    experience=experience,
                    score=score,
                    semantic_similarity=semantic_similarity,
                )
            )
        results.sort(key=lambda item: (-item.score, item.experience.id or ""))
        return results[:k]
