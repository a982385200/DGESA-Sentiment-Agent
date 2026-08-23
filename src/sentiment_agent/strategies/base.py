from __future__ import annotations

from typing import Literal, Protocol

from sentiment_agent.memory.retrieval import RetrievedExperience
from sentiment_agent.schemas import PredictionInput

StrategyName = Literal["direct", "translation", "memory", "reflection_verified"]


class PromptStrategy(Protocol):
    def build(
        self,
        item: PredictionInput,
        experiences: list[RetrievedExperience],
    ) -> list[dict[str, str]]: ...
