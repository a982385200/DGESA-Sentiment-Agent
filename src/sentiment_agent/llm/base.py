from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import Field

from sentiment_agent.schemas import SentimentLabel, StrictModel, Usage


class PredictionPayload(StrictModel):
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class LLMResult(StrictModel):
    payload: PredictionPayload
    usage: Usage = Usage()
    cache_key: str | None = None


class LLMBackend(Protocol):
    async def classify(self, messages: Sequence[BaseMessage]) -> LLMResult: ...
