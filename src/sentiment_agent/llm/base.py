from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import Field, PositiveInt

from sentiment_agent.schemas import SentimentLabel, StrictModel, Usage


class PredictionPayload(StrictModel):
    language: str = Field(min_length=1, max_length=80)
    domain: str = Field(min_length=1, max_length=80)
    label: SentimentLabel
    reason: str = Field(min_length=1, max_length=240)


class TranslationPayload(StrictModel):
    text: str = Field(min_length=1)


class LLMResult(StrictModel):
    payload: PredictionPayload
    usage: Usage = Usage()
    cache_key: str | None = None
    model_calls: PositiveInt = 1


class TextResult(StrictModel):
    text: str = Field(min_length=1)
    usage: Usage = Usage()
    model_calls: PositiveInt = 1


class LLMBackend(Protocol):
    async def classify(self, messages: Sequence[BaseMessage]) -> LLMResult: ...

    async def complete_text(self, messages: Sequence[BaseMessage]) -> TextResult: ...
