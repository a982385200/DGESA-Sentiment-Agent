from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from sentiment_agent.schemas import Language, SentimentLabel, StrictModel


class CaseEvidence(StrictModel):
    id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language
    source: str = Field(min_length=1)
    predicted_label: SentimentLabel
    gold_label: SentimentLabel
    prediction_reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    retrieved_experience_ids: tuple[str, ...] = ()
    batch_id: int = Field(ge=1)
    correct: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
