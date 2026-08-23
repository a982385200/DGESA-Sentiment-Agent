from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SentimentLabel = Literal["positive", "neutral", "negative"]
Language = Literal["vi", "th", "id", "ms", "km"]
ExperienceType = Literal["successful_case", "error_correction"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictionInput(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language
    source: str = Field(min_length=1)


class SentimentExample(PredictionInput):
    label: SentimentLabel

    def to_prediction_input(self) -> PredictionInput:
        return PredictionInput(**self.model_dump(exclude={"label"}))


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class Prediction(StrictModel):
    sample_id: str = Field(min_length=1)
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    retrieved_experience_ids: tuple[str, ...] = ()
    model_name: str = Field(min_length=1)
    usage: Usage = Usage()
    latency_seconds: float = Field(default=0.0, ge=0.0)
    cache_key: str | None = None


class Feedback(StrictModel):
    sample_id: str = Field(min_length=1)
    predicted_label: SentimentLabel
    gold_label: SentimentLabel
    correct: bool

    @model_validator(mode="after")
    def validate_correct(self) -> Feedback:
        if self.correct != (self.predicted_label == self.gold_label):
            raise ValueError("correct must match predicted and gold labels")
        return self


class Experience(StrictModel):
    id: str = Field(min_length=1)
    type: ExperienceType
    language: Language
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_summary: str = Field(min_length=1)
    sentiment: SentimentLabel
    reason: str = Field(min_length=1)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    source_sample_id: str = Field(min_length=1)
    created_batch: int = Field(ge=0)
    last_used_batch: int = Field(ge=0)
    status: Literal["active"] = "active"


class RetrievedExperience(StrictModel):
    experience: Experience
    score: float
    score_components: dict[str, float]
    rank: int = Field(ge=1)


class ExperienceEvent(StrictModel):
    id: int | None = None
    experience_id: str
    batch_id: int = Field(ge=0)
    event_type: Literal["created", "reinforced", "penalized"]
    old_value: dict | None = None
    new_value: dict
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
