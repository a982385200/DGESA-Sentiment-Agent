from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

SentimentLabel = Literal["negative", "neutral", "positive"]
Split = Literal["train", "dev", "test"]
LanguageCode = Literal["vi", "th", "id", "ms", "km"]
ExperienceType = Literal["successful_case", "error_correction", "generalized_rule"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SentimentExample(StrictModel):
    id: str = Field(min_length=1)
    text: str
    language: LanguageCode
    source: str = Field(min_length=1)
    split: Split
    label: SentimentLabel

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


class PredictionInput(StrictModel):
    id: str = Field(min_length=1)
    text: str
    language: LanguageCode
    source: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


class Usage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Prediction(StrictModel):
    sample_id: str
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    strategy: str
    retrieved_experience_ids: list[str] = Field(default_factory=list)
    model_name: str
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    cache_key: str | None = None


class Feedback(StrictModel):
    sample_id: str
    predicted_label: SentimentLabel
    gold_label: SentimentLabel
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def correct(self) -> bool:
        return self.predicted_label == self.gold_label


class Experience(StrictModel):
    id: str | None = None
    text: str
    language: LanguageCode
    source: str
    semantic_meaning: str
    sentiment: SentimentLabel
    reason: str
    experience_type: ExperienceType
    success_count: int = Field(default=1, ge=0)
    failure_count: int = Field(default=0, ge=0)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    created_round: int = Field(default=0, ge=0)
    last_used_round: int = Field(default=0, ge=0)


class ExperimentRecord(StrictModel):
    experiment_id: str
    config_hash: str
    seed: int
    git_commit: str | None = None
    model_name: str
    dataset_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    estimated_cost: float = Field(default=0.0, ge=0.0)
