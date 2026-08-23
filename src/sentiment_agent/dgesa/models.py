from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from sentiment_agent.schemas import Language, SentimentLabel, StrictModel

PatternStatus = Literal["candidate", "active", "suppressed"]
PatternScope = Literal["local", "language", "global"]


class SampleExperience(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    experience: str = Field(min_length=1)
    sentiment: SentimentLabel
    language: Language
    source: str = Field(min_length=1)
    source_sample_id: str = Field(min_length=1)
    created_batch: int = Field(ge=1)


class PatternExperience(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sentiment: SentimentLabel
    source_language: Language
    status: PatternStatus = "candidate"
    scope: PatternScope = "local"
    evidence_texts: tuple[str, ...] = ()
    support_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    support_by_language: dict[Language, int] = {}
    contradiction_by_language: dict[Language, int] = {}
    created_batch: int = Field(ge=1)
    last_updated_batch: int = Field(ge=1)

    @computed_field
    @property
    def reliability(self) -> float:
        return (self.support_count + 1) / (
            self.support_count + self.contradiction_count + 2
        )

    @computed_field
    @property
    def conflict_ratio(self) -> float:
        total = self.support_count + self.contradiction_count
        return 0.0 if total == 0 else self.contradiction_count / total


class DualExperiencePayload(StrictModel):
    sample_experience: str = Field(min_length=1)
    pattern_experience: str = Field(min_length=1)
    pattern_label: SentimentLabel


class SampleAdmissionPayload(StrictModel):
    admission: Literal["informative", "redundant"]


class PatternAlignmentPayload(StrictModel):
    alignment: str = Field(pattern=r"^(new|align\([^)]+\))$")


class PatternAbstractionPayload(StrictModel):
    updated_pattern_experience: str = Field(min_length=1)


class PaperPredictionPayload(StrictModel):
    language: str = Field(min_length=1)
    sentiment: SentimentLabel
    reason: str = Field(min_length=1)


class RetrievedSample(StrictModel):
    experience: SampleExperience
    similarity: float
    rank: int = Field(ge=1)


class RetrievedPattern(StrictModel):
    experience: PatternExperience
    similarity: float
    score: float
    rank: int = Field(ge=1)


class PaperPrediction(PaperPredictionPayload):
    sample_id: str = Field(min_length=1)
    sample_experience_ids: tuple[str, ...] = ()
    pattern_experience_ids: tuple[str, ...] = ()


class PaperEvaluation(StrictModel):
    predictions: tuple[PaperPrediction, ...]
    metrics: dict
