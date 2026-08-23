from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from sentiment_agent.attribution.models import ErrorType
from sentiment_agent.schemas import Language, SentimentLabel, StrictModel

RuleStatus = Literal["candidate", "active", "conflicted", "suppressed"]


class GeneralizedExperience(StrictModel):
    id: str = Field(min_length=1)
    status: RuleStatus = "candidate"
    semantic: str = Field(min_length=1)
    sentiment: SentimentLabel
    rule: str = Field(min_length=1)
    corrected_reason: str = Field(min_length=1)
    error_types: tuple[ErrorType, ...] = ()
    scope_languages: tuple[Language, ...] = ()
    scope_sources: tuple[str, ...] = ()
    phenomena: tuple[str, ...] = ()
    support_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    supporting_batches: tuple[int, ...] = ()
    created_batch: int = Field(ge=1)
    last_updated_batch: int = Field(ge=1)
    version: int = Field(default=1, ge=1)

    @computed_field
    @property
    def reliability(self) -> float:
        return (self.support_count + 1) / (
            self.support_count + self.contradiction_count + 2
        )

    @computed_field
    @property
    def contradiction_ratio(self) -> float:
        total = self.support_count + self.contradiction_count
        return 0.0 if total == 0 else self.contradiction_count / total
