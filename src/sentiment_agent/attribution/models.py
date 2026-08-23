from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from sentiment_agent.schemas import Language, StrictModel

ErrorType = Literal[
    "missing_knowledge", "wrong_experience", "retrieval_failure",
    "negative_transfer", "negation_error", "domain_knowledge_error",
    "reasoning_error", "label_ambiguity",
]


class AttributionPayload(StrictModel):
    error_type: ErrorType
    root_cause: str = Field(min_length=1)
    corrected_reason: str = Field(min_length=1)
    candidate_rule: str = Field(min_length=1)
    scope_languages: tuple[Language, ...] = ()
    scope_sources: tuple[str, ...] = ()
    phenomena: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class Attribution(AttributionPayload):
    id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    raw_response: str | None = None
    status: Literal["pending", "merged"] = "pending"
    created_batch: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
