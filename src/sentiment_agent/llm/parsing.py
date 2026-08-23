from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from sentiment_agent.schemas import SentimentLabel

PayloadT = TypeVar("PayloadT", bound=BaseModel)


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def parse_model_json(content: str, response_model: type[PayloadT]) -> PayloadT:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        return response_model.model_validate(payload)
    raise ValueError("model response does not contain a valid JSON object")
