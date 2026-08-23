from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from sentiment_agent.memory.retrieval import RetrievedExperience
from sentiment_agent.schemas import Feedback, Prediction, PredictionInput, StrictModel


class ReflectionPayload(StrictModel):
    error_type: str = Field(min_length=1)
    corrected_reason: str = Field(min_length=1)
    generalized_rule: str = Field(min_length=1)
    scope: str = Field(min_length=1)


class ReflectionClient(Protocol):
    def chat_json(self, messages: Sequence[dict[str, str]], response_model: type[ReflectionPayload]): ...


class Reflector:
    def __init__(self, client: ReflectionClient, *, enabled: bool) -> None:
        self.client = client
        self.enabled = enabled

    def reflect(
        self,
        item: PredictionInput,
        prediction: Prediction,
        feedback: Feedback,
        experiences: Sequence[RetrievedExperience],
    ) -> ReflectionPayload | None:
        if not self.enabled:
            return None
        context = {
            "text": item.text,
            "language": item.language,
            "source": item.source,
            "predicted_label": prediction.label,
            "predicted_reason": prediction.reason,
            "gold_label": feedback.gold_label,
            "correct": feedback.correct,
            "retrieved_experiences": [
                {
                    "sentiment": result.experience.sentiment,
                    "reason": result.experience.reason,
                    "reliability": result.experience.reliability,
                }
                for result in experiences
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Analyze the prediction feedback and return JSON with error_type, corrected_reason, "
                    "generalized_rule, and scope. The rule must not copy the sample text verbatim."
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        return self.client.chat_json(messages, ReflectionPayload).payload
