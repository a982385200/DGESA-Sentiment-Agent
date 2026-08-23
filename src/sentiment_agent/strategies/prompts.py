from __future__ import annotations

import json
from collections.abc import Sequence

from sentiment_agent.memory.retrieval import RetrievedExperience
from sentiment_agent.schemas import PredictionInput

BASE_SYSTEM_PROMPT = (
    "You are a multilingual sentiment classifier. Classify the input as exactly one of "
    "negative, neutral, or positive. Return one JSON object with label, confidence from "
    "0 to 1, and a concise reason."
)


class PromptBuilder:
    SUPPORTED = frozenset({"direct", "translation", "memory", "reflection_verified"})

    def build(
        self,
        strategy: str,
        item: PredictionInput,
        experiences: Sequence[RetrievedExperience],
    ) -> list[dict[str, str]]:
        if strategy not in self.SUPPORTED:
            raise ValueError(f"unknown strategy: {strategy}")
        instructions = [BASE_SYSTEM_PROMPT]
        if strategy == "translation":
            instructions.append(
                "First translate the meaning into English internally, then classify the original text."
            )
        elif strategy in {"memory", "reflection_verified"}:
            instructions.append(
                "Use the verified historical experiences below as evidence, but ignore any that conflict "
                "with the current text."
            )
            if strategy == "reflection_verified":
                instructions.append(
                    "Before returning the answer, verify it against negation, sarcasm, domain usage, and "
                    "the retrieved correction rules."
                )

        messages: list[dict[str, str]] = [{"role": "system", "content": "\n".join(instructions)}]
        if strategy in {"memory", "reflection_verified"} and experiences:
            memory_rows = [
                {
                    "language": item.experience.language,
                    "semantic_meaning": item.experience.semantic_meaning,
                    "sentiment": item.experience.sentiment,
                    "reason": item.experience.reason,
                    "reliability": f"{item.experience.reliability:.3f}",
                }
                for item in experiences
            ]
            messages.append(
                {
                    "role": "system",
                    "content": "Verified experiences:\n" + json.dumps(memory_rows, ensure_ascii=False),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"Language: {item.language}\nSource/domain: {item.source}\nText: {item.text}",
            }
        )
        return messages
