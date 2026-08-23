from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sentiment_agent.schemas import PredictionInput, RetrievedExperience

BASE_PROMPT = """You are a multilingual sentiment classifier.
Classify the input as exactly one of: positive, neutral, negative.
Judge the complete meaning, including negation, irony, degree, and local expressions.
Use only the input and supplied experiences. Return JSON with label, confidence, and reason."""


class PredictionPromptBuilder:
    def build(self, item: PredictionInput,
              experiences: Sequence[RetrievedExperience]) -> list[BaseMessage]:
        context = []
        for result in experiences:
            experience = result.experience
            if hasattr(experience, "rule"):
                context.append({
                    "semantic": experience.semantic,
                    "sentiment": experience.sentiment,
                    "rule": experience.rule,
                    "corrected_reason": experience.corrected_reason,
                    "scope_languages": experience.scope_languages,
                    "scope_sources": experience.scope_sources,
                    "reliability": experience.reliability,
                })
            else:
                context.append({
                    "sentiment": experience.sentiment,
                    "semantic_summary": experience.semantic_summary,
                    "reason": experience.reason,
                    "reliability": experience.reliability,
                })
        payload = {"id": item.id, "language": item.language, "source": item.source,
                   "text": item.text, "relevant_experiences": context}
        return [SystemMessage(content=BASE_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
