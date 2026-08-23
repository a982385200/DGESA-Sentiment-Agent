from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sentiment_agent.llm.base import PredictionPayload
from sentiment_agent.schemas import PredictionInput, RetrievedExperience

BASE_PROMPT = """You are a multilingual sentiment analysis model. Analyze the input text in four steps:

1. Language identification: identify the primary language, such as Vietnamese or Thai.
2. Domain identification: infer the main domain, such as education or social media.
3. Sentiment classification: classify the sentiment as positive, neutral, or negative based on the overall meaning and dominant sentiment.
4. Reason generation: provide a concise reason for the classification.

Use only the input text. Return only JSON:
{"language":"...","domain":"...","label":"positive|neutral|negative","reason":"..."}"""


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
                    "native_rule": experience.native_rules.get(item.language, ""),
                    "applies_when": experience.applies_when,
                    "excludes_when": experience.excludes_when,
                    "corrected_reason": experience.corrected_reason,
                    "scope": experience.scope,
                    "scope_languages": experience.scope_languages,
                    "scope_sources": experience.scope_sources,
                    "phenomena": experience.phenomena,
                    "language_cues": experience.language_cues.get(item.language, ()),
                    "reliability": experience.reliability,
                })
            else:
                context.append({
                    "sentiment": experience.sentiment,
                    "semantic_summary": experience.semantic_summary,
                    "reason": experience.reason,
                    "reliability": experience.reliability,
                })
        payload = {"text": item.text}
        if context:
            payload["relevant_experiences"] = context
        return [SystemMessage(content=BASE_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False))]

    def build_translation(self, item: PredictionInput, translation: str) -> list[BaseMessage]:
        payload = {
            "id": item.id, "language": item.language, "source": item.source,
            "text": item.text, "translation": translation, "relevant_experiences": [],
        }
        return [
            SystemMessage(content=BASE_PROMPT + " Use the English translation to clarify meaning."),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

    def build_reflection(self, item: PredictionInput,
                         initial: PredictionPayload) -> list[BaseMessage]:
        payload = {
            "id": item.id, "language": item.language, "source": item.source,
            "text": item.text, "initial_prediction": initial.model_dump(mode="json"),
        }
        return [
            SystemMessage(content=(
                BASE_PROMPT + " Verify the initial prediction, correct it if needed, and return "
                "the final JSON classification. Do not copy initial_prediction or any of its "
                "metadata into the output."
            )),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
