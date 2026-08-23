from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from sentiment_agent.attribution.deterministic import infer_error_type
from sentiment_agent.attribution.models import Attribution, AttributionPayload
from sentiment_agent.evidence.models import CaseEvidence
from sentiment_agent.schemas import StrictModel


class TextCompletionClient(Protocol):
    async def complete(self, messages) -> str: ...


class AttributionResult(StrictModel):
    attribution: Attribution
    used_fallback: bool
    raw_responses: tuple[str, ...] = ()


class LangChainTextClient:
    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    async def complete(self, messages) -> str:
        response = await self.chat_model.ainvoke(list(messages))
        return response.content if isinstance(response.content, str) else json.dumps(response.content)


def _extract_json(content: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start, end = content.find("{"), content.rfind("}")
    return "" if start < 0 or end < start else content[start:end + 1]


class LLMAttributor:
    def __init__(self, client: TextCompletionClient, *, max_retries: int = 2) -> None:
        self.client = client
        self.max_retries = max_retries

    async def attribute(self, case: CaseEvidence, retrieved: Sequence) -> AttributionResult:
        hint = infer_error_type(case, retrieved)
        context = {
            "text": case.text, "language": case.language, "source": case.source,
            "predicted_label": case.predicted_label, "gold_label": case.gold_label,
            "prediction_reason": case.prediction_reason, "deterministic_hint": hint,
            "retrieved_experiences": [
                {"sentiment": _item_value(item, "sentiment"),
                 "rule": _item_value(item, "rule") or _item_value(item, "reason")}
                for item in retrieved
            ],
        }
        messages = [
            SystemMessage(content=(
                "Attribute this sentiment error. Return JSON only with error_type, root_cause, "
                "corrected_reason, candidate_rule, scope_languages, scope_sources, phenomena, confidence."
            )),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
        ]
        raw_responses: list[str] = []
        for attempt in range(self.max_retries + 1):
            raw = await self.client.complete(messages)
            raw_responses.append(raw)
            try:
                payload = AttributionPayload.model_validate_json(_extract_json(raw))
                return AttributionResult(
                    attribution=self._attribution(case, payload, raw),
                    used_fallback=False, raw_responses=tuple(raw_responses),
                )
            except Exception:
                messages.append(HumanMessage(content=(
                    "The previous response was invalid. Return one valid JSON object only."
                )))
        fallback = AttributionPayload(
            error_type=hint,
            root_cause=f"Prediction {case.predicted_label} disagreed with gold label {case.gold_label}.",
            corrected_reason=(
                f"The verified sentiment is {case.gold_label}; the previous reasoning was: "
                f"{case.prediction_reason}"
            ),
            candidate_rule=(
                f"For expressions semantically similar to '{case.text}', prefer sentiment "
                f"{case.gold_label} when the same meaning and context apply."
            ),
            scope_languages=(case.language,), scope_sources=(case.source,),
            phenomena=(hint,), confidence=0.5,
        )
        return AttributionResult(
            attribution=self._attribution(case, fallback, raw_responses[-1] if raw_responses else None),
            used_fallback=True, raw_responses=tuple(raw_responses),
        )

    @staticmethod
    def _attribution(case: CaseEvidence, payload: AttributionPayload,
                     raw: str | None) -> Attribution:
        digest = hashlib.sha256(f"{case.id}\x1fattribution".encode()).hexdigest()[:24]
        return Attribution(
            id=digest, case_id=case.id, created_batch=case.batch_id,
            raw_response=raw, **payload.model_dump(),
        )


def _item_value(item, name: str):
    value = getattr(item, name, None)
    if value is None and hasattr(item, "experience"):
        value = getattr(item.experience, name, None)
    return value
