from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from openai import LengthFinishReasonError
from pydantic import ValidationError

from sentiment_agent.llm.base import (
    LLMResult,
    PredictionPayload,
    TextResult,
    TranslationPayload,
)
from sentiment_agent.schemas import Usage


class LangChainQwenBackend:
    def __init__(self, *, model_name: str, chat_model: Any | None = None,
                 base_url: str | None = None, api_key_env: str = "QWEN_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout_seconds: float = 60.0, max_retries: int = 3,
                 enable_thinking: bool = False) -> None:
        self.model_name = model_name
        if chat_model is None:
            from langchain_openai import ChatOpenAI
            key = os.getenv(api_key_env)
            if not key:
                raise ValueError(f"missing API key environment variable: {api_key_env}")
            chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=key,
                                   temperature=temperature, max_tokens=max_tokens,
                                   timeout=timeout_seconds, max_retries=max_retries,
                                   extra_body={"enable_thinking": enable_thinking})
        self.chat_model = chat_model
        self.structured_output_retries = max_retries
        self.prediction_model = chat_model.with_structured_output(
            PredictionPayload, include_raw=True,
        )
        self.translation_model = chat_model.with_structured_output(
            TranslationPayload, include_raw=True,
        )

    async def classify(self, messages: Sequence[BaseMessage]) -> LLMResult:
        payload, usage, model_calls = await _invoke_structured(
            self.prediction_model, messages, PredictionPayload,
            max_retries=self.structured_output_retries,
        )
        return LLMResult(payload=payload, usage=usage, model_calls=model_calls)

    async def complete_text(self, messages: Sequence[BaseMessage]) -> TextResult:
        payload, usage, model_calls = await _invoke_structured(
            self.translation_model, messages, TranslationPayload,
            max_retries=self.structured_output_retries,
        )
        return TextResult(text=payload.text, usage=usage, model_calls=model_calls)


async def _invoke_structured(runnable, messages, schema, *, max_retries: int):
    total_usage = Usage()
    request_messages = list(messages)
    for attempt in range(max_retries + 1):
        try:
            response = await runnable.ainvoke(request_messages)
        except LengthFinishReasonError as error:
            usage = getattr(error.completion, "usage", None)
            if usage is not None:
                total_usage = Usage(
                    input_tokens=total_usage.input_tokens + usage.prompt_tokens,
                    output_tokens=total_usage.output_tokens + usage.completion_tokens,
                )
            if attempt == max_retries:
                raise
            request_messages = _length_corrected_messages(request_messages, schema)
            continue
        except ValidationError as error:
            if attempt == max_retries:
                raise
            request_messages = _corrected_messages(request_messages, schema, error)
            continue
        response_usage = _response_usage(response)
        total_usage = Usage(
            input_tokens=total_usage.input_tokens + response_usage.input_tokens,
            output_tokens=total_usage.output_tokens + response_usage.output_tokens,
        )
        try:
            payload, _ = _structured_value(response, schema)
        except Exception as error:
            if attempt == max_retries:
                raise
            request_messages = _corrected_messages(request_messages, schema, error)
            continue
        return payload, total_usage, attempt + 1
    raise RuntimeError("unreachable structured output retry state")


def _corrected_messages(messages, schema, error: Exception):
    allowed_fields = ", ".join(schema.model_fields)
    details = str(error).replace("\n", " ")[:800]
    return [
        *messages,
        HumanMessage(content=(
            "Your previous output failed Pydantic validation. Do not repeat the invalid "
            f"output. Validation error: {details}. Return exactly one JSON object containing "
            f"only these fields: {allowed_fields}. Do not include input metadata, Markdown, "
            "or any text outside the JSON object."
        )),
    ]


def _length_corrected_messages(messages, schema):
    allowed_fields = ", ".join(schema.model_fields)
    return [
        *messages,
        HumanMessage(content=(
            "Your previous output exceeded the output token limit and was truncated. Do not "
            "repeat it. Return exactly one compact JSON object containing only these fields: "
            f"{allowed_fields}. Use one short sentence for each explanatory text field. Output "
            "only the final values—no analysis steps, internal reasoning, examples, experience "
            "lists, alternatives, Markdown, or text outside the JSON object."
        )),
    ]


def _structured_value(response: Any, schema: type[PredictionPayload] | type[TranslationPayload]):
    parsing_error = response.get("parsing_error")
    if parsing_error is not None:
        raise parsing_error
    payload = schema.model_validate(response.get("parsed"))
    usage = _response_usage(response)
    return payload, usage


def _response_usage(response: Any) -> Usage:
    metadata = getattr(response.get("raw"), "usage_metadata", None) or {}
    usage = Usage(
        input_tokens=int(metadata.get("input_tokens", 0)),
        output_tokens=int(metadata.get("output_tokens", 0)),
    )
    return usage
