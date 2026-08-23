from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from sentiment_agent.llm.base import LLMResult, PredictionPayload
from sentiment_agent.schemas import Usage


class LangChainQwenBackend:
    def __init__(self, *, model_name: str, chat_model: Any | None = None,
                 base_url: str | None = None, api_key_env: str = "QWEN_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 256,
                 timeout_seconds: float = 60.0, max_retries: int = 3) -> None:
        self.model_name = model_name
        if chat_model is None:
            from langchain_openai import ChatOpenAI
            key = os.getenv(api_key_env)
            if not key:
                raise ValueError(f"missing API key environment variable: {api_key_env}")
            chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=key,
                                   temperature=temperature, max_tokens=max_tokens,
                                   timeout=timeout_seconds, max_retries=max_retries)
        self.chat_model = chat_model

    async def classify(self, messages: Sequence[BaseMessage]) -> LLMResult:
        response = await self.chat_model.ainvoke(list(messages))
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        raw = match.group(1) if match else content[content.find("{"):content.rfind("}") + 1]
        payload = PredictionPayload.model_validate_json(raw)
        metadata = getattr(response, "usage_metadata", None) or {}
        usage = Usage(input_tokens=int(metadata.get("input_tokens", 0)),
                      output_tokens=int(metadata.get("output_tokens", 0)))
        return LLMResult(payload=payload, usage=usage)
