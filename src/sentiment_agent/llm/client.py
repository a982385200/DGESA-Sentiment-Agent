from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel

from sentiment_agent.config import ModelConfig
from sentiment_agent.llm.cache import SQLiteResponseCache, build_cache_key
from sentiment_agent.llm.parsing import PredictionPayload, parse_model_json
from sentiment_agent.schemas import LanguageCode, Usage

PayloadT = TypeVar("PayloadT", bound=BaseModel)


@dataclass(frozen=True)
class LLMResult(Generic[PayloadT]):
    payload: PayloadT
    usage: Usage
    attempts: int
    cache_key: str
    cached: bool


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class OpenAICompatibleClient:
    RETRYABLE_STATUSES = frozenset({408, 409, 429})

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str,
        http_client: httpx.Client | None = None,
        cache: SQLiteResponseCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.config = config
        self._client = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._cache = cache
        self._sleep = sleep
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def classify(self, text: str, language: LanguageCode) -> LLMResult[PredictionPayload]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify sentiment as negative, neutral, or positive. "
                    "Return JSON with label, confidence from 0 to 1, and reason."
                ),
            },
            {"role": "user", "content": f"Language: {language}\nText: {text}"},
        ]
        return self.chat_json(messages, PredictionPayload)

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[PayloadT],
    ) -> LLMResult[PayloadT]:
        request_payload: dict[str, Any] = {
            "model": self.config.name,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        cache_key = build_cache_key(request_payload)
        cached = self._cache.get(cache_key) if self._cache is not None else None
        if cached is not None:
            return self._parse_chat_response(cached, response_model, attempts=0, cache_key=cache_key, cached=True)

        response_json, attempts = self._post_with_retries("chat/completions", request_payload)
        if self._cache is not None:
            self._cache.put(cache_key, response_json)
        return self._parse_chat_response(
            response_json,
            response_model,
            attempts=attempts,
            cache_key=cache_key,
            cached=False,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        request_payload = {
            "model": self.config.embedding_model or self.config.name,
            "input": list(texts),
        }
        response_json, _ = self._post_with_retries("embeddings", request_payload)
        data = response_json.get("data")
        if not isinstance(data, list):
            raise LLMRequestError("embedding response is missing data", attempts=1)
        ordered = sorted(data, key=lambda item: item["index"])
        return [[float(value) for value in item["embedding"]] for item in ordered]

    def _post_with_retries(self, endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        last_status: int | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self._client.post(
                    f"{self.config.base_url.rstrip('/')}/{endpoint}",
                    headers=self._headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.config.max_attempts:
                    raise LLMRequestError(
                        f"request failed after {attempt} attempts: {type(exc).__name__}",
                        attempts=attempt,
                    ) from exc
                self._sleep(0.5 * (2 ** (attempt - 1)))
                continue

            last_status = response.status_code
            if response.is_success:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise LLMRequestError("API returned invalid JSON", attempts=attempt, status_code=last_status) from exc
                if not isinstance(body, dict):
                    raise LLMRequestError("API response must be a JSON object", attempts=attempt, status_code=last_status)
                return body, attempt

            retryable = response.status_code in self.RETRYABLE_STATUSES or response.status_code >= 500
            if not retryable or attempt == self.config.max_attempts:
                raise LLMRequestError(
                    f"API request failed with HTTP {response.status_code}",
                    attempts=attempt,
                    status_code=response.status_code,
                )
            self._sleep(0.5 * (2 ** (attempt - 1)))

        raise LLMRequestError(
            "request failed without a response",
            attempts=self.config.max_attempts,
            status_code=last_status,
        )

    @staticmethod
    def _parse_chat_response(
        response: dict[str, Any],
        response_model: type[PayloadT],
        *,
        attempts: int,
        cache_key: str,
        cached: bool,
    ) -> LLMResult[PayloadT]:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("chat response is missing message content", attempts=max(attempts, 1)) from exc
        usage = response.get("usage") or {}
        parsed = parse_model_json(str(content), response_model)
        return LLMResult(
            payload=parsed,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            attempts=attempts,
            cache_key=cache_key,
            cached=cached,
        )
