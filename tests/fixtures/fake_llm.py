from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from sentiment_agent.llm.parsing import PredictionPayload
from sentiment_agent.reflection.reflector import ReflectionPayload
from sentiment_agent.schemas import Usage


class FakeLLMClient:
    def embed(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([digest[0] / 255, digest[1] / 255, digest[2] / 255])
        return vectors

    def chat_json(self, messages, response_model):
        rendered = json.dumps(messages, ensure_ascii=False).casefold()
        if response_model is PredictionPayload:
            if " bad" in rendered or "negative sample" in rendered:
                label = "negative"
            elif " okay" in rendered or "neutral sample" in rendered:
                label = "neutral"
            else:
                label = "positive"
            return SimpleNamespace(
                payload=PredictionPayload(label=label, confidence=0.9, reason="deterministic fixture"),
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                cache_key=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            )
        if response_model is ReflectionPayload:
            return SimpleNamespace(
                payload=ReflectionPayload(
                    error_type="none",
                    corrected_reason="Use the explicit fixture polarity cue.",
                    generalized_rule="Explicit fixture polarity cues determine sentiment.",
                    scope="fixture",
                )
            )
        raise AssertionError(f"unsupported response model: {response_model}")
