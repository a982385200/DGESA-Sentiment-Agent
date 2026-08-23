from pathlib import Path

import langchain_openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from sentiment_agent.llm.cache import ResponseCache
from sentiment_agent.llm.base import PredictionPayload, TranslationPayload
from sentiment_agent.llm.langchain_qwen import LangChainQwenBackend
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import PredictionInput


class FakeChatModel:
    def __init__(self) -> None:
        self.schemas = []

    def with_structured_output(self, schema, *, include_raw=False):
        self.schemas.append((schema, include_raw))
        payload = (
            PredictionPayload(language="Vietnamese", domain="education",
                              label="negative", reason="negation")
            if schema is PredictionPayload
            else TranslationPayload(text="I did not receive the money.")
        )

        class Runnable:
            async def ainvoke(self, messages):
                return {
                    "raw": AIMessage(content="", usage_metadata={
                        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                    }),
                    "parsed": payload,
                    "parsing_error": None,
                }

        return Runnable()


def test_base_prompt_is_stable() -> None:
    builder = PredictionPromptBuilder()
    item = PredictionInput(id="x", text="not good", language="vi", source="tiny")
    assert builder.build(item, [])[0].content == builder.build(item, [])[0].content


def test_prediction_prompt_requires_only_exact_json_shape() -> None:
    prompt = PredictionPromptBuilder().build(
        PredictionInput(id="x", text="not good", language="vi", source="tiny"), []
    )[0].content
    assert '"language":"..."' in prompt
    assert '"domain":"..."' in prompt
    assert '"label":"positive|neutral|negative"' in prompt
    assert '"reason":"..."' in prompt
    assert "confidence" not in prompt


def test_prediction_prompt_uses_four_step_method() -> None:
    prompt = PredictionPromptBuilder().build(
        PredictionInput(id="x", text="mixed feedback", language="vi", source="tiny"), []
    )[0].content
    assert "1. Language identification" in prompt
    assert "such as Vietnamese or Thai" in prompt
    assert "2. Domain identification" in prompt
    assert "such as education or social media" in prompt
    assert "3. Sentiment classification" in prompt
    assert "overall meaning and dominant sentiment" in prompt
    assert "4. Reason generation" in prompt


def test_zero_shot_prompt_sends_only_input_text() -> None:
    messages = PredictionPromptBuilder().build(
        PredictionInput(id="x", text="not good", language="vi", source="tiny"), []
    )
    assert messages[-1].content == '{"text": "not good"}'


def test_prediction_payload_matches_method_output_without_confidence() -> None:
    payload = PredictionPayload(
        language="Vietnamese", domain="education", label="negative", reason="Complaint.",
    )
    assert payload.model_dump() == {
        "language": "Vietnamese", "domain": "education",
        "label": "negative", "reason": "Complaint.",
    }


@pytest.mark.anyio
async def test_qwen_backend_parses_fenced_json() -> None:
    chat_model = FakeChatModel()
    backend = LangChainQwenBackend(model_name="qwen", chat_model=chat_model)
    result = await backend.classify(PredictionPromptBuilder().build(
        PredictionInput(id="x", text="not good", language="vi", source="tiny"), []))
    assert result.payload.label == "negative"
    assert result.usage.input_tokens == 10
    assert (PredictionPayload, True) in chat_model.schemas


@pytest.mark.anyio
async def test_qwen_backend_returns_text_with_usage() -> None:
    chat_model = FakeChatModel()
    backend = LangChainQwenBackend(model_name="qwen", chat_model=chat_model)
    result = await backend.complete_text([HumanMessage(content="translate")])
    assert result.text == "I did not receive the money."
    assert result.usage.input_tokens == 10
    assert (TranslationPayload, True) in chat_model.schemas


@pytest.mark.anyio
async def test_qwen_backend_retries_structured_validation_error() -> None:
    class InvalidThenValidChatModel:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema, *, include_raw=False):
            parent = self

            class Runnable:
                async def ainvoke(self, messages):
                    parent.calls += 1
                    parsed = (
                        {
                            "language": "Indonesian", "domain": "product review",
                            "label": "positive", "reason": "favorable",
                            "indicating a highly favorable sentiment": "extra",
                        }
                        if parent.calls == 1
                        else PredictionPayload(
                            language="Indonesian", domain="product review",
                            label="positive", reason="favorable",
                        )
                    )
                    return {
                        "raw": AIMessage(content="", usage_metadata={
                            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        }),
                        "parsed": parsed,
                        "parsing_error": None,
                    }

            return Runnable()

    chat_model = InvalidThenValidChatModel()
    backend = LangChainQwenBackend(
        model_name="qwen", chat_model=chat_model, max_retries=1,
    )
    result = await backend.classify([HumanMessage(content="JSON sentiment")])
    assert result.payload.label == "positive"
    assert result.model_calls == 2
    assert chat_model.calls == 2


@pytest.mark.anyio
async def test_qwen_backend_retries_provider_side_pydantic_error() -> None:
    class ProviderParsingChatModel:
        def __init__(self) -> None:
            self.calls = 0
            self.messages = []

        def with_structured_output(self, schema, *, include_raw=False):
            parent = self

            class Runnable:
                async def ainvoke(self, messages):
                    parent.calls += 1
                    parent.messages.append(list(messages))
                    if parent.calls == 1:
                        return PredictionPayload.model_validate({
                            "language": "Vietnamese", "domain": "education",
                            "label": "positive", "reason": "favorable", "source": "test",
                        })
                    return {
                        "raw": AIMessage(content="", usage_metadata={
                            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        }),
                        "parsed": PredictionPayload(
                            language="Vietnamese", domain="education",
                            label="positive", reason="favorable",
                        ),
                        "parsing_error": None,
                    }

            return Runnable()

    chat_model = ProviderParsingChatModel()
    result = await LangChainQwenBackend(
        model_name="qwen", chat_model=chat_model, max_retries=1,
    ).classify([HumanMessage(content="JSON sentiment")])
    assert result.payload.label == "positive"
    assert result.model_calls == 2
    assert chat_model.calls == 2
    retry_instruction = chat_model.messages[1][-1].content
    assert "previous output failed Pydantic validation" in retry_instruction
    assert "source" in retry_instruction
    assert "language, domain, label, reason" in retry_instruction
    assert "Do not repeat the invalid output" in retry_instruction


def test_prediction_payload_rejects_overlong_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        PredictionPayload(
            language="Thai", domain="social media",
            label="positive", reason="x" * 241,
        )


@pytest.mark.anyio
async def test_qwen_backend_retries_length_truncation_with_concise_instruction() -> None:
    completion = ChatCompletion.model_validate({
        "id": "length-test",
        "choices": [{
            "finish_reason": "length", "index": 0, "logprobs": None,
            "message": {"content": "{", "role": "assistant"},
        }],
        "created": 0,
        "model": "qwen",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 256, "prompt_tokens": 100, "total_tokens": 356,
        },
    })

    class LengthThenValidChatModel:
        def __init__(self) -> None:
            self.calls = 0
            self.messages = []

        def with_structured_output(self, schema, *, include_raw=False):
            parent = self

            class Runnable:
                async def ainvoke(self, messages):
                    parent.calls += 1
                    parent.messages.append(list(messages))
                    if parent.calls == 1:
                        raise LengthFinishReasonError(completion=completion)
                    return {
                        "raw": AIMessage(content="", usage_metadata={
                            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        }),
                        "parsed": PredictionPayload(
                            language="Thai", domain="social media",
                            label="positive", reason="Brief positive signal.",
                        ),
                        "parsing_error": None,
                    }

            return Runnable()

    chat_model = LengthThenValidChatModel()
    result = await LangChainQwenBackend(
        model_name="qwen", chat_model=chat_model, max_retries=1,
    ).classify([HumanMessage(content="JSON sentiment")])
    assert result.payload.reason == "Brief positive signal."
    assert result.model_calls == 2
    assert result.usage.input_tokens == 110
    assert result.usage.output_tokens == 261
    retry_instruction = chat_model.messages[1][-1].content
    assert "exceeded the output token limit" in retry_instruction
    assert "one short sentence" in retry_instruction


def test_response_cache_round_trip(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "cache.sqlite3") as cache:
        key = cache.key("qwen", {"temperature": 0}, ["a"])
        cache.put(key, '{"ok":true}')
        assert cache.get(key) == '{"ok":true}'
        assert key != cache.key("qwen", {"temperature": 0}, ["b"])


def test_qwen_backend_disables_thinking_in_openai_request(monkeypatch) -> None:
    captured = {}

    class ConstructedChatModel(FakeChatModel):
        pass

    def make_chat_model(**kwargs):
        captured.update(kwargs)
        return ConstructedChatModel()

    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", make_chat_model)
    LangChainQwenBackend(model_name="qwen3.5-flash", enable_thinking=False)
    assert captured["extra_body"] == {"enable_thinking": False}
