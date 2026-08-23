from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from sentiment_agent.llm.cache import ResponseCache
from sentiment_agent.llm.langchain_qwen import LangChainQwenBackend
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import PredictionInput


class FakeChatModel:
    async def ainvoke(self, messages):
        return AIMessage(content='```json\n{"label":"negative","confidence":0.9,"reason":"negation"}\n```',
                         usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def test_base_prompt_is_stable() -> None:
    builder = PredictionPromptBuilder()
    item = PredictionInput(id="x", text="not good", language="vi", source="tiny")
    assert builder.build(item, [])[0].content == builder.build(item, [])[0].content


@pytest.mark.anyio
async def test_qwen_backend_parses_fenced_json() -> None:
    backend = LangChainQwenBackend(model_name="qwen", chat_model=FakeChatModel())
    result = await backend.classify(PredictionPromptBuilder().build(
        PredictionInput(id="x", text="not good", language="vi", source="tiny"), []))
    assert result.payload.label == "negative"
    assert result.usage.input_tokens == 10


def test_response_cache_round_trip(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "cache.sqlite3") as cache:
        key = cache.key("qwen", {"temperature": 0}, ["a"])
        cache.put(key, '{"ok":true}')
        assert cache.get(key) == '{"ok":true}'
        assert key != cache.key("qwen", {"temperature": 0}, ["b"])
