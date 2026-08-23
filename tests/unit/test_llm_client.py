from collections.abc import Iterator

import httpx
import pytest

from sentiment_agent.config import ModelConfig
from sentiment_agent.llm.client import LLMRequestError, OpenAICompatibleClient


def response_sequence(*responses: httpx.Response):
    iterator: Iterator[httpx.Response] = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        response = next(iterator)
        response.request = request
        return response

    return handler


def completion_response(label: str = "neutral") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"label":"'
                            + label
                            + '","confidence":0.7,"reason":"fixture"}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        },
    )


def make_client(handler, *, max_attempts: int = 3) -> OpenAICompatibleClient:
    config = ModelConfig(
        name="fixture-model",
        base_url="https://example.invalid/v1",
        max_attempts=max_attempts,
    )
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient(config, api_key="test-key", http_client=http_client, sleep=lambda _: None)


def test_client_retries_rate_limit_then_returns_prediction() -> None:
    handler = response_sequence(httpx.Response(429, json={"error": "limited"}), completion_response())
    client = make_client(handler)

    result = client.classify("ข้อความ", "th")

    assert result.payload.label == "neutral"
    assert result.attempts == 2
    assert result.usage.total_tokens == 9


def test_client_does_not_retry_bad_request() -> None:
    client = make_client(response_sequence(httpx.Response(400, json={"error": "bad"})))

    with pytest.raises(LLMRequestError) as error:
        client.classify("text", "vi")

    assert error.value.attempts == 1


def test_embed_returns_vectors() -> None:
    response = httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})
    client = make_client(response_sequence(response))

    vectors = client.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
