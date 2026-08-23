# LangChain LLM Access Design

## Goal

Replace the custom `httpx` OpenAI-compatible transport with LangChain while preserving the application's existing LLM-facing interfaces and reproducible offline workflow.

## Scope

The change continues to target OpenAI-compatible chat and embedding endpoints. It does not add provider switching or alter the sentiment agent, reflection, experiment, or memory APIs.

## Architecture

`OpenAICompatibleClient` remains the application boundary used by the rest of the project. Internally it delegates chat requests to `langchain_openai.ChatOpenAI` and embedding requests to `langchain_openai.OpenAIEmbeddings`. Constructor injection allows tests to supply deterministic fake LangChain models without credentials or network access.

The public methods `classify`, `chat_json`, and `embed`, plus `LLMResult` and `LLMRequestError`, remain available. Existing `ModelConfig` fields supply model name, base URL, API key, timeout, retry count, temperature, token limit, and embedding model.

## Data Flow

For chat calls, the client constructs the same role/content message sequence and deterministic cache payload used today. A cache hit is parsed without calling LangChain. On a miss, LangChain returns an AI message; its content is validated through the existing Pydantic parsing path, usage metadata is normalized into `Usage`, and a stable response representation is stored in SQLite.

For embeddings, the client sends the input sequence through the injected or configured LangChain embedding model and normalizes returned values to `list[list[float]]`.

## Errors and Retries

LangChain owns transport retries according to `ModelConfig.max_attempts`. Exceptions crossing the application boundary are wrapped as `LLMRequestError` with the original exception as the cause. Application-level response validation remains local so malformed content receives the project's existing error type.

## Dependencies and Configuration

Add `langchain-openai` as a runtime dependency and update `uv.lock` using `uv`. Remove the direct runtime dependency on `httpx` if no production module still imports it. Existing YAML and environment variable names remain valid.

## Testing and Documentation

Tests inject fake chat and embedding objects, assert message/config mapping, structured parsing, usage normalization, cache behavior, embeddings, and error conversion without external API calls. Every new public function or method receives an isolated unit test. The complete unit suite and offline integration workflow must pass through `uv run pytest`. README and CLI wording will describe LangChain-backed OpenAI-compatible access.
