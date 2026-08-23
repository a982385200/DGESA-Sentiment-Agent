# Pydantic Structured Model Output Design

## Goal

Use LangChain `with_structured_output` with Pydantic models for every LLM response so application code never parses model-generated JSON manually.

## Scope

The change covers sentiment prediction, translation, and error attribution. It does not change JSON files, experiment artifacts, configuration parsing, repositories, or other non-LLM serialization.

## Architecture

`LangChainQwenBackend` binds separate structured-output runnables for `PredictionPayload` and `TranslationPayload`. Classification and translation invoke those runnables and wrap the validated Pydantic values in the existing result envelopes, preserving usage accounting.

Attribution binds an `AttributionPayload` structured-output runnable with `include_raw=True`. The attribution client returns a typed result containing the parsed payload, raw response text, and any parsing error. `LLMAttributor` retains its configured retry count and deterministic fallback, but no longer extracts JSON with regular expressions.

## Failure Semantics

Prediction and translation propagate invocation or Pydantic validation failures. Attribution retries any invocation or structured parsing failure; after all attempts fail it creates the existing deterministic fallback. Raw attribution responses remain available for auditing when supplied by LangChain.

## Testing

Unit tests use local fakes that implement `with_structured_output` and `ainvoke`, verifying application-visible typed results, usage preservation, retries, and fallback. Tests must fail against the current manual-parsing implementation before production changes. The complete unit suite and both offline integration workflows must pass without API credentials.

