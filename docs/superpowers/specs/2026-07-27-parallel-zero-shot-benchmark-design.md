# Parallel Zero-Shot Benchmark Design

## Goal

Evaluate the 2,500 test examples in `datasets/medium_dataset` with one direct LLM sentiment prompt per example, running five languages concurrently while preserving a single global request limit.

## Behavior

The benchmark loads 500 test examples for each of Indonesian, Khmer, Malay, Thai, and Vietnamese. It disables retrieval, translation, reflection, attribution, and learning. Each request uses `PredictionPromptBuilder` and the existing Pydantic `PredictionPayload` structured output.

One global semaphore bounds all requests across languages. A failed request is recorded without cancelling other samples. Each language directory receives `predictions.jsonl`, `failures.jsonl` when needed, `metrics.json`, and `costs.json`. The benchmark root receives `summary.json` with aggregate and per-language metrics, coverage, calls, failures, tokens, and elapsed time.

## Verification

Offline tests use a deterministic fake LLM to verify global concurrency, language isolation, failure handling, metrics, and files. The full offline test suite must pass. The real 2,500-request benchmark is run only through the explicit script command.

