# Mini-Batch Evolution Design

## Goal

Add deterministic mini-batch learning so a batch of samples can share one experience snapshot, batch local embeddings, and run LLM predictions concurrently before feedback is applied in original sample order.

## Semantics

For each batch, the agent embeds all inputs together, retrieves from the experience store before any sample in that batch is learned, selects strategies deterministically in input order, and invokes chat requests concurrently. Only after every prediction succeeds does the runner create feedback and call `learn` sequentially in input order. Experiences produced by a batch become visible to the next batch.

`train_batch_size=1` exactly preserves the current online-learning path. Checkpoints are hard boundaries: a configured checkpoint splits a batch so evaluation occurs at the exact processed count.

## Configuration

`ExperimentConfig` adds positive `train_batch_size` and `llm_concurrency` fields, both defaulting to 1 for backward compatibility. The Python launcher exposes `--train-batch-size` and `--llm-concurrency`; saved expanded configs and hashes include both values.

## Agent and Runner

`SentimentAgent.predict_batch(items, max_concurrency)` performs one local embedding call, prepares retrieval/prompts sequentially, then uses a bounded thread pool for chat calls. Predictions and contexts are assembled in input order regardless of completion order. `predict` delegates to a one-item batch.

`ExperimentRunner` partitions training data without crossing checkpoints, calls `predict_batch` when supported, then performs ordered learning. Evaluation uses the same batch prediction API but never learns. Protocol-compatible test agents without `predict_batch` retain sequential fallback behavior.

## Failure and Logging

If any prediction fails, the batch performs no learning and the original exception propagates. Existing LangChain retry handling remains active. Logs record batch number, input range, size, concurrency, prediction duration, ordered learning completion, and checkpoint metrics; they never contain raw samples or secrets.

## Testing

Tests first prove batch embedding, bounded parallel chat, stable result order, delayed experience visibility, exact checkpoint splitting, batch failure atomicity, `batch_size=1` compatibility, parameter mapping, and log events. The full offline suite and a small real Qwen timing run verify the result.
