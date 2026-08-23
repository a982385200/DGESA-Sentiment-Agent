from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.evaluation.metrics import classification_metrics
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.experiments.progress import NullProgressReporter, ProgressEvent, ProgressReporter
from sentiment_agent.schemas import Feedback, SentimentExample


def partition_batches(items: Sequence, batch_size: int, checkpoints: Sequence[int]) -> list[list]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    boundaries = {value for value in checkpoints if 0 < value <= len(items)} | {len(items)}
    batches = []
    start = 0
    while start < len(items):
        next_checkpoint = min(value for value in boundaries if value > start)
        end = min(start + batch_size, next_checkpoint)
        batches.append(list(items[start:end]))
        start = end
    return batches


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    completed_samples: int
    checkpoints: tuple[int, ...]
    metrics: dict


class ExperimentRunner:
    def __init__(self, *, agent: SentimentAgent, writer: ArtifactWriter,
                 batch_size: int, concurrency: int, checkpoints: Sequence[int] = (),
                 manifest_metadata: dict | None = None,
                 progress_reporter: ProgressReporter | None = None) -> None:
        self.agent = agent
        self.writer = writer
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.checkpoints = tuple(sorted(set(checkpoints)))
        self.manifest_metadata = dict(manifest_metadata or {})
        self.progress_reporter = progress_reporter or NullProgressReporter()
        self._started_monotonic = time.monotonic()
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def _record_usage(self, predictions) -> None:
        self._calls += len(predictions)
        self._input_tokens += sum(item.usage.input_tokens for item in predictions)
        self._output_tokens += sum(item.usage.output_tokens for item in predictions)

    def _report(self, *, stage: str, completed: int, total: int,
                completed_batches: int, total_batches: int,
                checkpoint: int | None = None) -> None:
        try:
            self.progress_reporter.update(ProgressEvent(
                stage=stage, completed_samples=completed, total_samples=total,
                completed_batches=completed_batches, total_batches=total_batches,
                successful_requests=self._calls, failed_requests=0,
                input_tokens=self._input_tokens, output_tokens=self._output_tokens,
                elapsed_seconds=time.monotonic() - self._started_monotonic,
                experience_count=self.agent.updater.repository.count(), checkpoint=checkpoint,
            ))
        except Exception:
            pass

    async def evaluate(self, examples: Sequence[SentimentExample], *, split: str,
                       checkpoint: int | None = None) -> dict:
        predictions = []
        batches = partition_batches(list(examples), self.batch_size, [])
        completed = 0
        for batch_number, batch in enumerate(batches, start=1):
            current = await self.agent.predict_batch(
                [example.to_prediction_input() for example in batch], max_concurrency=self.concurrency)
            predictions.extend(current)
            self._record_usage(current)
            completed += len(batch)
            self._report(stage=split, completed=completed, total=len(examples),
                         completed_batches=batch_number, total_batches=len(batches),
                         checkpoint=checkpoint)
        metrics = classification_metrics([item.label for item in examples], [item.label for item in predictions])
        self.writer.write_json(f"metrics-{split}-{checkpoint or 'final'}.json", metrics)
        return metrics

    async def run(self, train: Sequence[SentimentExample], dev: Sequence[SentimentExample],
                  test: Sequence[SentimentExample]) -> RunSummary:
        started_at = datetime.now(UTC).isoformat()
        self.writer.write_json("manifest.json", {
            **self.manifest_metadata, "status": "running", "started_at": started_at})
        processed = 0
        reached = []
        for batch_id, batch in enumerate(partition_batches(list(train), self.batch_size, self.checkpoints), start=1):
            items = [example.to_prediction_input() for example in batch]
            predictions = await self.agent.predict_batch(items, max_concurrency=self.concurrency)
            self._record_usage(predictions)
            feedback = [Feedback(sample_id=prediction.sample_id, predicted_label=prediction.label,
                                 gold_label=example.label, correct=prediction.label == example.label)
                        for example, prediction in zip(batch, predictions, strict=True)]
            self.agent.learn_batch(items, predictions, feedback, batch_id=batch_id)
            processed += len(batch)
            training_batches = partition_batches(list(train), self.batch_size, self.checkpoints)
            self._report(stage="train", completed=processed, total=len(train),
                         completed_batches=batch_id, total_batches=len(training_batches),
                         checkpoint=processed if processed in self.checkpoints else None)
            for example, prediction in zip(batch, predictions, strict=True):
                self.writer.append_jsonl("predictions.jsonl", {
                    "split": "train", "batch_id": batch_id, "sample_id": example.id,
                    "gold_label": example.label, **prediction.model_dump(mode="json")})
            if processed in self.checkpoints:
                reached.append(processed)
                await self.evaluate(dev, split="dev", checkpoint=processed)
        metrics = await self.evaluate(test, split="test")
        self.writer.write_json("metrics.json", metrics)
        self.writer.write_json("costs.json", {
            "calls": self._calls, "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        })
        self.writer.write_json("manifest.json", {
            **self.manifest_metadata,
            "status": "completed", "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "completed_samples": processed, "checkpoints": reached,
        })
        return RunSummary(self.writer.run_dir, processed, tuple(reached), metrics)
