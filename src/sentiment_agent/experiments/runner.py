from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.evaluation.metrics import classification_metrics
from sentiment_agent.experiments.artifacts import ArtifactWriter
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
                 batch_size: int, concurrency: int, checkpoints: Sequence[int] = ()) -> None:
        self.agent = agent
        self.writer = writer
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.checkpoints = tuple(sorted(set(checkpoints)))

    async def evaluate(self, examples: Sequence[SentimentExample], *, split: str,
                       checkpoint: int | None = None) -> dict:
        predictions = []
        for batch in partition_batches(list(examples), self.batch_size, []):
            predictions.extend(await self.agent.predict_batch(
                [example.to_prediction_input() for example in batch], max_concurrency=self.concurrency))
        metrics = classification_metrics([item.label for item in examples], [item.label for item in predictions])
        self.writer.write_json(f"metrics-{split}-{checkpoint or 'final'}.json", metrics)
        return metrics

    async def run(self, train: Sequence[SentimentExample], dev: Sequence[SentimentExample],
                  test: Sequence[SentimentExample]) -> RunSummary:
        processed = 0
        reached = []
        for batch_id, batch in enumerate(partition_batches(list(train), self.batch_size, self.checkpoints), start=1):
            items = [example.to_prediction_input() for example in batch]
            predictions = await self.agent.predict_batch(items, max_concurrency=self.concurrency)
            feedback = [Feedback(sample_id=prediction.sample_id, predicted_label=prediction.label,
                                 gold_label=example.label, correct=prediction.label == example.label)
                        for example, prediction in zip(batch, predictions, strict=True)]
            self.agent.learn_batch(items, predictions, feedback, batch_id=batch_id)
            processed += len(batch)
            for example, prediction in zip(batch, predictions, strict=True):
                self.writer.append_jsonl("predictions.jsonl", {
                    "split": "train", "batch_id": batch_id, "sample_id": example.id,
                    "gold_label": example.label, **prediction.model_dump(mode="json")})
            if processed in self.checkpoints:
                reached.append(processed)
                await self.evaluate(dev, split="dev", checkpoint=processed)
        metrics = await self.evaluate(test, split="test")
        self.writer.write_json("metrics.json", metrics)
        return RunSummary(self.writer.run_dir, processed, tuple(reached), metrics)
