from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from sentiment_agent.data.loader import prediction_input
from sentiment_agent.data.stream import EvaluationStream, TrainingStream
from sentiment_agent.evaluation.artifacts import ArtifactWriter
from sentiment_agent.evaluation.evaluator import CostTracker, Evaluator
from sentiment_agent.evaluation.metrics import MetricsReport
from sentiment_agent.schemas import Prediction, PredictionInput, SentimentExample, StrictModel


class ExperimentAgent(Protocol):
    store: object

    def predict(self, item: PredictionInput) -> Prediction: ...

    def learn(self, item, prediction, feedback): ...


class EvolutionStage(StrictModel):
    processed_training_samples: int = Field(ge=0)
    experience_count: int = Field(ge=0)
    metrics: MetricsReport


class ExperimentSummary(StrictModel):
    experiment_type: str
    processed_training_samples: int = Field(ge=0)
    evaluation_samples: int = Field(ge=0)
    stages: list[EvolutionStage]
    costs: dict[str, float | int]


class ExperimentRunner:
    def __init__(
        self,
        agent: ExperimentAgent,
        writer: ArtifactWriter,
        *,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
    ) -> None:
        self.agent = agent
        self.writer = writer
        self.costs = CostTracker(input_price_per_million, output_price_per_million)

    def evaluate(
        self,
        examples: Sequence[SentimentExample],
        *,
        checkpoint: int,
    ) -> MetricsReport:
        evaluator = Evaluator()
        stream = EvaluationStream(examples)
        for example, item in zip(examples, stream, strict=True):
            prediction = self.agent.predict(item)
            evaluator.add(example.label, prediction.label)
            self.costs.record(prediction.usage, latency_seconds=prediction.latency_seconds)
            self.writer.append_prediction(
                {
                    **prediction.model_dump(mode="json"),
                    "gold_label": example.label,
                    "split": example.split,
                    "checkpoint": checkpoint,
                }
            )
        return evaluator.report()

    def run_evolution(
        self,
        training_examples: Sequence[SentimentExample],
        evaluation_examples: Sequence[SentimentExample],
        *,
        checkpoints: Sequence[int],
    ) -> ExperimentSummary:
        ordered_checkpoints = sorted(set(checkpoints))
        if not ordered_checkpoints:
            raise ValueError("at least one checkpoint is required")
        if ordered_checkpoints[0] < 0 or ordered_checkpoints[-1] > len(training_examples):
            raise ValueError("checkpoint is outside the training stream")

        stages: list[EvolutionStage] = []
        if 0 in ordered_checkpoints:
            stages.append(self._evaluate_stage(evaluation_examples, processed=0))

        training_stream = TrainingStream(training_examples)
        for processed, (example, item) in enumerate(
            zip(training_examples, training_stream, strict=True),
            start=1,
        ):
            prediction = self.agent.predict(item)
            feedback = training_stream.feedback(
                item,
                predicted=prediction.label,
                sample_id=prediction.sample_id,
            )
            self.agent.learn(item, prediction, feedback)
            self.costs.record(prediction.usage, latency_seconds=prediction.latency_seconds)
            self.writer.append_prediction(
                {
                    **prediction.model_dump(mode="json"),
                    "gold_label": example.label,
                    "split": "train",
                    "checkpoint": processed,
                }
            )
            if processed in ordered_checkpoints:
                stages.append(self._evaluate_stage(evaluation_examples, processed=processed))

        summary = ExperimentSummary(
            experiment_type="evolution",
            processed_training_samples=len(training_examples),
            evaluation_samples=len(evaluation_examples),
            stages=stages,
            costs=self.costs.as_dict(),
        )
        self.writer.write_json("metrics.json", summary)
        self.writer.write_json("costs.json", self.costs.as_dict())
        return summary

    def _evaluate_stage(
        self,
        evaluation_examples: Sequence[SentimentExample],
        *,
        processed: int,
    ) -> EvolutionStage:
        report = self.evaluate(evaluation_examples, checkpoint=processed)
        count_method = getattr(self.agent.store, "count")
        return EvolutionStage(
            processed_training_samples=processed,
            experience_count=int(count_method()),
            metrics=report,
        )
