from __future__ import annotations

from dataclasses import dataclass, field

from sentiment_agent.evaluation.metrics import MetricsReport, compute_metrics
from sentiment_agent.schemas import SentimentLabel, Usage


@dataclass
class Evaluator:
    _gold: list[SentimentLabel] = field(default_factory=list)
    _predicted: list[SentimentLabel] = field(default_factory=list)

    def add(self, gold: SentimentLabel, predicted: SentimentLabel) -> None:
        self._gold.append(gold)
        self._predicted.append(predicted)

    def report(self) -> MetricsReport:
        return compute_metrics(self._gold, self._predicted)


@dataclass
class CostTracker:
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    total_latency_seconds: float = 0.0

    def record(self, usage: Usage, *, latency_seconds: float) -> None:
        if latency_seconds < 0:
            raise ValueError("latency_seconds must not be negative")
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.request_count += 1
        self.total_latency_seconds += latency_seconds

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost(self) -> float:
        return (
            self.prompt_tokens * self.input_price_per_million
            + self.completion_tokens * self.output_price_per_million
        ) / 1_000_000

    @property
    def average_latency_seconds(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_seconds / self.request_count

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "average_latency_seconds": self.average_latency_seconds,
            "estimated_cost": self.estimated_cost,
        }
