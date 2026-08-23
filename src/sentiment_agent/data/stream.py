from __future__ import annotations

from collections.abc import Iterator, Sequence

from sentiment_agent.data.loader import prediction_input
from sentiment_agent.schemas import Feedback, PredictionInput, SentimentExample, SentimentLabel


class EvaluationStream:
    def __init__(self, examples: Sequence[SentimentExample]) -> None:
        self._inputs = tuple(prediction_input(example) for example in examples)

    def __iter__(self) -> Iterator[PredictionInput]:
        return iter(self._inputs)

    def __len__(self) -> int:
        return len(self._inputs)


class TrainingStream:
    def __init__(self, examples: Sequence[SentimentExample]) -> None:
        self._inputs = tuple(prediction_input(example) for example in examples)
        self._gold = {example.id: example.label for example in examples}

    def __iter__(self) -> Iterator[PredictionInput]:
        return iter(self._inputs)

    def __len__(self) -> int:
        return len(self._inputs)

    def feedback(
        self,
        item: PredictionInput,
        *,
        predicted: SentimentLabel,
        sample_id: str,
    ) -> Feedback:
        if sample_id != item.id:
            raise ValueError(f"sample id mismatch: expected {item.id}, received {sample_id}")
        try:
            gold = self._gold[item.id]
        except KeyError as exc:
            raise ValueError(f"sample id is not part of this training stream: {item.id}") from exc
        return Feedback(sample_id=item.id, predicted_label=predicted, gold_label=gold)
