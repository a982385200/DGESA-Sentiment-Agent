from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from sentiment_agent.schemas import PredictionInput, SentimentExample


def load_examples(path: Path) -> list[SentimentExample]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        decoded = json.loads(text)
        rows = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    examples: list[SentimentExample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        try:
            example = SentimentExample.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"invalid sample at {path}:{index}: {exc}") from exc
        if example.id in seen:
            raise ValueError(f"duplicate sample id: {example.id}")
        seen.add(example.id)
        examples.append(example)
    return examples


def without_labels(examples: Sequence[SentimentExample]) -> list[PredictionInput]:
    return [example.to_prediction_input() for example in examples]
