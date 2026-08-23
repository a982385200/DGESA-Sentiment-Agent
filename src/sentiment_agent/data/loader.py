from __future__ import annotations

import json
from pathlib import Path

from sentiment_agent.schemas import PredictionInput, SentimentExample, Split


def load_jsonl(path: Path, split: Split) -> list[SentimentExample]:
    examples: list[SentimentExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raw["split"] = split
                examples.append(SentimentExample.model_validate(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid dataset row at {path}:{line_number}: {exc}") from exc
    return examples


def prediction_input(example: SentimentExample) -> PredictionInput:
    return PredictionInput(
        id=example.id,
        text=example.text,
        language=example.language,
        source=example.source,
    )
