from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ProgressEvent:
    stage: Literal["train", "dev", "test"]
    completed_samples: int
    total_samples: int
    completed_batches: int
    total_batches: int
    successful_requests: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    experience_count: int
    checkpoint: int | None = None


class ProgressReporter(Protocol):
    def update(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...


class NullProgressReporter:
    def update(self, event: ProgressEvent) -> None:
        return None

    def close(self) -> None:
        return None


class RecordingProgressReporter:
    """In-memory reporter useful for programmatic experiment observers."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def update(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None
