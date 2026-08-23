from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from sentiment_agent.experiments.progress import ProgressEvent


class RichProgressReporter:
    def __init__(self, *, console: Console | None = None) -> None:
        self._progress = Progress(
            TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(), console=console,
            transient=False, refresh_per_second=8,
        )
        self._task_id: int | None = None
        self._stage: str | None = None
        self._started = False

    def update(self, event: ProgressEvent) -> None:
        if not self._started:
            self._progress.start()
            self._started = True
        description = (
            f"{event.stage} {event.completed_samples}/{event.total_samples} | "
            f"batch {event.completed_batches}/{event.total_batches} | "
            f"ok {event.successful_requests} failed {event.failed_requests} | "
            f"tokens {event.input_tokens + event.output_tokens} | "
            f"experiences {event.experience_count}"
        )
        if self._task_id is None or self._stage != event.stage:
            if self._task_id is not None:
                self._progress.remove_task(self._task_id)
            self._task_id = self._progress.add_task(
                description, total=max(event.total_samples, 1),
                completed=event.completed_samples,
            )
            self._stage = event.stage
        else:
            self._progress.update(
                self._task_id, description=description,
                completed=event.completed_samples, total=max(event.total_samples, 1),
                refresh=True,
            )

    def close(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False
