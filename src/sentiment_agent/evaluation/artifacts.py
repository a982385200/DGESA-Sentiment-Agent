from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value

    def append_prediction(self, record: BaseModel | dict[str, Any]) -> None:
        line = json.dumps(self._jsonable(record), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, (self.output_dir / "predictions.jsonl").open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)

    def write_json(self, filename: str, value: BaseModel | dict[str, Any]) -> Path:
        destination = self.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        serialized = json.dumps(self._jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with self._lock:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, destination)
        return destination
