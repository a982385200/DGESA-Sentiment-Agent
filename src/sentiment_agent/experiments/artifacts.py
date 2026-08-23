from __future__ import annotations

import json
import os
from pathlib import Path


class ArtifactWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return target

    def append_jsonl(self, relative_path: str, payload) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return target
