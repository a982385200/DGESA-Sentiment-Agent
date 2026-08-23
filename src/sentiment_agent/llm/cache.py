from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY,response TEXT NOT NULL)")
        self.connection.commit()

    def key(self, model: str, parameters: dict, messages: list[str]) -> str:
        raw = json.dumps({"model": model, "parameters": parameters, "messages": messages},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def put(self, key: str, response: str) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO responses(key,response) VALUES(?,?)", (key, response))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
