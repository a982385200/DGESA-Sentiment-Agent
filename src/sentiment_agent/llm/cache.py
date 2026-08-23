from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


def build_cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, cache_key: str, response: dict[str, Any]) -> None:
        serialized = json.dumps(response, ensure_ascii=False, sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses(cache_key, response_json) VALUES (?, ?)",
                (cache_key, serialized),
            )
