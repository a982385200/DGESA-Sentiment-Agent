from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from threading import RLock

from sentiment_agent.schemas import Experience


def _reliability(success_count: int, failure_count: int) -> float:
    return (success_count + 1) / (success_count + failure_count + 2)


def _dedup_key(experience: Experience) -> str:
    parts = (
        " ".join(experience.text.casefold().split()),
        experience.language,
        experience.source.casefold(),
        experience.sentiment,
        experience.experience_type,
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class ExperienceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    dedup_key TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def add_or_update(self, experience: Experience, vector: list[float]) -> str:
        if not vector:
            raise ValueError("experience vector must not be empty")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("experience vector must contain numbers")
        key = _dedup_key(experience)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id, payload_json FROM experiences WHERE dedup_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                experience_id = experience.id or uuid.uuid4().hex
                stored = experience.model_copy(
                    update={
                        "id": experience_id,
                        "reliability": _reliability(experience.success_count, experience.failure_count),
                    }
                )
                connection.execute(
                    "INSERT INTO experiences(id, dedup_key, payload_json, vector_json) VALUES (?, ?, ?, ?)",
                    (
                        experience_id,
                        key,
                        stored.model_dump_json(),
                        json.dumps([float(value) for value in vector]),
                    ),
                )
                return experience_id

            experience_id, payload_json = row
            current = Experience.model_validate_json(payload_json)
            success_count = current.success_count + 1
            updated = current.model_copy(
                update={
                    "success_count": success_count,
                    "reliability": _reliability(success_count, current.failure_count),
                }
            )
            connection.execute(
                """
                UPDATE experiences
                SET payload_json = ?, vector_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    updated.model_dump_json(),
                    json.dumps([float(value) for value in vector]),
                    experience_id,
                ),
            )
            return str(experience_id)

    def record_outcome(self, experience_id: str, *, correct: bool) -> Experience:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM experiences WHERE id = ?",
                (experience_id,),
            ).fetchone()
            if row is None:
                raise KeyError(experience_id)
            current = Experience.model_validate_json(row[0])
            success_count = current.success_count + int(correct)
            failure_count = current.failure_count + int(not correct)
            updated = current.model_copy(
                update={
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "reliability": _reliability(success_count, failure_count),
                }
            )
            connection.execute(
                "UPDATE experiences SET payload_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (updated.model_dump_json(), experience_id),
            )
            return updated

    def get(self, experience_id: str) -> Experience:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM experiences WHERE id = ?",
                (experience_id,),
            ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        return Experience.model_validate_json(row[0])

    def get_with_vector(self, experience_id: str) -> tuple[Experience, list[float]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, vector_json FROM experiences WHERE id = ?",
                (experience_id,),
            ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        return Experience.model_validate_json(row[0]), [float(value) for value in json.loads(row[1])]

    def all_with_vectors(self) -> list[tuple[Experience, list[float]]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json, vector_json FROM experiences ORDER BY id"
            ).fetchall()
        return [
            (Experience.model_validate_json(payload), [float(value) for value in json.loads(vector)])
            for payload, vector in rows
        ]

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM experiences").fetchone()
        return int(row[0])
