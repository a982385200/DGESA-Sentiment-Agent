from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sentiment_agent.schemas import Experience, ExperienceEvent


class ExperienceRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata(version INTEGER NOT NULL);
            INSERT INTO schema_metadata(version)
            SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_metadata);
            CREATE TABLE IF NOT EXISTS experiences(
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                dedup_key TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS experience_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id TEXT NOT NULL REFERENCES experiences(id),
                batch_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                old_value_json TEXT,
                new_value_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS experience_outcomes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id TEXT NOT NULL REFERENCES experiences(id),
                sample_id TEXT NOT NULL,
                batch_id INTEGER NOT NULL,
                retrieval_rank INTEGER NOT NULL,
                score REAL NOT NULL,
                injected INTEGER NOT NULL,
                prediction_correct INTEGER NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def dedup_key(experience: Experience) -> str:
        return "\x1f".join(
            (" ".join(experience.text.casefold().split()), experience.language,
             experience.source.casefold(), experience.sentiment, experience.type)
        )

    def create(self, experience: Experience, *, batch_id: int) -> None:
        payload = experience.model_dump_json()
        with self.connection:
            self.connection.execute(
                "INSERT INTO experiences(id,payload_json,dedup_key) VALUES(?,?,?)",
                (experience.id, payload, self.dedup_key(experience)),
            )
            self._insert_event(experience.id, batch_id, "created", None, experience, "new feedback")

    def get(self, experience_id: str) -> Experience:
        row = self.connection.execute(
            "SELECT payload_json FROM experiences WHERE id=?", (experience_id,)
        ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        return Experience.model_validate_json(row[0])

    def find_by_dedup_key(self, key: str) -> Experience | None:
        row = self.connection.execute(
            "SELECT payload_json FROM experiences WHERE dedup_key=?", (key,)
        ).fetchone()
        return None if row is None else Experience.model_validate_json(row[0])

    def list(self) -> list[Experience]:
        rows = self.connection.execute("SELECT payload_json FROM experiences ORDER BY id").fetchall()
        return [Experience.model_validate_json(row[0]) for row in rows]

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def merge_counts(
        self, experience_id: str, *, success_delta: int, failure_delta: int, batch_id: int
    ) -> Experience:
        if success_delta < 0 or failure_delta < 0:
            raise ValueError("count deltas must be nonnegative")
        current = self.get(experience_id)
        success = current.success_count + success_delta
        failure = current.failure_count + failure_delta
        updated = current.model_copy(update={
            "success_count": success,
            "failure_count": failure,
            "reliability": (success + 1) / (success + failure + 2),
            "last_used_batch": batch_id,
        })
        event_type = "reinforced" if success_delta else "penalized"
        with self.connection:
            self.connection.execute(
                "UPDATE experiences SET payload_json=? WHERE id=?",
                (updated.model_dump_json(), experience_id),
            )
            self._insert_event(experience_id, batch_id, event_type, current, updated, "prediction outcome")
        return updated

    def _insert_event(self, experience_id, batch_id, event_type, old, new, reason) -> None:
        self.connection.execute(
            "INSERT INTO experience_events(experience_id,batch_id,event_type,old_value_json,new_value_json,reason) VALUES(?,?,?,?,?,?)",
            (experience_id, batch_id, event_type,
             None if old is None else old.model_dump_json(), new.model_dump_json(), reason),
        )

    def history(self, experience_id: str) -> list[ExperienceEvent]:
        rows = self.connection.execute(
            "SELECT id,batch_id,event_type,old_value_json,new_value_json,reason,created_at FROM experience_events WHERE experience_id=? ORDER BY id",
            (experience_id,),
        ).fetchall()
        return [ExperienceEvent(
            id=row[0], experience_id=experience_id, batch_id=row[1], event_type=row[2],
            old_value=None if row[3] is None else json.loads(row[3]),
            new_value=json.loads(row[4]), reason=row[5], created_at=row[6],
        ) for row in rows]

    def record_outcome(self, *, experience_id: str, sample_id: str, batch_id: int,
                       retrieval_rank: int, score: float, injected: bool,
                       prediction_correct: bool) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO experience_outcomes(experience_id,sample_id,batch_id,retrieval_rank,score,injected,prediction_correct) VALUES(?,?,?,?,?,?,?)",
                (experience_id, sample_id, batch_id, retrieval_rank, score, injected, prediction_correct),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExperienceRepository:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
