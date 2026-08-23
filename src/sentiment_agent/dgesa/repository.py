from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from sentiment_agent.dgesa.models import PatternExperience, SampleExperience


class DGESARepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sample_experiences(
                id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, vector_json TEXT NOT NULL,
                experience_vector_json TEXT NOT NULL DEFAULT '[]');
            CREATE TABLE IF NOT EXISTS pattern_experiences(
                id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, vector_json TEXT NOT NULL,
                evidence_vectors_json TEXT NOT NULL DEFAULT '[]');
            """
        )
        self.connection.commit()

    def save_sample(self, experience: SampleExperience, vector: np.ndarray, *,
                    experience_vector: np.ndarray | None = None) -> None:
        local_vector = vector if experience_vector is None else experience_vector
        self.connection.execute(
            "INSERT OR REPLACE INTO sample_experiences"
            "(id,payload_json,vector_json,experience_vector_json) VALUES(?,?,?,?)",
            (experience.id, experience.model_dump_json(),
             json.dumps(np.asarray(vector, dtype=np.float32).reshape(-1).tolist()),
             json.dumps(np.asarray(local_vector, dtype=np.float32).reshape(-1).tolist())),
        )
        self.connection.commit()

    def save_pattern(self, experience: PatternExperience, vector: np.ndarray, *,
                     evidence_vectors: list[np.ndarray] | None = None) -> None:
        payload = experience.model_dump_json(exclude={"reliability", "conflict_ratio"})
        vectors = evidence_vectors
        if vectors is None:
            row = self.connection.execute(
                "SELECT evidence_vectors_json FROM pattern_experiences WHERE id=?",
                (experience.id,),
            ).fetchone()
            vectors = [_vector(value) for value in json.loads(row[0])] if row else [vector]
        encoded = json.dumps([
            np.asarray(value, dtype=np.float32).reshape(-1).tolist() for value in vectors
        ])
        self.connection.execute(
            "INSERT OR REPLACE INTO pattern_experiences"
            "(id,payload_json,vector_json,evidence_vectors_json) VALUES(?,?,?,?)",
            (experience.id, payload, json.dumps(
                np.asarray(vector, dtype=np.float32).reshape(-1).tolist()), encoded),
        )
        self.connection.commit()

    def get_sample(self, experience_id: str) -> SampleExperience:
        return SampleExperience.model_validate_json(
            self._payload("sample_experiences", experience_id))

    def get_pattern(self, experience_id: str) -> PatternExperience:
        return PatternExperience.model_validate_json(
            self._payload("pattern_experiences", experience_id))

    def list_samples(self) -> list[SampleExperience]:
        return [SampleExperience.model_validate_json(row[0]) for row in
                self.connection.execute("SELECT payload_json FROM sample_experiences ORDER BY id")]

    def list_patterns(self) -> list[PatternExperience]:
        return [PatternExperience.model_validate_json(row[0]) for row in
                self.connection.execute("SELECT payload_json FROM pattern_experiences ORDER BY id")]

    def sample_vectors(self) -> list[tuple[SampleExperience, np.ndarray]]:
        return [(SampleExperience.model_validate_json(payload), _vector(vector))
                for payload, vector in self.connection.execute(
                    "SELECT payload_json, vector_json FROM sample_experiences ORDER BY id")]

    def sample_records(self) -> list[tuple[SampleExperience, np.ndarray, np.ndarray]]:
        return [(SampleExperience.model_validate_json(payload), _vector(retrieval),
                 _vector(experience))
                for payload, retrieval, experience in self.connection.execute(
                    "SELECT payload_json, vector_json, experience_vector_json "
                    "FROM sample_experiences ORDER BY id")]

    def pattern_vectors(self) -> list[tuple[PatternExperience, np.ndarray]]:
        return [(PatternExperience.model_validate_json(payload), _vector(vector))
                for payload, vector in self.connection.execute(
                    "SELECT payload_json, vector_json FROM pattern_experiences ORDER BY id")]

    def pattern_records(self) -> list[tuple[PatternExperience, np.ndarray, list[np.ndarray]]]:
        return [(PatternExperience.model_validate_json(payload), _vector(vector),
                 [_vector(value) for value in json.loads(evidence_vectors)])
                for payload, vector, evidence_vectors in self.connection.execute(
                    "SELECT payload_json, vector_json, evidence_vectors_json "
                    "FROM pattern_experiences ORDER BY id")]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> DGESARepository:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _save(self, table: str, experience_id: str, payload: str,
              vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()
        self.connection.execute(
            f"INSERT OR REPLACE INTO {table}(id,payload_json,vector_json) VALUES(?,?,?)",
            (experience_id, payload, json.dumps(values)),
        )
        self.connection.commit()

    def _payload(self, table: str, experience_id: str) -> str:
        row = self.connection.execute(
            f"SELECT payload_json FROM {table} WHERE id=?", (experience_id,)
        ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        return row[0]


def _vector(payload) -> np.ndarray:
    value = json.loads(payload) if isinstance(payload, str) else payload
    return np.asarray(value, dtype=np.float32)
