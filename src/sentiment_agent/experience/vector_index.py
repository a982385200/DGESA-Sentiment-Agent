from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorSnapshot:
    ids: tuple[str, ...]
    vectors: np.ndarray


class VectorIndex:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors_path = directory / "vectors.npy"
        self.ids_path = directory / "vector_ids.json"

    def snapshot(self) -> VectorSnapshot:
        if not self.vectors_path.exists():
            return VectorSnapshot((), np.empty((0, 0), dtype=np.float32))
        ids = tuple(json.loads(self.ids_path.read_text(encoding="utf-8")))
        vectors = np.load(self.vectors_path, allow_pickle=False).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("vector index is inconsistent")
        return VectorSnapshot(ids, vectors.copy())

    def upsert(self, experience_id: str, vector: np.ndarray) -> None:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise ValueError("experience vector must not be zero")
        value = value / norm
        current = self.snapshot()
        ids = list(current.ids)
        if ids and current.vectors.shape[1] != value.shape[0]:
            raise ValueError("vector dimension changed")
        if experience_id in ids:
            vectors = current.vectors.copy()
            vectors[ids.index(experience_id)] = value
        else:
            ids.append(experience_id)
            vectors = value.reshape(1, -1) if not current.ids else np.vstack((current.vectors, value))
        vectors_tmp = self.vectors_path.with_suffix(".npy.tmp")
        ids_tmp = self.ids_path.with_suffix(".json.tmp")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32), allow_pickle=False)
        ids_tmp.write_text(json.dumps(ids), encoding="utf-8")
        os.replace(vectors_tmp, self.vectors_path)
        os.replace(ids_tmp, self.ids_path)
