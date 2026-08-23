from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class LocalBGEEmbedding:
    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 32,
        encoder: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._encoder = encoder

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_id, device=self.device)
        return self._encoder

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if any(not text.strip() for text in values):
            raise ValueError("embedding text must not be empty")
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        raw = self._load_encoder().encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(values):
            raise ValueError("encoder returned an invalid shape")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("encoder returned a zero vector")
        return vectors / norms
