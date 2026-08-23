from __future__ import annotations

from collections.abc import Protocol, Sequence

import numpy as np


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
