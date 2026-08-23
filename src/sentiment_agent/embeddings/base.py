from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
