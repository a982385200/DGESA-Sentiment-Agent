from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("vector dimension mismatch")
    if not first:
        raise ValueError("vectors must not be empty")
    dot_product = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot_product / (first_norm * second_norm)
