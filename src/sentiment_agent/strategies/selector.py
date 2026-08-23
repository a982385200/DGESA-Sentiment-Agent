from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyStat:
    count: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.count == 0 else self.total_reward / self.count


class EpsilonGreedySelector:
    def __init__(self, strategies: list[str], *, epsilon: float) -> None:
        if not strategies or len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be a non-empty unique list")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.strategies = tuple(strategies)
        self.epsilon = epsilon
        self._stats: dict[str, dict[str, StrategyStat]] = {}

    def _language_stats(self, language: str) -> dict[str, StrategyStat]:
        return self._stats.setdefault(
            language,
            {strategy: StrategyStat() for strategy in self.strategies},
        )

    def select(self, language: str, rng: random.Random) -> str:
        if rng.random() < self.epsilon:
            return rng.choice(self.strategies)
        stats = self._language_stats(language)
        return max(self.strategies, key=lambda strategy: stats[strategy].mean_reward)

    def update(self, language: str, strategy: str, reward: float) -> None:
        if strategy not in self.strategies:
            raise ValueError(f"unknown strategy: {strategy}")
        stats = self._language_stats(language)
        current = stats[strategy]
        stats[strategy] = StrategyStat(
            count=current.count + 1,
            total_reward=current.total_reward + float(reward),
        )

    def stats(self, language: str) -> dict[str, StrategyStat]:
        return dict(self._language_stats(language))
