from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSection(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SENTIMENT_AGENT_", env_nested_delimiter="__")


class ModelConfig(ConfigSection):
    name: str = "test-model"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    embedding_model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, gt=0)


class MemoryConfig(ConfigSection):
    database_path: Path = Path("outputs/experience.sqlite3")
    retrieval_k: int = Field(default=5, ge=0)
    min_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_weight: float = 0.6
    language_weight: float = 0.1
    domain_weight: float = 0.1
    reliability_weight: float = 0.15
    recency_weight: float = 0.05


class StrategyConfig(ConfigSection):
    names: list[str] = Field(default_factory=lambda: ["direct", "translation", "memory", "reflection_verified"])
    selector: Literal["fixed", "random", "epsilon_greedy"] = "epsilon_greedy"
    fixed_strategy: str = "direct"
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)


class ExperimentConfig(ConfigSection):
    name: str = "baseline"
    train_paths: list[Path] = Field(default_factory=list)
    dev_paths: list[Path] = Field(default_factory=list)
    test_paths: list[Path] = Field(default_factory=list)
    checkpoints: list[int] = Field(default_factory=lambda: [0])
    reflection_enabled: bool = True
    cross_lingual_enabled: bool = True
    max_failure_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    output_root: Path = Path("outputs")


class AppConfig(ConfigSection):
    model_config = SettingsConfigDict(extra="forbid", env_prefix="SENTIMENT_AGENT_", env_nested_delimiter="__")

    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @computed_field
    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"config_hash"})
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.model_validate(raw)
