from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import Field, HttpUrl, PositiveInt

from sentiment_agent.schemas import StrictModel


class ModelConfig(StrictModel):
    name: str
    base_url: HttpUrl
    api_key_env: str
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: PositiveInt = 256
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    concurrency: PositiveInt = 4


class EmbeddingConfig(StrictModel):
    model_id: str
    device: str = "cpu"
    batch_size: PositiveInt = 32


class RetrievalConfig(StrictModel):
    enabled: bool = True
    k: PositiveInt = 5
    minimum_reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_weight: float = 1.0
    language_weight: float = 0.1
    source_weight: float = 0.05
    reliability_weight: float = 0.1
    cross_lingual: bool = True


class AttributionConfig(StrictModel):
    enabled: bool = True
    llm_for_errors_only: bool = True
    max_retries: int = Field(default=2, ge=0)


class GeneralizationConfig(StrictModel):
    enabled: bool = True
    merge_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    minimum_support: int = Field(default=2, ge=1)
    minimum_batches: int = Field(default=2, ge=1)
    maximum_contradiction_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    minimum_active_reliability: float = Field(default=0.60, ge=0.0, le=1.0)


class RunConfig(StrictModel):
    train_paths: list[Path]
    dev_paths: list[Path]
    test_paths: list[Path]
    output_root: Path
    train_batch_size: PositiveInt = 1
    checkpoints: list[PositiveInt] = []
    seed: int = 42
    use_cache: bool = True


class ExperimentConfig(StrictModel):
    model: ModelConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    attribution: AttributionConfig = AttributionConfig()
    generalization: GeneralizationConfig = GeneralizationConfig()
    experiment: RunConfig


def load_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return ExperimentConfig.model_validate(payload)


def redacted_config(config: ExperimentConfig) -> dict:
    return config.model_dump(mode="json")


def config_hash(config: ExperimentConfig) -> str:
    canonical = json.dumps(redacted_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
