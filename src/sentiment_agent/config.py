from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, HttpUrl, PositiveInt, model_validator

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
    enable_thinking: bool = False


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
    activate_all: bool = False
    merge_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    minimum_support: int = Field(default=2, ge=1)
    minimum_batches: int = Field(default=2, ge=1)
    maximum_contradiction_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    minimum_active_reliability: float = Field(default=0.60, ge=0.0, le=1.0)
    minimum_language_support: int = Field(default=2, ge=1)
    minimum_cross_lingual_languages: int = Field(default=2, ge=2)
    minimum_global_languages: int = Field(default=3, ge=3)
    require_active_experience: bool = False
    consolidation_enabled: bool = False
    consolidation_batch_size: int = Field(default=20, ge=2)
    consolidation_target_rules: int = Field(default=20, ge=1)


class DGESAConfig(StrictModel):
    enabled: bool = False
    admission_candidates: int = Field(default=5, ge=1)
    coverage_temperature: float = Field(default=.10, gt=0)
    low_coverage_threshold: float = Field(default=.25, ge=0, le=1)
    high_coverage_threshold: float = Field(default=.85, ge=0, le=1)
    alignment_candidates: int = Field(default=5, ge=1)
    sample_similarity_threshold: float = Field(default=.80, ge=0, le=1)
    local_similarity_threshold: float = Field(default=.95, ge=0, le=1)
    sample_retrieval_k: int = Field(default=3, ge=1)
    pattern_retrieval_k: int = Field(default=3, ge=1)
    score_weights: tuple[float, float, float] = (.6, .3, .1)
    minimum_active_reliability: float = Field(default=.60, ge=0, le=1)
    maximum_conflict_ratio: float = Field(default=.20, ge=0, le=1)
    minimum_global_languages: int = Field(default=3, ge=2)
    minimum_language_support: int = Field(default=5, ge=1)
    max_generation_attempts: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> DGESAConfig:
        if self.low_coverage_threshold > self.high_coverage_threshold:
            raise ValueError("low coverage threshold must not exceed high threshold")
        return self


StrategyName = Literal["direct", "translation", "experience", "reflection"]


class StrategyConfig(StrictModel):
    enabled: bool = False
    default_strategy: StrategyName = "experience"
    allowed_strategies: list[StrategyName] = [
        "direct", "translation", "experience", "reflection"
    ]
    exploration_weight: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def validate_strategies(self) -> StrategyConfig:
        if not self.allowed_strategies:
            raise ValueError("allowed_strategies must not be empty")
        if len(set(self.allowed_strategies)) != len(self.allowed_strategies):
            raise ValueError("allowed_strategies must not contain duplicates")
        if self.default_strategy not in self.allowed_strategies:
            raise ValueError("default_strategy must be included in allowed_strategies")
        return self


class RunConfig(StrictModel):
    train_paths: list[Path]
    test_paths: list[Path]
    output_root: Path
    train_batch_size: PositiveInt = 1
    checkpoints: list[PositiveInt] = []
    seed: int = 42
    use_cache: bool = True
    train_limit: PositiveInt | None = None
    test_limit: PositiveInt | None = None


class ExperimentConfig(StrictModel):
    model: ModelConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    attribution: AttributionConfig = AttributionConfig()
    generalization: GeneralizationConfig = GeneralizationConfig()
    strategy: StrategyConfig = StrategyConfig()
    dgesa: DGESAConfig = DGESAConfig()
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
