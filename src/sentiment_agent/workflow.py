from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import Field

from sentiment_agent.agent.agent import SentimentAgent
from sentiment_agent.config import AppConfig
from sentiment_agent.data.loader import load_jsonl
from sentiment_agent.evaluation.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import ExperimentRunner
from sentiment_agent.llm.cache import SQLiteResponseCache
from sentiment_agent.llm.client import OpenAICompatibleClient
from sentiment_agent.memory.retrieval import ExperienceRetriever
from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.schemas import StrictModel
from sentiment_agent.strategies.prompts import PromptBuilder
from sentiment_agent.strategies.selector import EpsilonGreedySelector


class WorkflowClient(Protocol):
    def embed(self, texts): ...

    def chat_json(self, messages, response_model): ...


class WorkflowSummary(StrictModel):
    output_dir: Path
    memory_path: Path
    processed_training_samples: int = Field(ge=0)
    evaluation_samples: int = Field(ge=0)
    memory_experiences: int = Field(ge=0)
    persisted_experience_texts: set[str]
    config_hash: str


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _real_client(config: AppConfig, output_dir: Path) -> OpenAICompatibleClient:
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise ValueError(f"missing API key environment variable: {config.model.api_key_env}")
    return OpenAICompatibleClient(
        config.model,
        api_key=api_key,
        cache=SQLiteResponseCache(output_dir / "responses.sqlite3"),
    )


def run_from_config(
    config_path: Path,
    *,
    output_root: Path | None = None,
    client: WorkflowClient | None = None,
) -> WorkflowSummary:
    config = AppConfig.load(config_path)
    root = output_root or config.experiment.output_root
    output_dir = root / f"{config.experiment.name}-{config.config_hash[:12]}"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    writer = ArtifactWriter(output_dir)
    memory_path = output_dir / "experience.sqlite3"
    store = ExperienceStore(memory_path)
    active_client = client or _real_client(config, output_dir)
    retriever = ExperienceRetriever(store, config.memory)
    selector = EpsilonGreedySelector(config.strategy.names, epsilon=config.strategy.epsilon)
    agent = SentimentAgent(
        client=active_client,
        retriever=retriever,
        store=store,
        prompt_builder=PromptBuilder(),
        selector=selector,
        model_name=config.model.name,
        reflection_enabled=config.experiment.reflection_enabled,
        cross_lingual=config.experiment.cross_lingual_enabled,
        retrieval_k=config.memory.retrieval_k,
        seed=config.seed,
    )
    training_examples = [
        example
        for path in config.experiment.train_paths
        for example in load_jsonl(path, split="train")
    ]
    evaluation_examples = [
        example
        for path in config.experiment.test_paths
        for example in load_jsonl(path, split="test")
    ]
    if not training_examples:
        raise ValueError("experiment requires at least one training example")
    if not evaluation_examples:
        raise ValueError("experiment requires at least one test example")

    expanded = config.model_dump(mode="json", exclude={"config_hash"})
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(expanded, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    writer.write_json(
        "manifest.json",
        {
            "config_hash": config.config_hash,
            "seed": config.seed,
            "model_name": config.model.name,
            "git_commit": _git_commit(),
            "status": "running",
        },
    )
    runner = ExperimentRunner(agent, writer)
    result = runner.run_evolution(
        training_examples,
        evaluation_examples,
        checkpoints=config.experiment.checkpoints,
    )
    writer.write_json(
        "manifest.json",
        {
            "config_hash": config.config_hash,
            "seed": config.seed,
            "model_name": config.model.name,
            "git_commit": _git_commit(),
            "status": "complete",
            "processed_training_samples": result.processed_training_samples,
        },
    )
    experiences = store.all_with_vectors()
    return WorkflowSummary(
        output_dir=output_dir,
        memory_path=memory_path,
        processed_training_samples=result.processed_training_samples,
        evaluation_samples=result.evaluation_samples,
        memory_experiences=len(experiences),
        persisted_experience_texts={experience.text for experience, _ in experiences},
        config_hash=config.config_hash,
    )
