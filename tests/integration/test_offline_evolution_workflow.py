import json
from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.retrieval import ExperienceRetriever, RetrievalWeights
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import ExperimentRunner
from sentiment_agent.experiments.progress import RecordingProgressReporter
from sentiment_agent.llm.base import LLMResult, PredictionPayload
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import SentimentExample


class FakeEmbedding:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeLLM:
    async def classify(self, messages):
        return LLMResult(payload=PredictionPayload(label="positive", confidence=.9, reason="offline fake"))


def example(id_: str, label: str = "positive") -> SentimentExample:
    return SentimentExample(id=id_, text=f"text {id_}", label=label, language="vi", source="tiny")


@pytest.mark.anyio
async def test_complete_offline_evolution_workflow(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with ExperienceRepository(run_dir / "experience_store" / "experiences.sqlite3") as repo:
        progress = RecordingProgressReporter()
        index = VectorIndex(run_dir / "experience_store")
        agent = SentimentAgent(embedding=FakeEmbedding(), llm=FakeLLM(),
            retriever=ExperienceRetriever(repo, RetrievalWeights()), updater=ExperienceUpdater(repo, index),
            vector_index=index, prompt_builder=PredictionPromptBuilder(), model_name="fake", retrieval_k=2)
        runner = ExperimentRunner(agent=agent, writer=ArtifactWriter(run_dir),
                                  batch_size=2, concurrency=2, checkpoints=[2, 4],
                                  manifest_metadata={"config_hash": "offline-test"},
                                  progress_reporter=progress)
        summary = await runner.run(
            [example("train-1"), example("train-2"), example("train-3"), example("train-4", "negative")],
            [example("dev-1")], [example("test-1")])
        assert summary.completed_samples == 4
        assert summary.checkpoints == (2, 4)
        assert repo.count() == 4
        assert "test-1" not in {item.source_sample_id for item in repo.list()}
        train_events = [event for event in progress.events if event.stage == "train"]
        assert [event.completed_samples for event in train_events] == [2, 4]
        assert [event.experience_count for event in train_events] == [2, 4]
        assert {event.stage for event in progress.events} == {"train", "dev", "test"}
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in rows if row["split"] == "train"]
    assert all(not row["retrieved_experience_ids"] for row in train_rows if row["batch_id"] == 1)
    assert any(row["retrieved_experience_ids"] for row in train_rows if row["batch_id"] == 2)
    test_rows = [row for row in rows if row["split"] == "test"]
    assert len(test_rows) == 1
    assert test_rows[0]["sample_id"] == "test-1"
    assert test_rows[0]["text"] == "text test-1"
    assert test_rows[0]["language"] == "vi"
    assert test_rows[0]["gold_label"] == "positive"
    assert test_rows[0]["label"] == "positive"
    assert test_rows[0]["checkpoint"] is None
    assert (run_dir / "metrics.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["config_hash"] == "offline-test"
    assert json.loads((run_dir / "costs.json").read_text())["calls"] >= 5
