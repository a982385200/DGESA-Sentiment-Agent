from pathlib import Path

from sentiment_agent.agent.agent import LearningResult
from sentiment_agent.evaluation.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import ExperimentRunner
from sentiment_agent.schemas import Prediction, SentimentExample


class FakeStore:
    def __init__(self) -> None:
        self.value = 0

    def count(self) -> int:
        return self.value


class FakeAgent:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.learned_ids: list[str] = []

    def predict(self, item):
        return Prediction(
            sample_id=item.id,
            label="positive",
            confidence=1.0,
            reason="fixture",
            strategy="direct",
            model_name="fake",
        )

    def learn(self, item, prediction, feedback):
        self.learned_ids.append(item.id)
        self.store.value += 1
        return LearningResult(sample_id=item.id, experience_ids=[f"exp-{item.id}"], reward=float(feedback.correct))


def example(sample_id: str, split: str, label: str = "positive") -> SentimentExample:
    return SentimentExample(
        id=sample_id,
        text=f"text {sample_id}",
        language="vi",
        source="fixture",
        split=split,
        label=label,
    )


def test_evolution_evaluates_at_configured_checkpoints(tmp_path: Path) -> None:
    agent = FakeAgent()
    runner = ExperimentRunner(agent, ArtifactWriter(tmp_path))
    train = [example(f"train-{index}", "train") for index in range(4)]
    test = [example("test-1", "test")]

    summary = runner.run_evolution(train, test, checkpoints=[0, 2, 4])

    assert [stage.processed_training_samples for stage in summary.stages] == [0, 2, 4]
    assert [stage.experience_count for stage in summary.stages] == [0, 2, 4]


def test_evaluation_samples_are_never_learned(tmp_path: Path) -> None:
    agent = FakeAgent()
    runner = ExperimentRunner(agent, ArtifactWriter(tmp_path))
    train = [example("train-1", "train")]
    test = [example("test-1", "test")]

    runner.run_evolution(train, test, checkpoints=[1])

    assert agent.learned_ids == ["train-1"]


def test_runner_rejects_checkpoint_beyond_training_size(tmp_path: Path) -> None:
    runner = ExperimentRunner(FakeAgent(), ArtifactWriter(tmp_path))

    try:
        runner.run_evolution([example("train-1", "train")], [example("test-1", "test")], checkpoints=[2])
    except ValueError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("expected invalid checkpoint")
