import json
from pathlib import Path

from sentiment_agent.workflow import run_from_config
from tests.fixtures.fake_llm import FakeLLMClient


def test_full_offline_research_workflow(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures"
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "seed: 42\n"
        "model:\n"
        "  name: fake-model\n"
        "memory:\n"
        "  retrieval_k: 2\n"
        "  min_reliability: 0.0\n"
        "strategy:\n"
        "  names: [direct, memory]\n"
        "  epsilon: 0.0\n"
        "experiment:\n"
        "  name: integration\n"
        f"  train_paths: ['{(fixture_root / 'tiny_dataset.jsonl').as_posix()}']\n"
        f"  test_paths: ['{(fixture_root / 'tiny_test.jsonl').as_posix()}']\n"
        "  checkpoints: [0, 3, 6]\n"
        "  reflection_enabled: true\n"
        "  cross_lingual_enabled: true\n",
        encoding="utf-8",
    )

    summary = run_from_config(config, output_root=tmp_path / "outputs", client=FakeLLMClient())

    assert summary.processed_training_samples == 6
    assert summary.evaluation_samples == 3
    assert summary.memory_experiences > 0
    assert (summary.output_dir / "config.yaml").exists()
    assert (summary.output_dir / "manifest.json").exists()
    assert (summary.output_dir / "predictions.jsonl").exists()
    assert (summary.output_dir / "metrics.json").exists()
    assert (summary.output_dir / "costs.json").exists()

    prediction_rows = [
        json.loads(line)
        for line in (summary.output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["split"] for row in prediction_rows} == {"train", "test"}
    assert summary.persisted_experience_texts.isdisjoint(
        {"good test sample", "bad test sample", "okay test sample"}
    )


def test_workflow_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parents[1] / "fixtures"
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "seed: 42\n"
        "model:\n  name: fake-model\n"
        "strategy:\n  names: [direct, memory]\n  epsilon: 0.0\n"
        "experiment:\n"
        "  name: deterministic\n"
        f"  train_paths: ['{(fixture_root / 'tiny_dataset.jsonl').as_posix()}']\n"
        f"  test_paths: ['{(fixture_root / 'tiny_test.jsonl').as_posix()}']\n"
        "  checkpoints: [0, 6]\n",
        encoding="utf-8",
    )

    first = run_from_config(config, output_root=tmp_path / "first", client=FakeLLMClient())
    second = run_from_config(config, output_root=tmp_path / "second", client=FakeLLMClient())

    first_metrics = json.loads((first.output_dir / "metrics.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert first_metrics == second_metrics
    assert first.persisted_experience_texts == second.persisted_experience_texts
