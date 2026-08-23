from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.dgesa.evolution import DGESAEvolutionService, DGESAParameters
from sentiment_agent.dgesa.models import (
    DualExperiencePayload, PaperPredictionPayload, PatternAbstractionPayload,
    PatternAlignmentPayload, PatternExperience, SampleAdmissionPayload,
)
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder
from sentiment_agent.dgesa.repository import DGESARepository
from sentiment_agent.schemas import PredictionInput


class Embedding:
    def embed(self, texts):
        values = []
        for text in texts:
            if "orthogonal" in text:
                values.append([0.0, 1.0])
            elif "mid" in text:
                values.append([.5, np.sqrt(.75)])
            else:
                values.append([1.0, 0.0])
        return np.asarray(values, dtype=np.float32)


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def complete(self, messages, schema):
        value = self.responses.pop(0)
        assert isinstance(value, schema)
        return value


def item() -> PredictionInput:
    return PredictionInput(id="x", text="misclassified text", language="vi", source="reviews")


@pytest.mark.anyio
async def test_learn_error_validates_both_granularities_before_saving(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        DualExperiencePayload(sample_experience="local fix", pattern_experience="abstract rule", pattern_label="negative"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="local"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="pattern"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        service = DGESAEvolutionService(repo, Embedding(), llm, AppendixPromptBuilder())
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong cue", gold_sentiment="negative", batch_id=1)
        assert result.sample_saved is True and result.pattern_id is not None
        assert len(repo.list_samples()) == 1 and len(repo.list_patterns()) == 1


@pytest.mark.anyio
async def test_learn_error_regenerates_when_validation_fails(tmp_path: Path) -> None:
    generated = DualExperiencePayload(sample_experience="local fix", pattern_experience="abstract rule", pattern_label="negative")
    llm = ScriptedLLM([
        generated,
        PaperPredictionPayload(language="Vietnamese", sentiment="positive", reason="still wrong"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="pattern"),
        generated,
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        service = DGESAEvolutionService(repo, Embedding(), llm, AppendixPromptBuilder(),
                                        DGESAParameters(max_generation_attempts=2))
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong", gold_sentiment="negative", batch_id=1)
        assert result.generation_attempts == 2


@pytest.mark.anyio
async def test_intermediate_sample_coverage_uses_b3_admission(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        DualExperiencePayload(sample_experience="mid", pattern_experience="orthogonal pattern", pattern_label="negative"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        SampleAdmissionPayload(admission="redundant"),
        PatternAlignmentPayload(alignment="new"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        from sentiment_agent.dgesa.models import SampleExperience
        repo.save_sample(SampleExperience(id="old", text="x", experience="base", sentiment="negative",
                                          language="vi", source="reviews", source_sample_id="old", created_batch=1),
                         np.array([1.0, 0.0]))
        service = DGESAEvolutionService(repo, Embedding(), llm, AppendixPromptBuilder())
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong", gold_sentiment="negative", batch_id=2)
        assert result.sample_saved is False
        assert len(repo.list_samples()) == 1


@pytest.mark.anyio
async def test_pattern_alignment_replaces_abstract_text_and_adds_support(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        DualExperiencePayload(sample_experience="local fix", pattern_experience="new rule", pattern_label="negative"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PatternAlignmentPayload(alignment="align(p1)"),
        PatternAbstractionPayload(updated_pattern_experience="better abstract rule"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        existing = PatternExperience(id="p1", text="old rule", sentiment="negative", source_language="vi",
                                     support_count=1, support_by_language={"vi": 1},
                                     created_batch=1, last_updated_batch=1)
        repo.save_pattern(existing, np.array([1.0, 0.0]))
        service = DGESAEvolutionService(repo, Embedding(), llm, AppendixPromptBuilder())
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong", gold_sentiment="negative", batch_id=2)
        updated = repo.get_pattern(result.pattern_id)
        assert updated.text == "better abstract rule"
        assert updated.support_count == 2


def test_record_retrieved_evidence_counts_contradiction_even_on_error(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        value = PatternExperience(id="p", text="positive rule", sentiment="positive", source_language="vi",
                                  status="active", support_count=2, support_by_language={"vi": 2},
                                  created_batch=1, last_updated_batch=1)
        repo.save_pattern(value, np.array([1.0, 0.0]))
        service = DGESAEvolutionService(repo, Embedding(), ScriptedLLM([]), AppendixPromptBuilder())
        service.record_retrieved_evidence(["p"], gold_sentiment="negative", language="vi", batch_id=2)
        assert repo.get_pattern("p").contradiction_count == 1


@pytest.mark.anyio
async def test_sample_admission_coverage_uses_experience_text_vectors(tmp_path: Path) -> None:
    class SeparateEmbedding:
        def embed(self, texts):
            return np.asarray([
                [0.0, 1.0] if "misclassified text new-local" in text else [1.0, 0.0]
                for text in texts
            ], dtype=np.float32)

    llm = ScriptedLLM([
        DualExperiencePayload(sample_experience="new-local", pattern_experience="new-pattern", pattern_label="negative"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        from sentiment_agent.dgesa.models import SampleExperience
        repo.save_sample(
            SampleExperience(id="old", text="old", experience="old-local", sentiment="negative",
                             language="vi", source="reviews", source_sample_id="old", created_batch=1),
            np.array([1.0, 0.0]), experience_vector=np.array([1.0, 0.0]))
        service = DGESAEvolutionService(repo, SeparateEmbedding(), llm, AppendixPromptBuilder())
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong", gold_sentiment="negative", batch_id=2)
        assert result.sample_saved is False


@pytest.mark.anyio
async def test_pattern_label_must_match_gold_even_if_validation_prediction_is_correct(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        DualExperiencePayload(sample_experience="local", pattern_experience="wrong label rule", pattern_label="positive"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
        PaperPredictionPayload(language="Vietnamese", sentiment="negative", reason="fixed"),
    ])
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        service = DGESAEvolutionService(
            repo, Embedding(), llm, AppendixPromptBuilder(),
            DGESAParameters(max_generation_attempts=1))
        result = await service.learn_error(
            item(), predicted_language="Vietnamese", predicted_sentiment="positive",
            prediction_reason="wrong", gold_sentiment="negative", batch_id=1)
        assert result.pattern_id is None
        assert repo.list_patterns() == []
