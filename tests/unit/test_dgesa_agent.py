import json
from pathlib import Path

import numpy as np
import pytest

from sentiment_agent.dgesa.agent import PaperDGESA
from sentiment_agent.dgesa.evolution import DGESAEvolutionService
from sentiment_agent.dgesa.models import DualExperiencePayload, PaperPredictionPayload
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder
from sentiment_agent.dgesa.repository import DGESARepository
from sentiment_agent.dgesa.retrieval import PatternRetriever, SampleRetriever
from sentiment_agent.schemas import SentimentExample


class Embedding:
    def embed(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class PaperLLM:
    async def complete(self, messages, schema):
        payload = json.loads(messages[1].content)
        if schema is DualExperiencePayload:
            return schema(sample_experience="local negative cue",
                          pattern_experience="treat this construction as negative",
                          pattern_label="negative")
        has_experience = bool(payload.get("sample_level_experiences") or
                              payload.get("pattern_level_experiences"))
        return schema(language="Vietnamese", sentiment="negative" if has_experience else "positive",
                      reason="experience" if has_experience else "baseline")


def make_agent(repo):
    embedding = Embedding()
    llm = PaperLLM()
    prompts = AppendixPromptBuilder()
    evolution = DGESAEvolutionService(repo, embedding, llm, prompts)
    return PaperDGESA(embedding=embedding, llm=llm, prompts=prompts,
                      sample_retriever=SampleRetriever(repo),
                      pattern_retriever=PatternRetriever(repo), evolution=evolution)


def example(id_="x"):
    return SentimentExample(id=id_, text="bad text", language="vi", source="reviews", label="negative")


@pytest.mark.anyio
async def test_predict_returns_retrieval_provenance(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        prediction = await make_agent(repo).predict(example().to_prediction_input())
        assert prediction.sentiment == "positive"
        assert prediction.sample_experience_ids == ()


@pytest.mark.anyio
async def test_train_creates_experience_only_for_error(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        results = await make_agent(repo).train([example()], start_batch=1)
        assert results[0].sentiment == "positive"
        assert len(repo.list_samples()) == 1 and len(repo.list_patterns()) == 1


@pytest.mark.anyio
async def test_evaluate_freezes_experience_store(tmp_path: Path) -> None:
    with DGESARepository(tmp_path / "db.sqlite3") as repo:
        agent = make_agent(repo)
        await agent.train([example()], start_batch=1)
        before = (len(repo.list_samples()), len(repo.list_patterns()))
        result = await agent.evaluate([example("test")])
        assert result.metrics["accuracy"] == 1.0
        assert (len(repo.list_samples()), len(repo.list_patterns())) == before
