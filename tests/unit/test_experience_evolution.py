from pathlib import Path

import numpy as np

from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.retrieval import ExperienceRetriever, RetrievalWeights
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.schemas import Experience, Feedback, Prediction, PredictionInput


def make_experience(id_: str = "one") -> Experience:
    return Experience(id=id_, type="successful_case", language="vi", source="tiny", text="tốt",
        semantic_summary="good", sentiment="positive", reason="positive", source_sample_id="vi-1",
        created_batch=1, last_used_batch=1)


def test_vector_snapshot_does_not_see_later_upsert(tmp_path: Path) -> None:
    index = VectorIndex(tmp_path)
    index.upsert("one", np.array([1.0, 0.0], dtype=np.float32))
    snapshot = index.snapshot()
    index.upsert("two", np.array([0.0, 1.0], dtype=np.float32))
    assert snapshot.ids == ("one",)


def test_retriever_ranks_exact_match_and_reports_components(tmp_path: Path) -> None:
    with ExperienceRepository(tmp_path / "db.sqlite3") as repo:
        repo.create(make_experience(), batch_id=1)
        index = VectorIndex(tmp_path / "index")
        index.upsert("one", np.array([1.0, 0.0], dtype=np.float32))
        retriever = ExperienceRetriever(repo, RetrievalWeights())
        result = retriever.search(np.array([1.0, 0.0], dtype=np.float32), index.snapshot(),
                                  language="vi", source="tiny", k=1)
        assert result[0].experience.id == "one"
        assert result[0].score == sum(result[0].score_components.values())


def test_wrong_prediction_creates_error_correction(tmp_path: Path) -> None:
    with ExperienceRepository(tmp_path / "db.sqlite3") as repo:
        index = VectorIndex(tmp_path / "index")
        updater = ExperienceUpdater(repo, index)
        item = PredictionInput(id="vi-2", text="không tốt", language="vi", source="tiny")
        prediction = Prediction(sample_id="vi-2", label="positive", confidence=.8, reason="contains good",
                                model_name="fake")
        feedback = Feedback(sample_id="vi-2", predicted_label="positive", gold_label="negative", correct=False)
        learned = updater.apply(item, prediction, feedback, [], np.array([1., 0.]), batch_id=2)
        assert learned.type == "error_correction"
        assert repo.count() == 1
