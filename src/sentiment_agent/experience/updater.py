from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.schemas import Experience, Feedback, Prediction, PredictionInput, RetrievedExperience


class ExperienceUpdater:
    def __init__(self, repository: ExperienceRepository, vector_index: VectorIndex) -> None:
        self.repository = repository
        self.vector_index = vector_index

    def apply(self, item: PredictionInput, prediction: Prediction, feedback: Feedback,
              retrieved: Sequence[RetrievedExperience], vector: np.ndarray, *, batch_id: int) -> Experience:
        if item.id != prediction.sample_id or item.id != feedback.sample_id:
            raise ValueError("sample id mismatch")
        if prediction.label != feedback.predicted_label:
            raise ValueError("feedback prediction mismatch")
        for result in retrieved:
            agrees = result.experience.sentiment == feedback.gold_label
            self.repository.merge_counts(result.experience.id, success_delta=int(agrees),
                                         failure_delta=int(not agrees), batch_id=batch_id)
            self.repository.record_outcome(
                experience_id=result.experience.id, sample_id=item.id, batch_id=batch_id,
                retrieval_rank=result.rank, score=result.score, injected=True,
                prediction_correct=feedback.correct,
            )
        kind = "successful_case" if feedback.correct else "error_correction"
        raw_id = "\x1f".join((item.text.casefold(), item.language, item.source, feedback.gold_label, kind))
        experience = Experience(
            id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24], type=kind,
            language=item.language, source=item.source, text=item.text,
            semantic_summary=prediction.reason, sentiment=feedback.gold_label,
            reason=prediction.reason if feedback.correct else f"Correct label: {feedback.gold_label}. Previous reasoning: {prediction.reason}",
            source_sample_id=item.id, created_batch=batch_id, last_used_batch=batch_id,
        )
        existing = self.repository.find_by_dedup_key(self.repository.dedup_key(experience))
        if existing is None:
            self.repository.create(experience, batch_id=batch_id)
            self.vector_index.upsert(experience.id, vector)
            return experience
        return self.repository.merge_counts(existing.id, success_delta=int(feedback.correct),
                                            failure_delta=int(not feedback.correct), batch_id=batch_id)
