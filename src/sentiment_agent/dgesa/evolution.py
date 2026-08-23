from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from sentiment_agent.dgesa.llm import AppendixLLM
from sentiment_agent.dgesa.models import (
    DualExperiencePayload, PaperPredictionPayload, PatternAbstractionPayload,
    PatternAlignmentPayload, PatternExperience, SampleAdmissionPayload,
    SampleExperience, StrictModel,
)
from sentiment_agent.dgesa.policies import pattern_scope, pattern_status, weighted_coverage
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder
from sentiment_agent.dgesa.repository import DGESARepository
from sentiment_agent.embeddings.base import EmbeddingBackend
from sentiment_agent.schemas import PredictionInput, SentimentLabel


@dataclass(frozen=True)
class DGESAParameters:
    admission_candidates: int = 5
    coverage_temperature: float = .10
    low_coverage_threshold: float = .25
    high_coverage_threshold: float = .85
    alignment_candidates: int = 5
    minimum_active_reliability: float = .60
    maximum_conflict_ratio: float = .20
    minimum_global_languages: int = 3
    minimum_language_support: int = 5
    max_generation_attempts: int = 3


class EvolutionResult(StrictModel):
    sample_saved: bool
    sample_id: str | None = None
    pattern_id: str | None = None
    generation_attempts: int


class DGESAEvolutionService:
    def __init__(self, repository: DGESARepository, embedding: EmbeddingBackend,
                 llm: AppendixLLM, prompts: AppendixPromptBuilder,
                 parameters: DGESAParameters | None = None) -> None:
        self.repository = repository
        self.embedding = embedding
        self.llm = llm
        self.prompts = prompts
        self.parameters = parameters or DGESAParameters()

    async def learn_error(self, item: PredictionInput, *, predicted_language: str,
                          predicted_sentiment: SentimentLabel, prediction_reason: str,
                          gold_sentiment: SentimentLabel, batch_id: int) -> EvolutionResult:
        generated = None
        for attempt in range(1, self.parameters.max_generation_attempts + 1):
            generated = await self.llm.complete(self.prompts.generation(
                text=item.text, predicted_language=predicted_language,
                predicted_sentiment=predicted_sentiment, gold_sentiment=gold_sentiment,
                prediction_reason=prediction_reason), DualExperiencePayload)
            sample_valid = await self._validate_sample(item, generated, gold_sentiment, batch_id)
            pattern_valid = await self._validate_pattern(item, generated, gold_sentiment, batch_id)
            pattern_valid = pattern_valid and generated.pattern_label == gold_sentiment
            if sample_valid and pattern_valid:
                break
        else:
            return EvolutionResult(sample_saved=False, generation_attempts=self.parameters.max_generation_attempts)
        sample = self._sample(item, generated, gold_sentiment, batch_id)
        sample_vector = self._embed(f"{item.text} {sample.experience}")
        sample_experience_vector = self._embed(sample.experience)
        sample_saved = await self._admit_sample(
            sample, sample_vector, sample_experience_vector,
            predicted_sentiment=predicted_sentiment,
            gold_sentiment=gold_sentiment)
        pattern_id = await self._merge_pattern(item, generated, batch_id)
        return EvolutionResult(
            sample_saved=sample_saved, sample_id=sample.id if sample_saved else None,
            pattern_id=pattern_id, generation_attempts=attempt,
        )

    def record_retrieved_evidence(self, pattern_ids: list[str], *,
                                  gold_sentiment: SentimentLabel, language: str,
                                  batch_id: int) -> None:
        for pattern_id in pattern_ids:
            pattern = self.repository.get_pattern(pattern_id)
            support = pattern.sentiment == gold_sentiment
            support_by_language = dict(pattern.support_by_language)
            contradiction_by_language = dict(pattern.contradiction_by_language)
            target = support_by_language if support else contradiction_by_language
            target[language] = target.get(language, 0) + 1
            updated = pattern.model_copy(update={
                "support_count": pattern.support_count + int(support),
                "contradiction_count": pattern.contradiction_count + int(not support),
                "support_by_language": support_by_language,
                "contradiction_by_language": contradiction_by_language,
                "last_updated_batch": batch_id,
            })
            updated = self._lifecycle(updated)
            self.repository.save_pattern(updated, self._embed(updated.text))

    async def _validate_sample(self, item, generated, gold, batch_id) -> bool:
        sample = self._sample(item, generated, gold, batch_id)
        result = await self.llm.complete(self.prompts.prediction(
            text=item.text, sample_experiences=[sample], pattern_experiences=[]),
            PaperPredictionPayload)
        return result.sentiment == gold

    async def _validate_pattern(self, item, generated, gold, batch_id) -> bool:
        pattern = PatternExperience(
            id="validation", text=generated.pattern_experience,
            sentiment=generated.pattern_label, source_language=item.language,
            created_batch=batch_id, last_updated_batch=batch_id,
        )
        result = await self.llm.complete(self.prompts.prediction(
            text=item.text, sample_experiences=[], pattern_experiences=[pattern]),
            PaperPredictionPayload)
        return result.sentiment == gold

    async def _admit_sample(self, sample: SampleExperience, retrieval_vector: np.ndarray,
                            experience_vector: np.ndarray, *,
                            predicted_sentiment: SentimentLabel,
                            gold_sentiment: SentimentLabel) -> bool:
        ranked = []
        query = _normalized(experience_vector)
        for candidate, _, candidate_experience_vector in self.repository.sample_records():
            similarity = float(query @ _normalized(candidate_experience_vector))
            ranked.append((similarity, candidate.id, candidate, candidate_experience_vector))
        ranked.sort(key=lambda value: (-value[0], value[1]))
        candidates = ranked[:self.parameters.admission_candidates]
        coverage = weighted_coverage(
            experience_vector, [row[3] for row in candidates],
            temperature=self.parameters.coverage_temperature)
        admitted = coverage < self.parameters.low_coverage_threshold
        if (self.parameters.low_coverage_threshold <= coverage
                <= self.parameters.high_coverage_threshold):
            decision = await self.llm.complete(self.prompts.admission(
                text=sample.text, predicted_sentiment=predicted_sentiment,
                gold_sentiment=gold_sentiment, current_experience=sample.experience,
                candidates=[row[2].experience for row in candidates]),
                SampleAdmissionPayload)
            admitted = decision.admission == "informative"
        if admitted:
            self.repository.save_sample(
                sample, retrieval_vector, experience_vector=experience_vector)
        return admitted

    async def _merge_pattern(self, item: PredictionInput, generated: DualExperiencePayload,
                             batch_id: int) -> str:
        vector = self._embed(generated.pattern_experience)
        query = _normalized(vector)
        candidates = []
        for pattern, candidate_vector in self.repository.pattern_vectors():
            if pattern.sentiment != generated.pattern_label or pattern.status == "suppressed":
                continue
            similarity = float(query @ _normalized(candidate_vector))
            candidates.append((similarity, pattern.id, pattern))
        candidates.sort(key=lambda value: (-value[0], value[1]))
        candidates = candidates[:self.parameters.alignment_candidates]
        selected = None
        if candidates:
            decision = await self.llm.complete(self.prompts.alignment(
                current_pattern=generated.pattern_experience,
                pattern_label=generated.pattern_label,
                candidates={row[1]: row[2].text for row in candidates}),
                PatternAlignmentPayload)
            selected_id = _aligned_id(decision.alignment)
            selected = next((row[2] for row in candidates if row[1] == selected_id), None)
        if selected is None:
            pattern_id = _id(generated.pattern_label, generated.pattern_experience)
            pattern = PatternExperience(
                id=pattern_id, text=generated.pattern_experience,
                sentiment=generated.pattern_label, source_language=item.language,
                evidence_texts=(item.text,), support_count=1,
                support_by_language={item.language: 1},
                created_batch=batch_id, last_updated_batch=batch_id,
            )
        else:
            abstracted = await self.llm.complete(self.prompts.abstraction(
                existing_pattern=selected.text,
                new_pattern=generated.pattern_experience), PatternAbstractionPayload)
            support_by_language = dict(selected.support_by_language)
            support_by_language[item.language] = support_by_language.get(item.language, 0) + 1
            pattern = selected.model_copy(update={
                "text": abstracted.updated_pattern_experience,
                "evidence_texts": tuple(dict.fromkeys((*selected.evidence_texts, item.text))),
                "support_count": selected.support_count + 1,
                "support_by_language": support_by_language,
                "last_updated_batch": batch_id,
            })
        pattern = self._lifecycle(pattern)
        self.repository.save_pattern(
            pattern, self._embed(pattern.text),
            evidence_vectors=[self._embed(text) for text in pattern.evidence_texts],
        )
        return pattern.id

    def _lifecycle(self, pattern: PatternExperience) -> PatternExperience:
        return pattern.model_copy(update={
            "status": pattern_status(
                pattern, self.parameters.minimum_active_reliability,
                self.parameters.maximum_conflict_ratio),
            "scope": pattern_scope(
                pattern, self.parameters.minimum_language_support,
                self.parameters.minimum_global_languages),
        })

    @staticmethod
    def _sample(item, generated, gold, batch_id) -> SampleExperience:
        return SampleExperience(
            id=_id(item.id, generated.sample_experience), text=item.text,
            experience=generated.sample_experience, sentiment=gold,
            language=item.language, source=item.source,
            source_sample_id=item.id, created_batch=batch_id,
        )

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.embedding.embed([text])[0], dtype=np.float32)


def _aligned_id(value: str) -> str | None:
    return value[6:-1] if value.startswith("align(") and value.endswith(")") else None


def _id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    return value / np.linalg.norm(value)
