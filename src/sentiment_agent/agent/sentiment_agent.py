from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sentiment_agent.embeddings.base import EmbeddingBackend
from sentiment_agent.experience.retrieval import ExperienceRetriever
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.llm.base import LLMBackend
from sentiment_agent.prompts.prediction import PredictionPromptBuilder
from sentiment_agent.schemas import Feedback, Prediction, PredictionInput, RetrievedExperience


@dataclass(frozen=True)
class PredictionContext:
    vector: np.ndarray
    retrieved: tuple[RetrievedExperience, ...]


class SentimentAgent:
    def __init__(self, *, embedding: EmbeddingBackend, llm: LLMBackend,
                 retriever: ExperienceRetriever, updater: ExperienceUpdater,
                 vector_index: VectorIndex, prompt_builder: PredictionPromptBuilder,
                 model_name: str, retrieval_k: int, evolution_service=None) -> None:
        self.embedding = embedding
        self.llm = llm
        self.retriever = retriever
        self.updater = updater
        self.vector_index = vector_index
        self.prompt_builder = prompt_builder
        self.model_name = model_name
        self.retrieval_k = retrieval_k
        self.evolution_service = evolution_service
        self._contexts: dict[str, PredictionContext] = {}

    async def predict_batch(self, items: Sequence[PredictionInput], *, max_concurrency: int) -> list[Prediction]:
        values = list(items)
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if len({item.id for item in values}) != len(values):
            raise ValueError("batch contains duplicate sample ids")
        vectors = self.embedding.embed([item.text for item in values])
        snapshot = self.vector_index.snapshot()
        prepared = []
        for item, vector in zip(values, vectors, strict=True):
            retrieved = self.retriever.search(vector, snapshot, language=item.language,
                                               source=item.source, k=self.retrieval_k)
            prepared.append((item, vector, retrieved, self.prompt_builder.build(item, retrieved)))
        semaphore = asyncio.Semaphore(max_concurrency)

        async def classify(entry):
            item, vector, retrieved, messages = entry
            started = time.perf_counter()
            async with semaphore:
                result = await self.llm.classify(messages)
            prediction = Prediction(
                sample_id=item.id, label=result.payload.label, confidence=result.payload.confidence,
                reason=result.payload.reason,
                retrieved_experience_ids=tuple(value.experience.id for value in retrieved),
                model_name=self.model_name, usage=result.usage,
                latency_seconds=time.perf_counter() - started, cache_key=result.cache_key,
            )
            return prediction, PredictionContext(np.asarray(vector).copy(), tuple(retrieved))

        results = await asyncio.gather(*(classify(entry) for entry in prepared))
        for prediction, context in results:
            self._contexts[prediction.sample_id] = context
        return [prediction for prediction, _ in results]

    def learn_batch(self, items: Sequence[PredictionInput], predictions: Sequence[Prediction],
                    feedback: Sequence[Feedback], *, batch_id: int):
        triples = self._validated_learning_rows(items, predictions, feedback)
        learned = []
        for item, prediction, outcome in triples:
            context = self._contexts[item.id]
            learned.append(self.updater.apply(item, prediction, outcome, context.retrieved,
                                              context.vector, batch_id=batch_id))
            del self._contexts[item.id]
        return learned

    async def evolve_batch(self, items: Sequence[PredictionInput], predictions: Sequence[Prediction],
                           feedback: Sequence[Feedback], *, batch_id: int):
        if self.evolution_service is None:
            raise RuntimeError("generalized experience evolution is not configured")
        triples = self._validated_learning_rows(items, predictions, feedback)
        contexts = [self._contexts[item.id].retrieved for item, _, _ in triples]
        learned = await self.evolution_service.learn_batch(
            items, predictions, feedback, contexts, batch_id=batch_id)
        for item, _, _ in triples:
            del self._contexts[item.id]
        return learned

    def experience_count(self) -> int:
        if self.evolution_service is not None:
            return int(self.evolution_service.repository.stats()["active_count"])
        return self.updater.repository.count()

    def _validated_learning_rows(self, items, predictions, feedback):
        triples = list(zip(items, predictions, feedback, strict=True))
        for item, prediction, outcome in triples:
            if item.id != prediction.sample_id or item.id != outcome.sample_id:
                raise ValueError("sample id mismatch in learning batch")
            if prediction.label != outcome.predicted_label or item.id not in self._contexts:
                raise ValueError("learning requires matching prior predictions")
        return triples
