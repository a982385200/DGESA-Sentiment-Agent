from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sentiment_agent.dgesa.evolution import DGESAEvolutionService
from sentiment_agent.dgesa.llm import AppendixLLM
from sentiment_agent.dgesa.models import PaperEvaluation, PaperPrediction, PaperPredictionPayload
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder
from sentiment_agent.dgesa.retrieval import PatternRetriever, SampleRetriever
from sentiment_agent.embeddings.base import EmbeddingBackend
from sentiment_agent.evaluation.metrics import classification_metrics
from sentiment_agent.schemas import PredictionInput, SentimentExample


class PaperDGESA:
    def __init__(self, *, embedding: EmbeddingBackend, llm: AppendixLLM,
                 prompts: AppendixPromptBuilder, sample_retriever: SampleRetriever,
                 pattern_retriever: PatternRetriever,
                 evolution: DGESAEvolutionService, sample_k: int = 3,
                 pattern_k: int = 3) -> None:
        self.embedding = embedding
        self.llm = llm
        self.prompts = prompts
        self.sample_retriever = sample_retriever
        self.pattern_retriever = pattern_retriever
        self.evolution = evolution
        self.sample_k = sample_k
        self.pattern_k = pattern_k

    async def predict(self, item: PredictionInput) -> PaperPrediction:
        vector = np.asarray(self.embedding.embed([item.text])[0], dtype=np.float32)
        samples = self.sample_retriever.search(vector, k=self.sample_k)
        patterns = self.pattern_retriever.search(
            vector, language=item.language, evidence_vector=vector, k=self.pattern_k)
        payload = await self.llm.complete(self.prompts.prediction(
            text=item.text,
            sample_experiences=[value.experience for value in samples],
            pattern_experiences=[value.experience for value in patterns]),
            PaperPredictionPayload)
        return PaperPrediction(
            sample_id=item.id, **payload.model_dump(),
            sample_experience_ids=tuple(value.experience.id for value in samples),
            pattern_experience_ids=tuple(value.experience.id for value in patterns),
        )

    async def train(self, examples: Sequence[SentimentExample], *,
                    start_batch: int = 1) -> list[PaperPrediction]:
        predictions = []
        for offset, example in enumerate(examples):
            batch_id = start_batch + offset
            prediction = await self.predict(example.to_prediction_input())
            predictions.append(prediction)
            self.evolution.record_retrieved_evidence(
                list(prediction.pattern_experience_ids), gold_sentiment=example.label,
                language=example.language, batch_id=batch_id)
            if prediction.sentiment != example.label:
                await self.evolution.learn_error(
                    example.to_prediction_input(),
                    predicted_language=prediction.language,
                    predicted_sentiment=prediction.sentiment,
                    prediction_reason=prediction.reason,
                    gold_sentiment=example.label, batch_id=batch_id)
        return predictions

    async def evaluate(self, examples: Sequence[SentimentExample]) -> PaperEvaluation:
        predictions = tuple([
            await self.predict(example.to_prediction_input()) for example in examples
        ])
        metrics = classification_metrics(
            [example.label for example in examples],
            [prediction.sentiment for prediction in predictions])
        return PaperEvaluation(predictions=predictions, metrics=metrics)
