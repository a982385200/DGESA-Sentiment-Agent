from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from sentiment_agent.llm.parsing import PredictionPayload
from sentiment_agent.memory.retrieval import ExperienceRetriever, RetrievedExperience
from sentiment_agent.memory.store import ExperienceStore
from sentiment_agent.reflection.reflector import ReflectionPayload, Reflector
from sentiment_agent.schemas import (
    Experience,
    Feedback,
    Prediction,
    PredictionInput,
    StrictModel,
)
from sentiment_agent.strategies.prompts import PromptBuilder
from sentiment_agent.strategies.selector import EpsilonGreedySelector


class AgentClient(Protocol):
    def embed(self, texts): ...

    def chat_json(self, messages, response_model): ...


class LearningResult(StrictModel):
    sample_id: str
    experience_ids: list[str] = Field(default_factory=list)
    reflection: ReflectionPayload | None = None
    reward: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class _PredictionContext:
    vector: list[float]
    retrieved: list[RetrievedExperience]
    prediction: Prediction


class SentimentAgent:
    def __init__(
        self,
        *,
        client: AgentClient,
        retriever: ExperienceRetriever,
        store: ExperienceStore,
        prompt_builder: PromptBuilder,
        selector: EpsilonGreedySelector,
        model_name: str,
        reflection_enabled: bool,
        cross_lingual: bool,
        retrieval_k: int,
        seed: int,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.store = store
        self.prompt_builder = prompt_builder
        self.selector = selector
        self.model_name = model_name
        self.cross_lingual = cross_lingual
        self.retrieval_k = retrieval_k
        self.rng = random.Random(seed)
        self.reflector = Reflector(client, enabled=reflection_enabled)
        self._contexts: dict[str, _PredictionContext] = {}
        self._round = 0

    def predict(self, item: PredictionInput) -> Prediction:
        started = time.perf_counter()
        vector = self.client.embed([item.text])[0]
        retrieved = self.retriever.search(
            vector,
            language=item.language,
            source=item.source,
            cross_lingual=self.cross_lingual,
            k=self.retrieval_k,
            current_round=self._round,
        )
        strategy = self.selector.select(item.language, self.rng)
        messages = self.prompt_builder.build(strategy, item, retrieved)
        result = self.client.chat_json(messages, PredictionPayload)
        prediction = Prediction(
            sample_id=item.id,
            label=result.payload.label,
            confidence=result.payload.confidence,
            reason=result.payload.reason,
            strategy=strategy,
            retrieved_experience_ids=[
                retrieved_item.experience.id
                for retrieved_item in retrieved
                if retrieved_item.experience.id is not None
            ],
            model_name=self.model_name,
            usage=result.usage,
            latency_seconds=time.perf_counter() - started,
            cache_key=result.cache_key,
        )
        self._contexts[item.id] = _PredictionContext(
            vector=list(vector),
            retrieved=list(retrieved),
            prediction=prediction,
        )
        return prediction

    def learn(
        self,
        item: PredictionInput,
        prediction: Prediction,
        feedback: Feedback,
    ) -> LearningResult:
        if item.id != prediction.sample_id or item.id != feedback.sample_id:
            raise ValueError("sample id mismatch between input, prediction, and feedback")
        if prediction.label != feedback.predicted_label:
            raise ValueError("feedback predicted label does not match prediction")
        context = self._contexts.get(item.id)
        if context is None or context.prediction != prediction:
            raise ValueError("learn requires the matching prior prediction")

        reflection = self.reflector.reflect(item, prediction, feedback, context.retrieved)
        reason = reflection.corrected_reason if reflection is not None else prediction.reason
        experience_type = "successful_case" if feedback.correct else "error_correction"
        case = Experience(
            text=item.text,
            language=item.language,
            source=item.source,
            semantic_meaning=reason,
            sentiment=feedback.gold_label,
            reason=reason,
            experience_type=experience_type,
            created_round=self._round,
            last_used_round=self._round,
        )
        experience_ids = [self.store.add_or_update(case, context.vector)]

        if reflection is not None:
            rule_vector = self.client.embed([reflection.generalized_rule])[0]
            rule = Experience(
                text=reflection.generalized_rule,
                language=item.language,
                source=item.source,
                semantic_meaning=reflection.generalized_rule,
                sentiment=feedback.gold_label,
                reason=reflection.corrected_reason,
                experience_type="generalized_rule",
                created_round=self._round,
                last_used_round=self._round,
            )
            experience_ids.append(self.store.add_or_update(rule, rule_vector))

        reward = float(feedback.correct)
        self.selector.update(item.language, prediction.strategy, reward)
        self._round += 1
        del self._contexts[item.id]
        return LearningResult(
            sample_id=item.id,
            experience_ids=experience_ids,
            reflection=reflection,
            reward=reward,
        )
