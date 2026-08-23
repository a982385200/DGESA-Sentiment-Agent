from __future__ import annotations

import hashlib
from collections.abc import Sequence

from sentiment_agent.attribution.llm_attributor import LLMAttributor
from sentiment_agent.evidence.models import CaseEvidence
from sentiment_agent.embeddings.base import EmbeddingBackend
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.generalization.lifecycle import LifecyclePolicy
from sentiment_agent.generalization.matcher import RuleMatcher
from sentiment_agent.generalization.models import GeneralizedExperience
from sentiment_agent.generalization.repository import EvolutionRepository
from sentiment_agent.schemas import Feedback, Prediction, PredictionInput


class ExperienceEvolutionService:
    def __init__(self, *, repository: EvolutionRepository, embedding: EmbeddingBackend,
                 attributor: LLMAttributor, matcher: RuleMatcher,
                 vector_index: VectorIndex, lifecycle: LifecyclePolicy) -> None:
        self.repository = repository
        self.embedding = embedding
        self.attributor = attributor
        self.matcher = matcher
        self.vector_index = vector_index
        self.lifecycle = lifecycle
        self.attribution_failures: list[dict] = []

    async def learn_batch(self, items: Sequence[PredictionInput],
                          predictions: Sequence[Prediction], feedback: Sequence[Feedback],
                          retrieved_contexts: Sequence[Sequence], *, batch_id: int) -> list[GeneralizedExperience]:
        rows = list(zip(items, predictions, feedback, retrieved_contexts, strict=True))
        learned: list[GeneralizedExperience] = []
        for item, prediction, outcome, retrieved in rows:
            if item.id != prediction.sample_id or item.id != outcome.sample_id:
                raise ValueError("sample id mismatch in evolution batch")
            case = self._case(item, prediction, outcome, retrieved, batch_id)
            self.repository.create_case(case)
            if outcome.correct:
                self._record_existing_rule_evidence(case, retrieved)
                continue
            result = await self.attributor.attribute(case, retrieved)
            if result.used_fallback:
                self.attribution_failures.append({
                    "case_id": case.id,
                    "raw_responses": list(result.raw_responses),
                })
            attribution = result.attribution
            self.repository.create_attribution(attribution)
            vector = self.embedding.embed([
                f"{attribution.candidate_rule} {attribution.corrected_reason}"
            ])[0]
            rule = self.matcher.find_match(vector, sentiment=case.gold_label)
            if rule is None:
                rule = self._new_rule(attribution, case.gold_label, batch_id)
                self.repository.create_rule(rule)
                self.vector_index.upsert(rule.id, vector)
            else:
                rule = self._merge_scope(rule, attribution, batch_id)
                self.repository.update_rule(rule, event_type="merged", batch_id=batch_id)
            self.repository.add_evidence(
                rule.id, case.id, relation="support", batch_id=batch_id,
                attribution_id=attribution.id,
            )
            learned.append(self._apply_lifecycle(rule.id, batch_id))
        return learned

    def _record_existing_rule_evidence(self, case: CaseEvidence, retrieved: Sequence) -> None:
        for item in retrieved:
            rule_id = _item_value(item, "id")
            if rule_id is None:
                continue
            try:
                rule = self.repository.get_rule(rule_id)
            except KeyError:
                continue
            relation = "support" if rule.sentiment == case.gold_label else "contradiction"
            if self.repository.add_evidence(rule.id, case.id, relation=relation, batch_id=case.batch_id):
                self._apply_lifecycle(rule.id, case.batch_id)
            self.repository.record_outcome(
                rule.id, sample_id=case.sample_id, batch_id=case.batch_id, correct=case.correct)

    def _apply_lifecycle(self, rule_id: str, batch_id: int) -> GeneralizedExperience:
        rule = self.repository.get_rule(rule_id)
        status = self.lifecycle.next_status(rule)
        if status != rule.status:
            rule = rule.model_copy(update={"status": status, "version": rule.version + 1})
            self.repository.update_rule(rule, event_type=f"status_{status}", batch_id=batch_id)
        return rule

    @staticmethod
    def _case(item: PredictionInput, prediction: Prediction, outcome: Feedback,
              retrieved: Sequence, batch_id: int) -> CaseEvidence:
        case_id = hashlib.sha256(f"{item.id}\x1f{batch_id}".encode()).hexdigest()[:24]
        return CaseEvidence(
            id=case_id, sample_id=item.id, text=item.text, language=item.language,
            source=item.source, predicted_label=prediction.label,
            gold_label=outcome.gold_label, prediction_reason=prediction.reason,
            confidence=prediction.confidence,
            retrieved_experience_ids=tuple(filter(None, (_item_value(value, "id") for value in retrieved))),
            batch_id=batch_id, correct=outcome.correct,
        )

    @staticmethod
    def _new_rule(attribution, sentiment, batch_id: int) -> GeneralizedExperience:
        raw = f"{sentiment}\x1f{attribution.candidate_rule.casefold()}"
        rule_id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        return GeneralizedExperience(
            id=rule_id, semantic=attribution.candidate_rule, sentiment=sentiment,
            rule=attribution.candidate_rule, corrected_reason=attribution.corrected_reason,
            error_types=(attribution.error_type,), scope_languages=attribution.scope_languages,
            scope_sources=attribution.scope_sources, phenomena=attribution.phenomena,
            created_batch=batch_id, last_updated_batch=batch_id,
        )

    @staticmethod
    def _merge_scope(rule: GeneralizedExperience, attribution, batch_id: int) -> GeneralizedExperience:
        return rule.model_copy(update={
            "error_types": tuple(sorted(set(rule.error_types) | {attribution.error_type})),
            "scope_languages": tuple(sorted(set(rule.scope_languages) | set(attribution.scope_languages))),
            "scope_sources": tuple(sorted(set(rule.scope_sources) | set(attribution.scope_sources))),
            "phenomena": tuple(sorted(set(rule.phenomena) | set(attribution.phenomena))),
            "last_updated_batch": batch_id,
            "version": rule.version + 1,
        })


def _item_value(item, name: str):
    value = getattr(item, name, None)
    if value is None and hasattr(item, "experience"):
        value = getattr(item.experience, name, None)
    return value
