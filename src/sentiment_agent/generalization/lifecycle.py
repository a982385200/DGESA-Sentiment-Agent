from __future__ import annotations

from sentiment_agent.generalization.models import GeneralizedExperience, RuleStatus


class LifecyclePolicy:
    def __init__(self, *, minimum_support: int = 2, minimum_batches: int = 2,
                 maximum_contradiction_ratio: float = 0.2,
                 minimum_active_reliability: float = 0.6) -> None:
        self.minimum_support = minimum_support
        self.minimum_batches = minimum_batches
        self.maximum_contradiction_ratio = maximum_contradiction_ratio
        self.minimum_active_reliability = minimum_active_reliability

    def next_status(self, rule: GeneralizedExperience) -> RuleStatus:
        evidence = rule.support_count + rule.contradiction_count
        if evidence >= 3 and rule.contradiction_ratio > 0.5:
            return "suppressed"
        if rule.status == "active" and rule.contradiction_ratio > self.maximum_contradiction_ratio:
            return "conflicted"
        if (rule.support_count >= self.minimum_support and
                len(rule.supporting_batches) >= self.minimum_batches and
                rule.contradiction_ratio <= self.maximum_contradiction_ratio and
                rule.reliability >= self.minimum_active_reliability):
            return "active"
        return "candidate"
