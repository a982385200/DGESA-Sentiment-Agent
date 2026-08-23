from __future__ import annotations

from sentiment_agent.schemas import StrictModel


class ExperimentCondition(StrictModel):
    name: str
    use_memory: bool = True
    reflection_enabled: bool = True
    cross_lingual_enabled: bool = True
    dynamic_strategy: bool = True
    few_shot: bool = False
    store_error_experiences: bool = True
    reliability_filter_enabled: bool = True
    generalized_rules_enabled: bool = True


def baseline_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(
            name="zero_shot",
            use_memory=False,
            reflection_enabled=False,
            cross_lingual_enabled=False,
            dynamic_strategy=False,
        ),
        ExperimentCondition(
            name="few_shot",
            use_memory=False,
            reflection_enabled=False,
            cross_lingual_enabled=False,
            dynamic_strategy=False,
            few_shot=True,
        ),
        ExperimentCondition(
            name="static_rag",
            use_memory=True,
            reflection_enabled=False,
            cross_lingual_enabled=False,
            dynamic_strategy=False,
        ),
        ExperimentCondition(
            name="memory",
            use_memory=True,
            reflection_enabled=False,
            cross_lingual_enabled=True,
            dynamic_strategy=True,
        ),
        ExperimentCondition(name="full"),
    ]


def ablation_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(name="full"),
        ExperimentCondition(name="no_reflection", reflection_enabled=False),
        ExperimentCondition(name="no_cross_lingual", cross_lingual_enabled=False),
        ExperimentCondition(name="no_error_experience", store_error_experiences=False),
        ExperimentCondition(name="no_reliability_filter", reliability_filter_enabled=False),
        ExperimentCondition(name="fixed_strategy", dynamic_strategy=False),
        ExperimentCondition(name="case_only", generalized_rules_enabled=False),
    ]
