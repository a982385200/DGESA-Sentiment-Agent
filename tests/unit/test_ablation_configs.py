from sentiment_agent.experiments.factories import ablation_conditions, baseline_conditions


def test_all_ablation_names_are_unique() -> None:
    conditions = ablation_conditions()

    assert len({condition.name for condition in conditions}) == len(conditions)


def test_ablation_suite_covers_each_core_component() -> None:
    names = {condition.name for condition in ablation_conditions()}

    assert names == {
        "full",
        "no_reflection",
        "no_cross_lingual",
        "no_error_experience",
        "no_reliability_filter",
        "fixed_strategy",
        "case_only",
    }


def test_baselines_have_unique_method_configuration() -> None:
    conditions = baseline_conditions()

    signatures = {
        (
            condition.use_memory,
            condition.reflection_enabled,
            condition.dynamic_strategy,
            condition.few_shot,
        )
        for condition in conditions
    }
    assert len(signatures) == len(conditions)
