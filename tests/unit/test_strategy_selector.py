import random

from sentiment_agent.strategies.selector import EpsilonGreedySelector


def test_selector_updates_only_selected_language() -> None:
    selector = EpsilonGreedySelector(["direct", "memory"], epsilon=0.0)

    selector.update("th", "memory", 1.0)

    assert selector.stats("th")["memory"].count == 1
    assert selector.stats("vi")["memory"].count == 0


def test_selector_exploits_highest_mean_reward() -> None:
    selector = EpsilonGreedySelector(["direct", "memory"], epsilon=0.0)
    selector.update("th", "direct", 0.0)
    selector.update("th", "memory", 1.0)

    assert selector.select("th", random.Random(42)) == "memory"


def test_selector_tie_breaks_by_configured_order() -> None:
    selector = EpsilonGreedySelector(["direct", "memory"], epsilon=0.0)

    assert selector.select("th", random.Random(42)) == "direct"


def test_selector_explores_with_seeded_rng() -> None:
    selector = EpsilonGreedySelector(["direct", "memory"], epsilon=1.0)

    assert selector.select("th", random.Random(0)) == "memory"
