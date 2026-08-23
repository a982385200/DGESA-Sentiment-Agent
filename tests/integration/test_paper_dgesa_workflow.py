import pytest

from tests.unit.test_dgesa_agent import example, make_agent
from sentiment_agent.dgesa.repository import DGESARepository


@pytest.mark.anyio
async def test_complete_paper_dgesa_workflow(tmp_path):
    with DGESARepository(tmp_path / "paper.sqlite3") as repo:
        agent = make_agent(repo)
        training = await agent.train([example("train-1"), example("train-2")])
        evaluation = await agent.evaluate([example("test-1")])
        pattern = repo.list_patterns()[0]
        assert [item.sentiment for item in training] == ["positive", "negative"]
        assert evaluation.metrics["macro_f1"] == pytest.approx(1 / 3)
        assert evaluation.predictions[0].pattern_experience_ids
        assert pattern.support_count == 2
