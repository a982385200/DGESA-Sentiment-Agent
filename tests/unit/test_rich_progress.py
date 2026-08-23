from io import StringIO

from rich.console import Console

from sentiment_agent.experiments.progress import ProgressEvent
from sentiment_agent.reporting.progress import RichProgressReporter


def test_rich_reporter_renders_stage_counts_tokens_and_experiences() -> None:
    output = StringIO()
    reporter = RichProgressReporter(console=Console(file=output, force_terminal=False, width=140))
    reporter.update(ProgressEvent(
        stage="train", completed_samples=8, total_samples=16,
        completed_batches=1, total_batches=2, successful_requests=8,
        failed_requests=0, input_tokens=120, output_tokens=30,
        elapsed_seconds=2.0, experience_count=7, checkpoint=None,
    ))
    reporter.close()
    rendered = output.getvalue()
    assert "train" in rendered
    assert "8/16" in rendered
    assert "batch 1/2" in rendered
    assert "tokens 150" in rendered
    assert "experiences 7" in rendered
