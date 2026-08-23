from __future__ import annotations

import asyncio
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

from sentiment_agent.agent.sentiment_agent import SentimentAgent
from sentiment_agent.config import config_hash, load_config, redacted_config
from sentiment_agent.data.loader import load_examples
from sentiment_agent.data.fingerprint import fingerprint_file
from sentiment_agent.embeddings.local_bge import DisabledEmbedding, LocalBGEEmbedding
from sentiment_agent.experience.repository import ExperienceRepository
from sentiment_agent.experience.retrieval import ExperienceRetriever, RetrievalWeights
from sentiment_agent.experience.updater import ExperienceUpdater
from sentiment_agent.experience.vector_index import VectorIndex
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.experiments.runner import ExperimentRunner
from sentiment_agent.llm.langchain_qwen import LangChainQwenBackend
from sentiment_agent.prompts.prediction import PredictionPromptBuilder

app = typer.Typer(help="Self-evolving multilingual sentiment experiments.")
experience_app = typer.Typer(help="Inspect and export an experiment experience store.")
app.add_typer(experience_app, name="experience")


def _repository(run: Path) -> ExperienceRepository:
    return ExperienceRepository(run / "experience_store" / "experiences.sqlite3")


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    loaded = load_config(config)
    typer.echo(f"Configuration valid: {config_hash(loaded)}")


@app.command()
def run(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    load_dotenv()
    settings = load_config(config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + config_hash(settings)[:8]
    run_dir = settings.experiment.output_root / run_id
    writer = ArtifactWriter(run_dir)
    writer.write_json("resolved_config.json", redacted_config(settings))
    store_dir = run_dir / "experience_store"
    repository = ExperienceRepository(store_dir / "experiences.sqlite3")
    index = VectorIndex(store_dir)
    embedding = (
        LocalBGEEmbedding(model_id=settings.embedding.model_id,
                          device=settings.embedding.device,
                          batch_size=settings.embedding.batch_size)
        if settings.retrieval.enabled
        else DisabledEmbedding()
    )
    llm = LangChainQwenBackend(
        model_name=settings.model.name, base_url=str(settings.model.base_url),
        api_key_env=settings.model.api_key_env, temperature=settings.model.temperature,
        max_tokens=settings.model.max_tokens, timeout_seconds=settings.model.timeout_seconds,
        max_retries=settings.model.max_retries,
    )
    weights = RetrievalWeights(
        semantic=settings.retrieval.semantic_weight, language=settings.retrieval.language_weight,
        source=settings.retrieval.source_weight, reliability=settings.retrieval.reliability_weight)
    agent = SentimentAgent(
        embedding=embedding, llm=llm,
        retriever=ExperienceRetriever(repository, weights,
            minimum_reliability=settings.retrieval.minimum_reliability,
            cross_lingual=settings.retrieval.cross_lingual),
        updater=ExperienceUpdater(repository, index), vector_index=index,
        prompt_builder=PredictionPromptBuilder(), model_name=settings.model.name,
        retrieval_k=settings.retrieval.k if settings.retrieval.enabled else 0,
    )
    runner = ExperimentRunner(agent=agent, writer=writer,
        batch_size=settings.experiment.train_batch_size,
        concurrency=settings.model.concurrency, checkpoints=settings.experiment.checkpoints,
        manifest_metadata={
            "config_hash": config_hash(settings),
            "model": settings.model.name,
            "embedding_model": settings.embedding.model_id,
            "seed": settings.experiment.seed,
            "dataset_fingerprints": {
                str(path): fingerprint_file(path)
                for path in (settings.experiment.train_paths + settings.experiment.dev_paths
                             + settings.experiment.test_paths)
            },
        })
    train = [item for path in settings.experiment.train_paths for item in load_examples(path)]
    dev = [item for path in settings.experiment.dev_paths for item in load_examples(path)]
    test = [item for path in settings.experiment.test_paths for item in load_examples(path)]
    try:
        summary = asyncio.run(runner.run(train, dev, test))
    finally:
        repository.close()
    typer.echo(f"Completed {summary.completed_samples} training samples: {run_dir}")


@app.command()
def summarize(run: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    typer.echo((run / "metrics.json").read_text(encoding="utf-8"))


@experience_app.command("list")
def list_experiences(run: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    with _repository(run) as repository:
        for experience in repository.list():
            typer.echo(f"{experience.id}\t{experience.language}\t{experience.sentiment}\t{experience.reliability:.3f}")


@experience_app.command()
def show(run: Path = typer.Option(..., exists=True, file_okay=False),
         id: str = typer.Option(...)) -> None:
    with _repository(run) as repository:
        typer.echo(repository.get(id).model_dump_json(indent=2))


@experience_app.command()
def stats(run: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    with _repository(run) as repository:
        rows = repository.list()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.language] = counts.get(row.language, 0) + 1
    typer.echo(json.dumps({"total": len(rows), "by_language": counts}, ensure_ascii=False))


@experience_app.command()
def history(run: Path = typer.Option(..., exists=True, file_okay=False),
            id: str = typer.Option(...)) -> None:
    with _repository(run) as repository:
        for event in repository.history(id):
            typer.echo(event.model_dump_json())


@experience_app.command("export")
def export_experiences(run: Path = typer.Option(..., exists=True, file_okay=False),
                       format: str = typer.Option("csv")) -> None:
    with _repository(run) as repository:
        rows = [item.model_dump(mode="json") for item in repository.list()]
    if format == "jsonl":
        target = run / "experience-export.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    elif format == "csv":
        target = run / "experience-export.csv"
        fields = list(rows[0]) if rows else list(__import__("sentiment_agent.schemas", fromlist=["Experience"]).Experience.model_fields)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise typer.BadParameter("format must be csv or jsonl")
    typer.echo(str(target))


if __name__ == "__main__":
    app()
