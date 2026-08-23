from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

from sentiment_agent.config import config_hash, load_config, redacted_config
from sentiment_agent.data.loader import load_examples
from sentiment_agent.embeddings.local_bge import LocalBGEEmbedding
from sentiment_agent.experiments.artifacts import ArtifactWriter
from sentiment_agent.llm.langchain_qwen import LangChainQwenBackend
from sentiment_agent.dgesa.agent import PaperDGESA
from sentiment_agent.dgesa.evolution import DGESAEvolutionService, DGESAParameters
from sentiment_agent.dgesa.llm import LangChainAppendixLLM
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder
from sentiment_agent.dgesa.repository import DGESARepository
from sentiment_agent.dgesa.retrieval import PatternRetriever, SampleRetriever

app = typer.Typer(help="Self-evolving multilingual sentiment experiments.")


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    loaded = load_config(config)
    typer.echo(f"Configuration valid: {config_hash(loaded)}")


@app.command("run-paper")
def run_paper(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Run the paper-aligned DGESA train/test protocol."""
    load_dotenv()
    settings = load_config(config)
    if not settings.dgesa.enabled:
        raise typer.BadParameter("run-paper requires dgesa.enabled=true")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-paper-" + config_hash(settings)[:8]
    run_dir = settings.experiment.output_root / run_id
    writer = ArtifactWriter(run_dir)
    writer.write_json("resolved_config.json", redacted_config(settings))
    repository = DGESARepository(run_dir / "experience_store" / "dgesa.sqlite3")
    try:
        embedding = LocalBGEEmbedding(
            model_id=settings.embedding.model_id, device=settings.embedding.device,
            batch_size=settings.embedding.batch_size)
        backend = LangChainQwenBackend(
            model_name=settings.model.name, base_url=str(settings.model.base_url),
            api_key_env=settings.model.api_key_env, temperature=settings.model.temperature,
            max_tokens=settings.model.max_tokens, timeout_seconds=settings.model.timeout_seconds,
            max_retries=settings.model.max_retries,
            enable_thinking=settings.model.enable_thinking)
        llm = LangChainAppendixLLM(backend.chat_model)
        prompts = AppendixPromptBuilder()
        weights = settings.dgesa.score_weights
        evolution = DGESAEvolutionService(
            repository, embedding, llm, prompts,
            DGESAParameters(
                admission_candidates=settings.dgesa.admission_candidates,
                coverage_temperature=settings.dgesa.coverage_temperature,
                low_coverage_threshold=settings.dgesa.low_coverage_threshold,
                high_coverage_threshold=settings.dgesa.high_coverage_threshold,
                alignment_candidates=settings.dgesa.alignment_candidates,
                minimum_active_reliability=settings.dgesa.minimum_active_reliability,
                maximum_conflict_ratio=settings.dgesa.maximum_conflict_ratio,
                minimum_global_languages=settings.dgesa.minimum_global_languages,
                minimum_language_support=settings.dgesa.minimum_language_support,
                max_generation_attempts=settings.dgesa.max_generation_attempts,
            ))
        agent = PaperDGESA(
            embedding=embedding, llm=llm, prompts=prompts,
            sample_retriever=SampleRetriever(
                repository, minimum_similarity=settings.dgesa.sample_similarity_threshold),
            pattern_retriever=PatternRetriever(
                repository, semantic_weight=weights[0], reliability_weight=weights[1],
                conflict_weight=weights[2],
                minimum_reliability=settings.dgesa.minimum_active_reliability,
                maximum_conflict_ratio=settings.dgesa.maximum_conflict_ratio,
                local_similarity=settings.dgesa.local_similarity_threshold,
                minimum_language_support=settings.dgesa.minimum_language_support),
            evolution=evolution, sample_k=settings.dgesa.sample_retrieval_k,
            pattern_k=settings.dgesa.pattern_retrieval_k)
        train = [item for path in settings.experiment.train_paths for item in load_examples(path)]
        test = [item for path in settings.experiment.test_paths for item in load_examples(path)]
        if settings.experiment.train_limit is not None:
            train = train[:settings.experiment.train_limit]
        if settings.experiment.test_limit is not None:
            test = test[:settings.experiment.test_limit]

        async def execute_paper():
            training = await agent.train(train)
            evaluation = await agent.evaluate(test)
            return training, evaluation

        training, evaluation = asyncio.run(execute_paper())
        for split, predictions in (("train", training), ("test", evaluation.predictions)):
            for prediction in predictions:
                writer.append_jsonl(f"{split}_predictions.jsonl", prediction.model_dump(mode="json"))
        for experience in repository.list_samples():
            writer.append_jsonl("sample_experiences.jsonl", experience.model_dump(mode="json"))
        for experience in repository.list_patterns():
            writer.append_jsonl("pattern_experiences.jsonl", experience.model_dump(mode="json"))
        writer.write_json("metrics.json", evaluation.metrics)
        writer.write_json("manifest.json", {
            "status": "completed", "method": "DGESA-paper-aligned",
            "training_samples": len(train), "test_samples": len(test),
            "sample_experiences": len(repository.list_samples()),
            "pattern_experiences": len(repository.list_patterns()),
        })
    finally:
        repository.close()
    typer.echo(f"Completed paper-aligned DGESA run: {run_dir}")


@app.command()
def summarize(run: Path = typer.Option(..., exists=True, file_okay=False)) -> None:
    typer.echo((run / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
