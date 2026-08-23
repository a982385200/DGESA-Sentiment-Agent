# Sentiment Agent Research Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, API-only research framework that evaluates feedback-driven experience evolution, cross-lingual transfer, reflection, and dynamic strategy selection for five ASEAN sentiment datasets.

**Architecture:** A single-process Python package separates prediction from feedback learning so test labels cannot enter memory. Components communicate through Pydantic models; SQLite stores structured experiences; an OpenAI-compatible HTTP client supplies classification, reflection, translation, and embeddings; YAML-driven experiment runners produce auditable JSONL and JSON artifacts.

**Tech Stack:** Python 3.12, Pydantic 2, pydantic-settings, httpx, PyYAML, NumPy, scikit-learn, Typer, pytest, pytest-httpx, pytest-cov, SQLite.

## Global Constraints

- Implement only OpenAI-compatible API inference; do not implement local Hugging Face models.
- `predict()` never receives a gold label; `learn()` is unavailable to evaluation streams.
- Test labels and test texts are never persisted to the experience database.
- Every public module requires isolated unit tests; the complete prediction-feedback-evaluation workflow requires an offline integration test.
- Default tests never access the network and never require an API key.
- Macro-F1 is the primary metric; also record Accuracy, per-class metrics, latency, tokens, failures, and estimated cost.
- Every experiment saves its expanded configuration, configuration hash, predictions, metrics, costs, errors, and memory snapshot.
- Use deterministic seed `42` unless an experiment configuration explicitly overrides it.

---

### Task 1: Package foundation, configuration, schemas, and dataset boundaries

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/sentiment_agent/__init__.py`
- Create: `src/sentiment_agent/config.py`
- Create: `src/sentiment_agent/schemas.py`
- Create: `src/sentiment_agent/data/loader.py`
- Create: `src/sentiment_agent/data/stream.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_schemas.py`
- Test: `tests/unit/test_data.py`

**Interfaces:**
- Produces: `AppConfig.load(path: Path) -> AppConfig`
- Produces: `load_jsonl(path: Path, split: Split) -> list[SentimentExample]`
- Produces: `prediction_input(example: SentimentExample) -> PredictionInput`
- Produces: `TrainingStream` and `EvaluationStream`; only `TrainingStream.feedback(...)` creates `Feedback`.

- [ ] **Step 1: Write failing configuration and schema tests**

```python
def test_prediction_input_cannot_contain_gold_label():
    with pytest.raises(ValidationError):
        PredictionInput(id="vi-1", text="tốt", language="vi", source="x", label="positive")

def test_config_hash_is_stable(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\nmodel:\n  name: test-model\n", encoding="utf-8")
    assert AppConfig.load(path).config_hash == AppConfig.load(path).config_hash
```

- [ ] **Step 2: Run tests and verify import failures**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_schemas.py -q`

Expected: collection fails because `sentiment_agent.config` and `sentiment_agent.schemas` do not exist.

- [ ] **Step 3: Implement strict Pydantic schemas and YAML configuration**

Implement `SentimentLabel = Literal["negative", "neutral", "positive"]`, `Split`, `SentimentExample`, `PredictionInput`, `Prediction`, `Feedback`, `Experience`, `Usage`, `ModelConfig`, `MemoryConfig`, `StrategyConfig`, `ExperimentConfig`, and `AppConfig`. Forbid extra fields on boundary models. Compute `config_hash` as SHA-256 of canonical JSON excluding the hash property. Resolve API keys at client construction time, never during config parsing.

- [ ] **Step 4: Write and run dataset leakage tests**

```python
def test_evaluation_stream_has_no_feedback_method(example):
    stream = EvaluationStream([example])
    item = next(iter(stream))
    assert item.model_dump().keys().isdisjoint({"label"})
    assert not hasattr(stream, "feedback")

def test_training_feedback_requires_matching_example(example):
    stream = TrainingStream([example])
    item = next(iter(stream))
    with pytest.raises(ValueError, match="sample id"):
        stream.feedback(item, predicted="positive", sample_id="wrong")
```

Run: `python -m pytest tests/unit/test_data.py -q`

Expected after implementation: all tests pass.

- [ ] **Step 5: Commit foundation**

```bash
git add pyproject.toml .gitignore .env.example src tests/unit/test_config.py tests/unit/test_schemas.py tests/unit/test_data.py
git commit -m "feat: add typed experiment foundation"
```

### Task 2: OpenAI-compatible client, structured parsing, retry, and cache

**Files:**
- Create: `src/sentiment_agent/llm/client.py`
- Create: `src/sentiment_agent/llm/parsing.py`
- Create: `src/sentiment_agent/llm/cache.py`
- Test: `tests/unit/test_llm_client.py`
- Test: `tests/unit/test_llm_parsing.py`
- Test: `tests/unit/test_cache.py`

**Interfaces:**
- Consumes: `ModelConfig`, `Prediction`, `Usage`
- Produces: `OpenAICompatibleClient.chat_json(messages, response_model) -> BaseModel`
- Produces: `OpenAICompatibleClient.embed(texts: Sequence[str]) -> list[list[float]]`
- Produces: `SQLiteResponseCache.get(key)` and `put(key, response)`.

- [ ] **Step 1: Write failing parser, cache, and HTTP tests**

```python
def test_parser_extracts_fenced_json():
    parsed = parse_model_json('```json\n{"label":"positive","confidence":0.9,"reason":"clear"}\n```', PredictionPayload)
    assert parsed.label == "positive"

def test_client_retries_429_then_returns(httpx_mock, client):
    httpx_mock.add_response(status_code=429, json={"error": "limited"})
    httpx_mock.add_response(json={"choices":[{"message":{"content":"{\\"label\\":\\"neutral\\",\\"confidence\\":0.7,\\"reason\\":\\"mixed\\"}"}}],"usage":{"prompt_tokens":5,"completion_tokens":4}})
    result = client.classify("ข้อความ", "th")
    assert result.payload.label == "neutral"
    assert result.attempts == 2
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest tests/unit/test_llm_parsing.py tests/unit/test_cache.py tests/unit/test_llm_client.py -q`

Expected: imports fail.

- [ ] **Step 3: Implement minimal HTTP boundary**

Use one injected `httpx.Client`, POST `${base_url}/chat/completions` and `${base_url}/embeddings`, exponential delays `0.5 * 2**attempt`, and retry only 408, 409, 429, and 5xx responses. Cache keys are SHA-256 hashes of canonical model parameters plus messages. Raise `LLMRequestError` after the configured attempt count and preserve status and a redacted response excerpt.

- [ ] **Step 4: Run client tests**

Run: `python -m pytest tests/unit/test_llm_parsing.py tests/unit/test_cache.py tests/unit/test_llm_client.py -q`

Expected: all tests pass without network access.

- [ ] **Step 5: Commit API layer**

```bash
git add src/sentiment_agent/llm tests/unit/test_llm_client.py tests/unit/test_llm_parsing.py tests/unit/test_cache.py
git commit -m "feat: add cached OpenAI-compatible client"
```

### Task 3: Metrics, cost accounting, and auditable result writer

**Files:**
- Create: `src/sentiment_agent/evaluation/metrics.py`
- Create: `src/sentiment_agent/evaluation/evaluator.py`
- Create: `src/sentiment_agent/evaluation/artifacts.py`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_artifacts.py`

**Interfaces:**
- Produces: `compute_metrics(gold, predicted) -> MetricsReport`
- Produces: `ArtifactWriter.append_prediction(record)`, `write_metrics(report)`, and `write_manifest(config)`.

- [ ] **Step 1: Write failing metric and artifact tests**

```python
def test_macro_f1_uses_all_three_labels():
    report = compute_metrics(["positive", "negative"], ["positive", "positive"])
    assert report.labels == ["negative", "neutral", "positive"]
    assert report.accuracy == 0.5

def test_artifact_writer_emits_valid_jsonl(tmp_path, prediction_record):
    writer = ArtifactWriter(tmp_path)
    writer.append_prediction(prediction_record)
    assert json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])["sample_id"] == prediction_record.sample_id
```

- [ ] **Step 2: Implement deterministic metrics and atomic JSON writes**

Call `precision_recall_fscore_support(..., labels=["negative","neutral","positive"], zero_division=0)` and `accuracy_score`. Write JSON to a temporary sibling file and replace the destination atomically; append JSONL under a process-local lock.

- [ ] **Step 3: Run and commit**

Run: `python -m pytest tests/unit/test_metrics.py tests/unit/test_artifacts.py -q`

Expected: all tests pass.

```bash
git add src/sentiment_agent/evaluation tests/unit/test_metrics.py tests/unit/test_artifacts.py
git commit -m "feat: add reproducible evaluation artifacts"
```

### Task 4: SQLite experience memory, reliability updates, and retrieval

**Files:**
- Create: `src/sentiment_agent/memory/store.py`
- Create: `src/sentiment_agent/memory/scoring.py`
- Create: `src/sentiment_agent/memory/retrieval.py`
- Test: `tests/unit/test_memory_store.py`
- Test: `tests/unit/test_retrieval.py`

**Interfaces:**
- Consumes: `Experience`, embedding vectors, `MemoryConfig`
- Produces: `ExperienceStore.add_or_update(experience, vector) -> str`
- Produces: `ExperienceStore.record_outcome(experience_id, correct) -> Experience`
- Produces: `ExperienceRetriever.search(query_vector, language, source, cross_lingual, k) -> list[RetrievedExperience]`.

- [ ] **Step 1: Write failing transaction and ranking tests**

```python
def test_duplicate_experience_updates_counts(store, experience):
    first = store.add_or_update(experience, [1.0, 0.0])
    second = store.add_or_update(experience, [1.0, 0.0])
    assert first == second
    assert store.get(first).success_count == 2

def test_cross_lingual_flag_controls_candidates(retriever):
    same = retriever.search([1.0, 0.0], language="vi", source="x", cross_lingual=False, k=10)
    cross = retriever.search([1.0, 0.0], language="vi", source="x", cross_lingual=True, k=10)
    assert all(item.experience.language == "vi" for item in same)
    assert any(item.experience.language != "vi" for item in cross)
```

- [ ] **Step 2: Implement schema and deterministic score**

Create SQLite tables `experiences` and `vectors`, enable foreign keys and WAL, and wrap updates in transactions. Deduplicate with SHA-256 of normalized text, language, source, label, and experience type. Compute `score = semantic_weight*cosine + language_weight*language_match + domain_weight*domain_match + reliability_weight*reliability + recency_weight*recency` and filter below `min_reliability`.

- [ ] **Step 3: Run and commit**

Run: `python -m pytest tests/unit/test_memory_store.py tests/unit/test_retrieval.py -q`

Expected: all tests pass using temporary SQLite files.

```bash
git add src/sentiment_agent/memory tests/unit/test_memory_store.py tests/unit/test_retrieval.py
git commit -m "feat: add reliable experience memory"
```

### Task 5: Prompt strategies, epsilon-greedy selector, and reflection

**Files:**
- Create: `src/sentiment_agent/strategies/base.py`
- Create: `src/sentiment_agent/strategies/prompts.py`
- Create: `src/sentiment_agent/strategies/selector.py`
- Create: `src/sentiment_agent/reflection/reflector.py`
- Test: `tests/unit/test_prompts.py`
- Test: `tests/unit/test_strategy_selector.py`
- Test: `tests/unit/test_reflector.py`

**Interfaces:**
- Produces strategies named `direct`, `translation`, `memory`, and `reflection_verified`.
- Produces `EpsilonGreedySelector.select(language, rng) -> str` and `update(language, strategy, reward)`.
- Produces `Reflector.reflect(input, prediction, feedback, experiences) -> ReflectionResult`.

- [ ] **Step 1: Write failing strategy tests**

```python
def test_memory_prompt_contains_only_reliable_experiences(prompt_builder, input_item, retrieved):
    messages = prompt_builder.build("memory", input_item, retrieved)
    rendered = json.dumps(messages, ensure_ascii=False)
    assert retrieved[0].experience.reason in rendered

def test_selector_updates_only_selected_language():
    selector = EpsilonGreedySelector(["direct", "memory"], epsilon=0.0)
    selector.update("th", "memory", 1.0)
    assert selector.stats("th")["memory"].count == 1
    assert selector.stats("vi")["memory"].count == 0
```

- [ ] **Step 2: Implement immutable prompt builders and selector**

Each strategy returns messages but never calls the API. Require exactly one final sentiment label. Sort retrieved experiences by score, cap at configured `k`, and omit gold labels from the current sample. Use injected `random.Random(seed)` for epsilon-greedy selection and deterministic tie-breaking by configured strategy order.

- [ ] **Step 3: Implement and test reflection**

Reflection accepts feedback only after prediction and returns `error_type`, `corrected_reason`, `generalized_rule`, and `scope`. Disabled reflection creates only case experience; enabled reflection adds a generalized rule when structured parsing succeeds and records a recoverable error otherwise.

Run: `python -m pytest tests/unit/test_prompts.py tests/unit/test_strategy_selector.py tests/unit/test_reflector.py -q`

Expected: all tests pass.

- [ ] **Step 4: Commit strategies and reflection**

```bash
git add src/sentiment_agent/strategies src/sentiment_agent/reflection tests/unit/test_prompts.py tests/unit/test_strategy_selector.py tests/unit/test_reflector.py
git commit -m "feat: add adaptive strategies and reflection"
```

### Task 6: Agent prediction-learning boundary and leakage enforcement

**Files:**
- Create: `src/sentiment_agent/agent/agent.py`
- Test: `tests/unit/test_agent.py`
- Test: `tests/unit/test_leakage.py`

**Interfaces:**
- Consumes: client, retriever, store, prompt builder, selector, reflector.
- Produces: `SentimentAgent.predict(item: PredictionInput) -> Prediction`
- Produces: `SentimentAgent.learn(item: PredictionInput, prediction: Prediction, feedback: Feedback) -> LearningResult`.

- [ ] **Step 1: Write failing orchestration and leakage tests**

```python
def test_predict_never_receives_or_persists_gold(agent, prediction_input, store):
    agent.predict(prediction_input)
    assert store.count() == 0

def test_learn_rejects_mismatched_feedback(agent, prediction_input, prediction):
    feedback = Feedback(sample_id="different", predicted_label=prediction.label, gold_label="negative")
    with pytest.raises(ValueError, match="sample id"):
        agent.learn(prediction_input, prediction, feedback)
```

- [ ] **Step 2: Implement two-phase agent**

`predict` embeds the query, retrieves allowed experiences, selects a strategy, calls the client, and returns complete provenance. `learn` validates IDs, records strategy reward, adds a successful or correction experience using the gold label, optionally reflects, and commits all memory changes in one transaction. Evaluation code never exposes `learn`.

- [ ] **Step 3: Run and commit**

Run: `python -m pytest tests/unit/test_agent.py tests/unit/test_leakage.py -q`

Expected: all tests pass.

```bash
git add src/sentiment_agent/agent tests/unit/test_agent.py tests/unit/test_leakage.py
git commit -m "feat: enforce prediction and learning boundary"
```

### Task 7: Baseline, evolution, transfer, strategy, and ablation runners

**Files:**
- Create: `src/sentiment_agent/experiments/runner.py`
- Create: `src/sentiment_agent/experiments/factories.py`
- Create: `configs/default.yaml`
- Create: `configs/experiments/baseline.yaml`
- Create: `configs/experiments/evolution.yaml`
- Create: `configs/experiments/transfer.yaml`
- Create: `configs/experiments/strategy.yaml`
- Create: `configs/experiments/ablation.yaml`
- Test: `tests/unit/test_experiment_runner.py`
- Test: `tests/unit/test_ablation_configs.py`

**Interfaces:**
- Produces: `ExperimentRunner.run() -> ExperimentSummary`
- Produces conditions `zero_shot`, `few_shot`, `static_rag`, `memory`, `full`; ablations toggle reflection, cross-lingual retrieval, correction experiences, reliability filtering, and dynamic selection.

- [ ] **Step 1: Write failing condition and checkpoint tests**

```python
def test_evolution_evaluates_at_configured_checkpoints(fake_runner):
    summary = fake_runner.run_evolution(checkpoints=[0, 2, 4])
    assert [stage.experience_count for stage in summary.stages] == [0, 2, 4]

def test_all_ablation_names_are_unique(load_ablation_config):
    conditions = load_ablation_config()
    assert len({condition.name for condition in conditions}) == len(conditions)
```

- [ ] **Step 2: Implement shared experiment loop**

Use the same dataset loader, model configuration, parser, retry policy, and evaluator for every condition. Evolution performs prediction then feedback on training items and evaluates without learning at checkpoints. Transfer loops over ordered source-target language pairs and writes a matrix-ready row per pair. Resume from a manifest checkpoint using processed sample IDs.

- [ ] **Step 3: Run and commit**

Run: `python -m pytest tests/unit/test_experiment_runner.py tests/unit/test_ablation_configs.py -q`

Expected: all tests pass with fake clients.

```bash
git add src/sentiment_agent/experiments configs tests/unit/test_experiment_runner.py tests/unit/test_ablation_configs.py
git commit -m "feat: add reproducible research experiment runners"
```

### Task 8: CLI, offline end-to-end workflow, and full verification

**Files:**
- Create: `src/sentiment_agent/cli.py`
- Create: `tests/integration/test_full_workflow.py`
- Create: `tests/fixtures/fake_llm.py`
- Create: `tests/fixtures/tiny_dataset.jsonl`
- Modify: `README.md`

**Interfaces:**
- Produces: `python -m sentiment_agent.cli run --config PATH`
- Produces: a complete offline workflow that exercises load, predict, evaluate, feedback, reflection, memory update, checkpoint evaluation, resume, and artifact generation.

- [ ] **Step 1: Write the failing full-flow test**

```python
def test_full_offline_research_workflow(tmp_path, fake_api_server, tiny_config):
    summary = run_from_config(tiny_config, output_root=tmp_path)
    assert summary.processed_training_samples == 6
    assert summary.evaluation_samples == 3
    assert summary.memory_experiences > 0
    assert (summary.output_dir / "config.yaml").exists()
    assert (summary.output_dir / "manifest.json").exists()
    assert (summary.output_dir / "predictions.jsonl").exists()
    assert (summary.output_dir / "metrics.json").exists()
    assert (summary.output_dir / "costs.json").exists()
    assert not any(row["split"] == "test" for row in read_memory_rows(summary.memory_path))
```

- [ ] **Step 2: Implement CLI and fixture API**

The fake API returns deterministic classification, embedding, translation, and reflection responses based on request content. The CLI supports `run`, `validate-config`, and `summarize`; exits nonzero for missing API key in real mode, invalid dataset paths, excessive API failure rate, or invalid structured output after retries.

- [ ] **Step 3: Run the complete integration test**

Run: `python -m pytest tests/integration/test_full_workflow.py -v`

Expected: one complete workflow passes with no external network requests.

- [ ] **Step 4: Run all tests and coverage**

Run: `python -m pytest --cov=sentiment_agent --cov-report=term-missing --cov-fail-under=85`

Expected: all unit and integration tests pass and total statement coverage is at least 85%.

- [ ] **Step 5: Verify the CLI help and configuration validation**

Run: `python -m sentiment_agent.cli --help`

Expected: commands `run`, `validate-config`, and `summarize` are listed.

Run: `python -m sentiment_agent.cli validate-config --config configs/experiments/evolution.yaml`

Expected: exit code 0 and the configuration hash is printed without revealing an API key.

- [ ] **Step 6: Document and commit the finished framework**

README must include installation, environment variables, dataset layout, dry-run command, each formal experiment command, output artifact descriptions, leakage guarantees, test command, and a warning that formal comparisons must keep model and decoding parameters fixed.

```bash
git add src/sentiment_agent/cli.py tests/integration tests/fixtures README.md
git commit -m "feat: complete tested sentiment research workflow"
```

### Task 9: Final research reproducibility audit

**Files:**
- Modify only files implicated by failed checks.

- [ ] **Step 1: Confirm repository hygiene**

Run: `git status --short`, `git ls-files | rg "(^|/)(\.env|outputs|.*\.sqlite3?)$"`, and `rg -l -i "(api[_-]?key|access[_-]?token|password\\s*[:=])" --glob '!.git/**'`.

Expected: no secret or generated artifact is tracked; only deliberate implementation changes are present.

- [ ] **Step 2: Confirm deterministic offline execution twice**

Run the integration workflow twice with seed 42 into separate temporary directories and compare `metrics.json`, the strategy-selection sequence, and memory experience IDs.

Expected: all deterministic artifacts match; timestamps and output paths may differ.

- [ ] **Step 3: Run final verification suite**

Run: `python -m pytest -q --cov=sentiment_agent --cov-report=term-missing --cov-fail-under=85`.

Expected: zero failures and coverage at least 85%.

- [ ] **Step 4: Commit audit fixes if required**

```bash
git add src tests configs README.md pyproject.toml .gitignore .env.example
git commit -m "test: harden research reproducibility"
```

If no audit fixes are required, do not create an empty commit.
