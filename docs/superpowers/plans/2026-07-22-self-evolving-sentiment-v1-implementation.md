# Self-Evolving Sentiment V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible mini-batch sentiment experiment loop in which fixed-prompt Qwen predictions use locally embedded SQLite-backed experiences created from earlier batches.

**Architecture:** Strongly typed boundary models isolate data, LLM, embedding, experience, agent, evaluation, and experiment concerns. The experiment runner freezes experience visibility per batch, predicts concurrently, then applies feedback in input order; evaluation receives a read-only prediction path and never learns.

**Tech Stack:** Python 3.12, uv, Pydantic 2, pydantic-settings, PyYAML, LangChain `ChatOpenAI`, ModelScope/SentenceTransformers BGE-M3, NumPy, SQLite, scikit-learn, Typer, pytest.

## Global Constraints

- Use `uv` as the only Python environment and dependency manager.
- Add a failing test before production code.
- Every public function or method requires an isolated unit test.
- Default tests must not call external APIs, load real credentials, or download models.
- Run the complete offline integration workflow before claiming completion.
- The base prompt and foundation-model parameters remain fixed within a comparison.
- Training labels become visible only after all predictions in the batch succeed.
- Experiences created by a batch become visible only to later batches.
- Development and test evaluation never writes experiences.
- Preserve unrelated user changes already present in the worktree.

---

## File Map

- `src/sentiment_agent/schemas.py`: immutable boundary models and enums.
- `src/sentiment_agent/config.py`: YAML loading, environment references, validation, hashing, redaction.
- `src/sentiment_agent/data/loader.py`: JSON/JSONL loading and validation.
- `src/sentiment_agent/data/fingerprint.py`: reproducible SHA-256 dataset fingerprints.
- `src/sentiment_agent/embeddings/base.py`: embedding protocol.
- `src/sentiment_agent/embeddings/local_bge.py`: lazy local BGE-M3 adapter.
- `src/sentiment_agent/experience/repository.py`: SQLite schema and transactional experience/event/outcome persistence.
- `src/sentiment_agent/experience/vector_index.py`: atomic NumPy vector matrix and ID map.
- `src/sentiment_agent/experience/retrieval.py`: immutable-snapshot exact retrieval and score decomposition.
- `src/sentiment_agent/experience/updater.py`: deterministic experience creation, merge, reinforcement, and penalty.
- `src/sentiment_agent/prompts/prediction.py`: fixed prompt and experience rendering.
- `src/sentiment_agent/llm/base.py`: async classification protocol.
- `src/sentiment_agent/llm/langchain_qwen.py`: LangChain Qwen implementation and structured parsing.
- `src/sentiment_agent/llm/cache.py`: SQLite response cache.
- `src/sentiment_agent/agent/sentiment_agent.py`: batch prediction and ordered learning coordination.
- `src/sentiment_agent/evaluation/metrics.py`: classification metrics.
- `src/sentiment_agent/experiments/artifacts.py`: atomic run artifact writers.
- `src/sentiment_agent/experiments/runner.py`: training/checkpoint/evaluation loop.
- `src/sentiment_agent/cli.py`: config validation, run, summarize, and experience inspection commands.
- `tests/fixtures/`: deterministic fakes and tiny datasets.
- `tests/unit/`: isolated public-interface tests.
- `tests/integration/test_offline_evolution_workflow.py`: complete no-network workflow.

### Task 1: Package, schemas, and configuration

**Files:**
- Create: `src/sentiment_agent/__init__.py`
- Create: `src/sentiment_agent/schemas.py`
- Create: `src/sentiment_agent/config.py`
- Create: `configs/default.yaml`
- Create: `.env.example`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_schemas.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `SentimentExample`, `PredictionInput`, `Prediction`, `Feedback`, `Experience`, `RetrievedExperience`, `ExperimentConfig`, `load_config(path)`, `config_hash(config)`, `redacted_config(config)`.

- [ ] **Step 1: Write failing schema and config tests**

```python
def test_prediction_input_cannot_receive_label():
    with pytest.raises(ValidationError):
        PredictionInput(id="x", text="ดี", language="th", source="tiny", label="positive")

def test_load_config_rejects_nonpositive_batch_size(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment:\n  train_batch_size: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train_batch_size"):
        load_config(path)

def test_redacted_config_never_contains_secret(monkeypatch, valid_config):
    monkeypatch.setenv("QWEN_API_KEY", "super-secret")
    assert "super-secret" not in json.dumps(redacted_config(valid_config))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/unit/test_schemas.py tests/unit/test_config.py -v`

Expected: collection fails because `sentiment_agent.schemas` and `sentiment_agent.config` do not exist.

- [ ] **Step 3: Implement strict models and validated configuration**

Use `ConfigDict(extra="forbid", frozen=True)`, `Literal["positive", "neutral", "negative"]`, language literals, positive integer constraints, explicit nested model/embedding/retrieval/experiment settings, YAML loading, canonical JSON hashing, and dictionary redaction. Configuration stores only `api_key_env`, never the secret value.

```python
class PredictionInput(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language
    source: str = Field(min_length=1)

class Feedback(StrictModel):
    sample_id: str
    predicted_label: SentimentLabel
    gold_label: SentimentLabel
    correct: bool

    @model_validator(mode="after")
    def validate_correct(self):
        if self.correct != (self.predicted_label == self.gold_label):
            raise ValueError("correct must match predicted and gold labels")
        return self
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_schemas.py tests/unit/test_config.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml uv.lock .env.example configs/default.yaml src/sentiment_agent tests/unit/test_schemas.py tests/unit/test_config.py
git commit -m "feat: add typed experiment configuration"
```

### Task 2: Dataset loading and fingerprints

**Files:**
- Create: `src/sentiment_agent/data/__init__.py`
- Create: `src/sentiment_agent/data/loader.py`
- Create: `src/sentiment_agent/data/fingerprint.py`
- Test: `tests/unit/test_data_loader.py`
- Test: `tests/unit/test_data_fingerprint.py`

**Interfaces:**
- Consumes: `SentimentExample`.
- Produces: `load_examples(path: Path) -> list[SentimentExample]`, `without_labels(examples) -> list[PredictionInput]`, `fingerprint_file(path: Path) -> str`.

- [ ] **Step 1: Write failing tests for JSON arrays, JSONL, validation, label removal, and stable hashing**

```python
def test_without_labels_returns_prediction_inputs(example):
    result = without_labels([example])
    assert result == [PredictionInput(id=example.id, text=example.text, language=example.language, source=example.source)]

def test_duplicate_ids_are_rejected(tmp_path):
    path = write_json(tmp_path, [ROW, ROW])
    with pytest.raises(ValueError, match="duplicate sample id"):
        load_examples(path)

def test_fingerprint_changes_with_content(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("[]", encoding="utf-8")
    first = fingerprint_file(path)
    path.write_text("[1]", encoding="utf-8")
    assert fingerprint_file(path) != first
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/test_data_loader.py tests/unit/test_data_fingerprint.py -v`

Expected: imports fail for missing data modules.

- [ ] **Step 3: Implement loaders and streaming SHA-256**

Detect a JSON array versus line-delimited JSON, validate every record with row context, reject duplicate IDs, and never expose labels through `without_labels`.

- [ ] **Step 4: Verify focused tests pass**

Run: `uv run pytest tests/unit/test_data_loader.py tests/unit/test_data_fingerprint.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/data tests/unit/test_data_loader.py tests/unit/test_data_fingerprint.py
git commit -m "feat: add validated sentiment datasets"
```

### Task 3: Local embedding boundary and BGE-M3 adapter

**Files:**
- Create: `src/sentiment_agent/embeddings/__init__.py`
- Create: `src/sentiment_agent/embeddings/base.py`
- Create: `src/sentiment_agent/embeddings/local_bge.py`
- Create: `tests/fixtures/fake_embedding.py`
- Test: `tests/unit/test_local_embedding.py`

**Interfaces:**
- Produces: `EmbeddingBackend.embed(texts: Sequence[str]) -> np.ndarray`, `LocalBGEEmbedding.embed(...)` returning normalized `float32` shape `(n, dimension)`.

- [ ] **Step 1: Write failing tests with an injected encoder**

```python
def test_local_embedding_batches_and_normalizes():
    encoder = FakeEncoder([[3.0, 4.0], [0.0, 2.0]])
    backend = LocalBGEEmbedding(model_id="local/test", encoder=encoder, batch_size=8)
    vectors = backend.embed(["a", "b"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])

def test_embed_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        FakeEmbedding().embed([""])
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_local_embedding.py -v`

Expected: missing embedding modules.

- [ ] **Step 3: Implement lazy ModelScope/SentenceTransformer loading and normalization**

Keep imports lazy so default tests do not load models. Allow encoder injection. Reject empty input strings and malformed encoder shapes.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_local_embedding.py -v`

Expected: all pass without network access.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/embeddings tests/fixtures/fake_embedding.py tests/unit/test_local_embedding.py
git commit -m "feat: add local multilingual embeddings"
```

### Task 4: SQLite experience repository and event history

**Files:**
- Create: `src/sentiment_agent/experience/__init__.py`
- Create: `src/sentiment_agent/experience/repository.py`
- Test: `tests/unit/test_experience_repository.py`

**Interfaces:**
- Consumes: `Experience`, `RetrievedExperience` outcomes.
- Produces: `ExperienceRepository.create`, `.get`, `.list`, `.count`, `.merge_counts`, `.record_event`, `.history`, `.record_outcomes`, `.close`.

- [ ] **Step 1: Write failing isolated persistence tests**

```python
def test_repository_persists_experience_and_creation_event(tmp_path, experience):
    with ExperienceRepository(tmp_path / "exp.sqlite3") as repo:
        repo.create(experience)
        assert repo.get(experience.id) == experience
        assert [event.event_type for event in repo.history(experience.id)] == ["created"]

def test_merge_counts_is_transactional(tmp_path, experience):
    with ExperienceRepository(tmp_path / "exp.sqlite3") as repo:
        repo.create(experience)
        updated = repo.merge_counts(experience.id, success_delta=1, failure_delta=0, batch_id=2)
        assert updated.reliability == pytest.approx(2 / 3)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_experience_repository.py -v`

Expected: repository import fails.

- [ ] **Step 3: Implement versioned schema and transactions**

Create `experiences`, `experience_events`, `experience_outcomes`, and `schema_metadata`; enable foreign keys and WAL; use explicit transactions; serialize Pydantic payloads as canonical JSON; parameterize every query.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_experience_repository.py -v`

Expected: persistence, rollback, query, history, and context-manager tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/experience tests/unit/test_experience_repository.py
git commit -m "feat: add auditable experience repository"
```

### Task 5: Atomic vector index, retrieval, and deterministic updates

**Files:**
- Create: `src/sentiment_agent/experience/vector_index.py`
- Create: `src/sentiment_agent/experience/retrieval.py`
- Create: `src/sentiment_agent/experience/updater.py`
- Test: `tests/unit/test_vector_index.py`
- Test: `tests/unit/test_retrieval.py`
- Test: `tests/unit/test_experience_updater.py`

**Interfaces:**
- Produces: `VectorIndex.upsert`, `.snapshot`, `ExperienceRetriever.search(vector, snapshot, ...)`, `ExperienceUpdater.apply(item, prediction, feedback, retrieved, vector, batch_id)`.

- [ ] **Step 1: Write failing index, snapshot, ranking, and reliability tests**

```python
def test_snapshot_does_not_see_later_upsert(index):
    index.upsert("one", np.array([1.0, 0.0], dtype=np.float32))
    snapshot = index.snapshot()
    index.upsert("two", np.array([0.0, 1.0], dtype=np.float32))
    assert snapshot.ids == ("one",)

def test_retrieval_records_score_components(retriever, snapshot):
    result = retriever.search(np.array([1.0, 0.0]), snapshot, language="vi", source="tiny", k=1)
    assert result[0].score == pytest.approx(sum(result[0].score_components.values()))

def test_wrong_prediction_creates_error_correction(updater, item, wrong_prediction):
    result = updater.apply(item, wrong_prediction, feedback_for(item, wrong_prediction), [], VECTOR, batch_id=1)
    assert result.experience.type == "error_correction"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_vector_index.py tests/unit/test_retrieval.py tests/unit/test_experience_updater.py -v`

Expected: missing modules or interfaces.

- [ ] **Step 3: Implement atomic `.npy`/JSON replacement, exact cosine ranking, deduplication, and Beta reliability**

Normalize incoming vectors, reject dimension changes, sort score ties by experience ID, update only injected experiences, use `(success+1)/(success+failure+2)`, and append every mutation to event history.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_vector_index.py tests/unit/test_retrieval.py tests/unit/test_experience_updater.py -v`

Expected: all pass, including restart and snapshot isolation tests.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/experience tests/unit/test_vector_index.py tests/unit/test_retrieval.py tests/unit/test_experience_updater.py
git commit -m "feat: add deterministic experience evolution"
```

### Task 6: Fixed prompt, Qwen backend, response cache, and fake LLM

**Files:**
- Create: `src/sentiment_agent/prompts/__init__.py`
- Create: `src/sentiment_agent/prompts/prediction.py`
- Create: `src/sentiment_agent/llm/__init__.py`
- Create: `src/sentiment_agent/llm/base.py`
- Create: `src/sentiment_agent/llm/cache.py`
- Create: `src/sentiment_agent/llm/langchain_qwen.py`
- Create: `tests/fixtures/fake_llm.py`
- Test: `tests/unit/test_prediction_prompt.py`
- Test: `tests/unit/test_llm_backend.py`
- Test: `tests/unit/test_llm_cache.py`

**Interfaces:**
- Produces: `PredictionPromptBuilder.build(item, experiences) -> list[BaseMessage]`, async `LLMBackend.classify(messages) -> LLMResult`, `LangChainQwenBackend`, `ResponseCache.get/put`.

- [ ] **Step 1: Write failing tests for stable prompt, safe rendering, parsing, retry boundary, and cache identity**

```python
def test_base_prompt_is_identical_with_and_without_experience(builder, item, experience):
    plain = builder.build(item, [])
    augmented = builder.build(item, [experience])
    assert plain[0].content == augmented[0].content

@pytest.mark.asyncio
async def test_qwen_backend_parses_fenced_json(fake_chat_model):
    fake_chat_model.response = '```json\n{"label":"negative","confidence":0.9,"reason":"negation"}\n```'
    result = await LangChainQwenBackend(chat_model=fake_chat_model).classify(MESSAGES)
    assert result.payload.label == "negative"

def test_cache_key_changes_when_prompt_changes(cache):
    assert cache.key("qwen", {}, ["a"]) != cache.key("qwen", {}, ["b"])
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_prediction_prompt.py tests/unit/test_llm_backend.py tests/unit/test_llm_cache.py -v`

Expected: missing prompt and LLM modules.

- [ ] **Step 3: Implement fixed prompt, escaped structured experience context, strict JSON parser, injectable ChatModel, and SQLite cache**

Instantiate `ChatOpenAI` only when no test model is injected. Read the key via the configured environment-variable name. Include model parameters and normalized messages in SHA-256 cache keys. Never cache exceptions.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_prediction_prompt.py tests/unit/test_llm_backend.py tests/unit/test_llm_cache.py -v`

Expected: all pass with no API calls.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/prompts src/sentiment_agent/llm tests/fixtures/fake_llm.py tests/unit/test_prediction_prompt.py tests/unit/test_llm_backend.py tests/unit/test_llm_cache.py
git commit -m "feat: add fixed-prompt Qwen classification"
```

### Task 7: Mini-batch agent

**Files:**
- Create: `src/sentiment_agent/agent/__init__.py`
- Create: `src/sentiment_agent/agent/sentiment_agent.py`
- Test: `tests/unit/test_sentiment_agent.py`
- Test: `tests/unit/test_batch_semantics.py`

**Interfaces:**
- Produces: async `SentimentAgent.predict_batch(items, max_concurrency) -> BatchPrediction`, `SentimentAgent.learn_batch(items, predictions, feedback, batch_id) -> list[LearningResult]`, and read-only async `predict` delegation.

- [ ] **Step 1: Write failing tests for one embedding call, bounded concurrency, stable output order, snapshot visibility, ordered learning, and failure atomicity**

```python
@pytest.mark.asyncio
async def test_batch_predictions_preserve_input_order(agent):
    predictions = await agent.predict_batch([SLOW_ITEM, FAST_ITEM], max_concurrency=2)
    assert [p.sample_id for p in predictions.items] == [SLOW_ITEM.id, FAST_ITEM.id]

@pytest.mark.asyncio
async def test_same_batch_cannot_retrieve_new_experience(agent):
    batch = await agent.predict_batch([ITEM_A, ITEM_B], max_concurrency=2)
    assert all(not prediction.retrieved_experience_ids for prediction in batch.items)
    agent.learn_batch(...)
    next_batch = await agent.predict_batch([ITEM_C], max_concurrency=1)
    assert next_batch.items[0].retrieved_experience_ids
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_sentiment_agent.py tests/unit/test_batch_semantics.py -v`

Expected: missing agent module.

- [ ] **Step 3: Implement snapshot preparation, one embedding batch, semaphore-bounded `asyncio.gather`, immutable prediction contexts, and ordered learning**

Do not expose a learn operation until the complete prediction batch exists. Verify sample IDs and predicted labels against feedback before the first mutation.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_sentiment_agent.py tests/unit/test_batch_semantics.py -v`

Expected: all batch semantics pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/agent tests/unit/test_sentiment_agent.py tests/unit/test_batch_semantics.py
git commit -m "feat: add atomic mini-batch evolution"
```

### Task 8: Metrics, artifacts, and experiment runner

**Files:**
- Create: `src/sentiment_agent/evaluation/__init__.py`
- Create: `src/sentiment_agent/evaluation/metrics.py`
- Create: `src/sentiment_agent/experiments/__init__.py`
- Create: `src/sentiment_agent/experiments/artifacts.py`
- Create: `src/sentiment_agent/experiments/runner.py`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_artifacts.py`
- Test: `tests/unit/test_experiment_runner.py`
- Test: `tests/unit/test_no_test_leakage.py`

**Interfaces:**
- Produces: `classification_metrics(gold, predicted)`, `ArtifactWriter`, async `ExperimentRunner.run(train, dev, test) -> RunSummary`.

- [ ] **Step 1: Write failing tests for three-class metrics, atomic JSON/JSONL output, hard checkpoint splitting, baseline/evolution parity, and evaluation non-mutation**

```python
def test_checkpoint_splits_batch_exactly(runner):
    runner.run_training(items(5), batch_size=4, checkpoints=[3, 5])
    assert runner.agent.observed_batch_sizes == [3, 2]

@pytest.mark.asyncio
async def test_test_evaluation_never_learns(runner, test_examples):
    before = runner.agent.experience_count
    await runner.evaluate(test_examples, split="test")
    assert runner.agent.experience_count == before

def test_macro_f1_uses_all_three_labels():
    result = classification_metrics(["positive"], ["positive"])
    assert set(result.per_class) == {"positive", "neutral", "negative"}
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_metrics.py tests/unit/test_artifacts.py tests/unit/test_experiment_runner.py tests/unit/test_no_test_leakage.py -v`

Expected: missing evaluation and experiment modules.

- [ ] **Step 3: Implement explicit-label sklearn metrics, atomic artifact writes, hard-boundary batching, read-only evaluation, manifests, and checkpoints**

Manifest fields include config hash, dataset hashes, seed, Git commit, package versions, model and embedding IDs, start/end time, and status. Refuse to overwrite an existing run directory.

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/unit/test_metrics.py tests/unit/test_artifacts.py tests/unit/test_experiment_runner.py tests/unit/test_no_test_leakage.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sentiment_agent/evaluation src/sentiment_agent/experiments tests/unit/test_metrics.py tests/unit/test_artifacts.py tests/unit/test_experiment_runner.py tests/unit/test_no_test_leakage.py
git commit -m "feat: add reproducible experiment runner"
```

### Task 9: CLI, experience inspection, and documentation

**Files:**
- Create: `src/sentiment_agent/cli.py`
- Create: `tests/unit/test_cli.py`
- Create: `README.md`
- Create: `configs/experiments/baseline_zero_shot.yaml`
- Create: `configs/experiments/evolution.yaml`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces CLI commands `validate-config`, `run`, `summarize`, and nested `experience list/show/stats/history/export`.

- [ ] **Step 1: Write failing Typer CLI tests**

```python
def test_validate_config_does_not_require_api_key(runner, config_path):
    result = runner.invoke(app, ["validate-config", "--config", str(config_path)])
    assert result.exit_code == 0

def test_experience_export_csv(runner, run_dir):
    result = runner.invoke(app, ["experience", "export", "--run", str(run_dir), "--format", "csv"])
    assert result.exit_code == 0
    assert (run_dir / "experience-export.csv").exists()
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_cli.py -v`

Expected: CLI import or commands fail.

- [ ] **Step 3: Implement commands, human-readable summaries, CSV/JSONL export, examples, and setup documentation**

README commands must use `uv sync --frozen --extra dev` and `uv run sentiment-agent ...`; document that SQLite needs no server and GUI tools are optional; explain offline versus explicit online commands.

- [ ] **Step 4: Verify CLI tests and help**

Run: `uv run pytest tests/unit/test_cli.py -v`

Run: `uv run sentiment-agent --help`

Expected: tests pass and help lists all commands.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml uv.lock README.md configs src/sentiment_agent/cli.py tests/unit/test_cli.py
git commit -m "feat: add research experiment CLI"
```

### Task 10: Complete offline integration workflow and verification

**Files:**
- Create: `tests/fixtures/tiny_train.jsonl`
- Create: `tests/fixtures/tiny_test.jsonl`
- Create: `tests/integration/test_offline_evolution_workflow.py`
- Modify: any V1 file only when the integration test exposes a defect.

**Interfaces:**
- Consumes all V1 public interfaces.
- Produces a verified complete offline research workflow.

- [ ] **Step 1: Write the failing complete workflow test**

```python
@pytest.mark.asyncio
async def test_complete_offline_evolution_workflow(tmp_path):
    summary = await build_fake_runner(tmp_path).run(TINY_TRAIN, TINY_DEV, TINY_TEST)
    assert summary.completed_samples == len(TINY_TRAIN)
    assert summary.checkpoints == [2, 4]
    assert (summary.run_dir / "metrics.json").exists()
    with ExperienceRepository(summary.run_dir / "experience_store" / "experiences.sqlite3") as repo:
        assert repo.count() > 0
        assert not set(TEST_IDS) & {experience.source_sample_id for experience in repo.list()}
    second_batch_predictions = read_predictions(summary.run_dir, batch_id=2)
    assert any(row["retrieved_experience_ids"] for row in second_batch_predictions)
```

- [ ] **Step 2: Verify the workflow test fails before final wiring**

Run: `uv run pytest tests/integration/test_offline_evolution_workflow.py -v`

Expected: failure identifies missing final factory or integration wiring.

- [ ] **Step 3: Add only the wiring needed to complete the fake-backed workflow**

Connect config, loaders, fake backends, repository, vector index, agent, runner, evaluator, and artifacts without introducing test-only branches in production code.

- [ ] **Step 4: Run complete verification**

Run: `uv run pytest -q`

Expected: all default tests pass without network or credentials.

Run: `uv run pytest tests/integration/test_offline_evolution_workflow.py -v`

Expected: the full offline workflow passes and proves next-batch visibility plus test isolation.

Run: `uv run pytest --cov=sentiment_agent --cov-report=term-missing --cov-fail-under=85`

Expected: all tests pass and coverage is at least 85%.

- [ ] **Step 5: Perform explicit opt-in smoke checks**

Run: `uv run sentiment-agent validate-config --config configs/experiments/evolution.yaml`

Expected: configuration is valid without printing secrets.

If the local BGE-M3 model and `.env` Qwen settings are available, run the documented one-batch smoke command. Record success or the exact external prerequisite; never make this optional network check part of the default suite.

- [ ] **Step 6: Commit**

```powershell
git add tests/fixtures tests/integration src configs README.md pyproject.toml uv.lock
git commit -m "test: verify offline self-evolution workflow"
```
