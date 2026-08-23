# Train/Test-Only Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove development-dataset configuration and evaluation so experiments train on `train_paths` and evaluate once on `test_paths`.

**Architecture:** Make the strict configuration model and runner API express the two-split workflow directly. The CLI fingerprints and loads those two splits only, while checkpoint boundaries continue to govern training batches and manifest metadata without triggering evaluation.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, pytest, YAML, `uv`.

## Global Constraints

- Use `uv` as the only Python environment and dependency manager.
- Write and observe a failing test before every production behavior change.
- Every public function or method changed by this work must retain isolated unit coverage.
- Tests must remain offline and require no real credentials.
- Run the complete offline integration workflow before claiming completion.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Remove `dev_paths` from configuration

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_cli_v1.py`
- Modify: `src/sentiment_agent/config.py`
- Modify: `configs/default.yaml`
- Modify: `configs/experiments/baseline_zero_shot.yaml`
- Modify: `configs/experiments/evolution.yaml`
- Modify: `configs/experiments/evolution_mini.yaml`

**Interfaces:**
- Consumes: `load_config(path: Path) -> ExperimentConfig`
- Produces: `RunConfig` with `train_paths: list[Path]` and `test_paths: list[Path]`, and strict rejection of obsolete `dev_paths`.

- [ ] **Step 1: Write failing configuration tests**

Update the valid YAML fixture to omit `dev_paths`, change the mini-config assertion to check only train/test, and add:

```python
def test_load_config_rejects_obsolete_dev_paths(tmp_path: Path) -> None:
    path = tmp_path / "obsolete.yaml"
    _write_config(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("  test_paths:", "  dev_paths: [dev.json]\n  test_paths:"), encoding="utf-8")

    with pytest.raises(ValueError, match="dev_paths"):
        load_config(path)
```

Remove `dev_paths: []` from the CLI validation fixture.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run python -m pytest tests/unit/test_config.py tests/unit/test_cli_v1.py -v`

Expected: FAIL because `RunConfig.dev_paths` is still required and shipped configs still expose it.

- [ ] **Step 3: Implement the minimal configuration change**

Change `RunConfig` to:

```python
class RunConfig(StrictModel):
    train_paths: list[Path]
    test_paths: list[Path]
    output_root: Path
    train_batch_size: PositiveInt = 1
    checkpoints: list[PositiveInt] = []
    seed: int = 42
    use_cache: bool = True
```

Delete `dev_paths` from all four shipped YAML files.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run python -m pytest tests/unit/test_config.py tests/unit/test_cli_v1.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the configuration change**

```powershell
git add -- tests/unit/test_config.py tests/unit/test_cli_v1.py src/sentiment_agent/config.py configs/default.yaml configs/experiments/baseline_zero_shot.yaml configs/experiments/evolution.yaml configs/experiments/evolution_mini.yaml
git commit -m "refactor: remove development dataset configuration"
```

### Task 2: Convert the runner to train/test-only execution

**Files:**
- Modify: `tests/integration/test_offline_evolution_workflow.py`
- Modify: `tests/integration/test_generalized_evolution_workflow.py`
- Modify: `tests/unit/test_metrics_runner.py`
- Modify: `src/sentiment_agent/experiments/runner.py`
- Modify: `src/sentiment_agent/experiments/progress.py`

**Interfaces:**
- Consumes: `ExperimentRunner.evaluate(examples, *, split, checkpoint=None) -> dict`
- Produces: `ExperimentRunner.run(train: Sequence[SentimentExample], test: Sequence[SentimentExample]) -> RunSummary` and progress stages `Literal["train", "test"]`.

- [ ] **Step 1: Write failing runner tests**

Change all direct calls from `run(train, dev, test)` to `run(train, test)`. In the offline workflow, retain checkpoints and assert:

```python
summary = await runner.run(
    [example("train-1"), example("train-2"), example("train-3"), example("train-4", "negative")],
    [example("test-1")],
)
assert {event.stage for event in progress.events} == {"train", "test"}
assert not list(tmp_path.glob("metrics-dev-*.json"))
```

Add or adjust the isolated runner unit test to assert `summary.checkpoints` still contains reached checkpoint boundaries while the fake agent receives only train and test predictions.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run python -m pytest tests/unit/test_metrics_runner.py tests/integration/test_offline_evolution_workflow.py tests/integration/test_generalized_evolution_workflow.py -v`

Expected: FAIL with a `run()` argument mismatch or remaining dev-stage behavior.

- [ ] **Step 3: Implement the minimal runner change**

Use this signature and remove checkpoint dev evaluation:

```python
async def run(self, train: Sequence[SentimentExample],
              test: Sequence[SentimentExample]) -> RunSummary:
```

Inside the checkpoint branch, retain only:

```python
if processed in self.checkpoints:
    reached.append(processed)
```

Change `ProgressEvent.stage` to `Literal["train", "test"]`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run python -m pytest tests/unit/test_metrics_runner.py tests/integration/test_offline_evolution_workflow.py tests/integration/test_generalized_evolution_workflow.py -v`

Expected: PASS with no dev event or dev metrics file.

- [ ] **Step 5: Commit the runner change**

```powershell
git add -- tests/unit/test_metrics_runner.py tests/integration/test_offline_evolution_workflow.py tests/integration/test_generalized_evolution_workflow.py src/sentiment_agent/experiments/runner.py src/sentiment_agent/experiments/progress.py
git commit -m "refactor: run experiments with train and test splits"
```

### Task 3: Remove dev loading from the CLI and update documentation

**Files:**
- Modify: `tests/unit/test_cli_v1.py`
- Modify: `src/sentiment_agent/cli.py`
- Modify: `README.md`
- Modify: `docs/experience-evolution-v1.md`

**Interfaces:**
- Consumes: `ExperimentRunner.run(train, test)` from Task 2.
- Produces: CLI dataset fingerprints and loads for train/test only.

- [ ] **Step 1: Write a failing CLI behavior test**

Add a minimal config with retrieval and generalization disabled, set a dummy API key, replace `ExperimentRunner.run` with an offline async recorder, and invoke the real CLI:

```python
def test_run_passes_only_train_and_test_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("""
model:
  name: qwen-plus
  base_url: https://example.invalid/v1
  api_key_env: TEST_QWEN_KEY
embedding:
  model_id: unused
retrieval:
  enabled: false
generalization:
  enabled: false
experiment:
  train_paths: []
  test_paths: []
  output_root: OUTPUT_ROOT
  train_batch_size: 2
""".replace("OUTPUT_ROOT", tmp_path.as_posix()), encoding="utf-8")
    monkeypatch.setenv("TEST_QWEN_KEY", "offline-placeholder")
    received = []

    async def fake_run(self, train, test):
        received.append((train, test))
        return RunSummary(self.writer.run_dir, 0, (), {})

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--no-progress"])

    assert result.exit_code == 0
    assert received == [([], [])]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run python -m pytest tests/unit/test_cli_v1.py -v`

Expected: FAIL with a missing `dev_paths` attribute because the CLI still accesses the removed field.

- [ ] **Step 3: Implement CLI and documentation changes**

Build fingerprints from:

```python
settings.experiment.train_paths + settings.experiment.test_paths
```

Load `train` and `test` only, then call:

```python
summary = asyncio.run(runner.run(train, test))
```

Update README and `docs/experience-evolution-v1.md` so dataset and artifact descriptions mention only train/test and explain that checkpoints no longer run development evaluation.

- [ ] **Step 4: Run focused tests and configuration validation**

Run:

```powershell
uv run python -m pytest tests/unit/test_cli_v1.py tests/unit/test_config.py -v
uv run sentiment-agent validate-config --config configs/experiments/evolution_mini.yaml
uv run sentiment-agent validate-config --config configs/experiments/evolution.yaml
```

Expected: all tests PASS and both configurations report `Configuration valid`.

- [ ] **Step 5: Commit the CLI and documentation change**

```powershell
git add -- tests/unit/test_cli_v1.py src/sentiment_agent/cli.py README.md docs/experience-evolution-v1.md
git commit -m "docs: describe train-test-only experiments"
```

### Task 4: Complete offline verification

**Files:**
- Verify only; modify a test or implementation file only if a failure exposes an in-scope regression, following a fresh RED/GREEN cycle.

**Interfaces:**
- Consumes: the train/test-only configuration, runner, and CLI behavior from Tasks 1-3.
- Produces: verified offline workflow with no dev references in executable paths.

- [ ] **Step 1: Search for stale executable dev references**

Run:

```powershell
rg -n -e 'dev_paths' -e 'split="dev"' -e 'stage.*dev' src tests configs
```

Expected: only the intentional obsolete-field rejection test may mention `dev_paths`; no production/config references remain.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run python -m pytest -q`

Expected: PASS.

- [ ] **Step 3: Run the required offline integration workflow**

Run: `uv run python -m pytest tests/integration/test_offline_evolution_workflow.py -v`

Expected: PASS without API calls or credentials.

- [ ] **Step 4: Inspect the final diff**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated pre-existing worktree changes remain untouched.

- [ ] **Step 5: Commit any final in-scope test-only cleanup**

If Task 4 required an in-scope correction, stage only its exact files and commit with `test: verify train-test-only workflow`. Otherwise, do not create an empty commit.
