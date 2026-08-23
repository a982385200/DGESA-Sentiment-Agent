# Attribution-Driven Experience Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-sample prompt memory with an auditable case → attribution → generalized-rule evolution loop that promotes rules only after cross-batch evidence.

**Architecture:** `ExperienceEvolutionService` owns feedback learning and composes a SQLite repository, error attributor, vector matcher, and lifecycle policy. `SentimentAgent` keeps batch snapshot semantics but retrieves generalized rules through an adapter; the experiment runner continues to isolate dev/test from learning.

**Tech Stack:** Python 3.12, uv, Pydantic 2, SQLite, NumPy, BGE-M3, LangChain Qwen, pytest.

## Global Constraints

- Use `uv` as the only Python environment and dependency manager.
- Add a failing test before production code; every public function and method has an isolated unit test.
- Default tests never call external APIs, load credentials, or download models.
- The base prompt and model parameters remain fixed within a comparison.
- A batch sees only the generalized-experience snapshot available at batch start.
- dev/test never writes cases, attributions, evidence, or rules.
- Attribution failure falls back deterministically and does not abort learning.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Typed evolution models and configuration

**Files:**
- Create: `src/sentiment_agent/evidence/__init__.py`
- Create: `src/sentiment_agent/evidence/models.py`
- Create: `src/sentiment_agent/attribution/__init__.py`
- Create: `src/sentiment_agent/attribution/models.py`
- Create: `src/sentiment_agent/generalization/__init__.py`
- Create: `src/sentiment_agent/generalization/models.py`
- Modify: `src/sentiment_agent/config.py`
- Modify: `configs/experiments/evolution_mini.yaml`
- Test: `tests/unit/test_evolution_models.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces `CaseEvidence`, `Attribution`, `AttributionPayload`, `GeneralizedExperience`, `AttributionConfig`, and `GeneralizationConfig`.

- [ ] Write failing tests for immutable models, fixed error/status enums, reliability, contradiction ratio, and validated thresholds.

```python
def test_generalized_experience_computes_quality():
    rule = make_rule(support_count=3, contradiction_count=1)
    assert rule.reliability == pytest.approx(4 / 6)
    assert rule.contradiction_ratio == pytest.approx(0.25)

def test_mini_config_enables_generalization():
    config = load_config(Path("configs/experiments/evolution_mini.yaml"))
    assert config.attribution.enabled
    assert config.generalization.minimum_support == 2
```

- [ ] Run `uv run python -m pytest tests/unit/test_evolution_models.py tests/unit/test_config.py -v`; expect import/config failures.
- [ ] Implement strict Pydantic models and config sections with defaults copied from the approved spec.
- [ ] Run focused tests; expect pass.
- [ ] Commit exact files with `feat: add experience evolution models`.

### Task 2: SQLite evidence and generalized-experience repository

**Files:**
- Create: `src/sentiment_agent/generalization/repository.py`
- Test: `tests/unit/test_generalized_repository.py`

**Interfaces:**
- Consumes typed models from Task 1.
- Produces `EvolutionRepository.create_case`, `.get_case`, `.create_attribution`, `.create_rule`, `.get_rule`, `.list_rules`, `.add_evidence`, `.update_rule`, `.record_event`, `.stats`, and context-manager methods.

- [ ] Write failing tests proving one case per sample/batch, attribution persistence, idempotent evidence relations, support/contradiction counts, event history, and restart persistence.

```python
def test_evidence_relation_is_idempotent(repo, rule, case):
    repo.create_case(case)
    repo.create_rule(rule)
    assert repo.add_evidence(rule.id, case.id, relation="support", batch_id=2)
    assert not repo.add_evidence(rule.id, case.id, relation="support", batch_id=2)
    assert repo.get_rule(rule.id).support_count == 1
```

- [ ] Run `uv run python -m pytest tests/unit/test_generalized_repository.py -v`; expect missing repository.
- [ ] Implement versioned tables, foreign keys, WAL, parameterized SQL, transactions, unique evidence constraints, JSON serialization, and derived count updates.
- [ ] Run focused tests; expect pass.
- [ ] Commit exact files with `feat: persist generalized experience evidence`.

### Task 3: Deterministic and LLM error attribution with fallback

**Files:**
- Create: `src/sentiment_agent/attribution/deterministic.py`
- Create: `src/sentiment_agent/attribution/llm_attributor.py`
- Modify: `src/sentiment_agent/llm/base.py`
- Modify: `src/sentiment_agent/llm/langchain_qwen.py`
- Test: `tests/unit/test_attribution.py`
- Test: `tests/unit/test_prompt_and_llm.py`

**Interfaces:**
- Produces `infer_error_type(prediction, feedback, retrieved) -> ErrorType`, async `LLMAttributor.attribute(case, retrieved) -> AttributionResult`, and `LLMBackend.attribute(messages) -> AttributionLLMResult`.

- [ ] Write failing tests for missing knowledge, wrong experience, negative transfer, valid attribution JSON, non-JSON retry, and deterministic fallback after final failure.

```python
@pytest.mark.anyio
async def test_attributor_falls_back_after_invalid_json():
    result = await LLMAttributor(AlwaysInvalidLLM(), max_retries=2).attribute(CASE, [])
    assert result.used_fallback
    assert result.attribution.error_type == "missing_knowledge"
```

- [ ] Run focused tests; expect missing attribution modules/interfaces.
- [ ] Implement structured prompts, bounded format-correction retry, raw response capture, and fallback candidate text derived from gold label plus corrected reason.
- [ ] Run focused tests; expect pass.
- [ ] Commit exact files with `feat: attribute sentiment prediction errors`.

### Task 4: Online rule matching and lifecycle

**Files:**
- Create: `src/sentiment_agent/generalization/matcher.py`
- Create: `src/sentiment_agent/generalization/lifecycle.py`
- Create: `src/sentiment_agent/generalization/service.py`
- Test: `tests/unit/test_generalization.py`

**Interfaces:**
- Produces `RuleMatcher.find_match(candidate, vector, snapshot)`, `LifecyclePolicy.next_status(rule)`, and async `ExperienceEvolutionService.learn_batch(items, predictions, feedback, contexts, batch_id)`.

- [ ] Write failing tests for same-label merge, conflicting-label separation, cross-batch promotion, single-batch non-promotion, reliability reduction, correct-sample no-attribution, and fallback failure artifact callback.

```python
@pytest.mark.anyio
async def test_rule_activates_only_after_two_batches(service):
    await service.learn_error(ERROR_A_BATCH_1)
    assert service.rules()[0].status == "candidate"
    await service.learn_error(SIMILAR_ERROR_BATCH_2)
    assert service.rules()[0].status == "active"
```

- [ ] Run `uv run python -m pytest tests/unit/test_generalization.py -v`; expect missing modules.
- [ ] Implement normalized rule vectors, deterministic similarity/tie-breaking, evidence merge, lifecycle transitions, and event recording.
- [ ] Run focused tests; expect pass.
- [ ] Commit exact files with `feat: generalize attributed experiences`.

### Task 5: Generalized retrieval, prompt injection, and agent integration

**Files:**
- Create: `src/sentiment_agent/generalization/retrieval.py`
- Modify: `src/sentiment_agent/prompts/prediction.py`
- Modify: `src/sentiment_agent/agent/sentiment_agent.py`
- Modify: `src/sentiment_agent/experiments/runner.py`
- Test: `tests/unit/test_generalized_retrieval.py`
- Test: `tests/unit/test_sentiment_agent.py`

**Interfaces:**
- Produces `GeneralizedExperienceRetriever.search`, generalized prompt rendering, and Agent delegation to `ExperienceEvolutionService` while retaining `predict_batch`/`learn_batch` public signatures.

- [ ] Write failing tests that candidate rules are invisible, active rules are retrieved, prompts omit original cases/provenance, same-batch rules are invisible, next-batch active rules are visible, and evaluation never learns.
- [ ] Run focused tests; expect failures against current case-memory pipeline.
- [ ] Implement a generalized vector snapshot, active-only scoring, concise rule context, and evolution-service learning delegation. Keep a legacy case-memory adapter only for old unit tests/config mode.
- [ ] Run focused tests; expect pass.
- [ ] Commit exact files with `feat: retrieve generalized sentiment experiences`.

### Task 6: CLI wiring, artifacts, metrics, docs, and full offline workflow

**Files:**
- Modify: `src/sentiment_agent/cli.py`
- Modify: `src/sentiment_agent/experiments/runner.py`
- Modify: `README.md`
- Modify: `tests/integration/test_offline_evolution_workflow.py`
- Create: `tests/integration/test_generalized_evolution_workflow.py`
- Test: `tests/unit/test_cli_v1.py`

**Interfaces:**
- Produces a configured generalized-experience run plus `attributions.jsonl`, `attribution_failures.jsonl`, `generalized_experiences.jsonl`, and `experience_evolution_metrics.json`.

- [ ] Write a failing offline workflow test that processes two similar errors in different batches, persists cases/attributions, promotes one active rule, retrieves it in a later batch, and proves test isolation.

```python
@pytest.mark.anyio
async def test_error_rules_generalize_and_transfer_to_next_batch(tmp_path):
    summary = await build_generalized_fake_runner(tmp_path).run(TRAIN, DEV, TEST)
    metrics = json.loads((summary.run_dir / "experience_evolution_metrics.json").read_text())
    assert metrics["case_count"] == len(TRAIN)
    assert metrics["active_count"] == 1
    assert read_test_predictions(summary.run_dir)[0]["retrieved_experience_ids"]
```

- [ ] Run integration test; expect missing wiring/artifacts.
- [ ] Wire config, repository, attributor, matcher, lifecycle, service, retriever, and artifact callbacks in CLI/factory code; export final rules and metrics at completion.
- [ ] Document model-call costs, tables, config, CLI, and the distinction between cases and generalized experiences.
- [ ] Run `uv run python -m pytest -q`; expect all pass.
- [ ] Run coverage with `uv run python -m pytest --cov=sentiment_agent --cov-report=term-missing --cov-fail-under=85`; expect at least 85%.
- [ ] Validate `configs/experiments/evolution_mini.yaml`; expect valid without reading the API key.
- [ ] Commit exact files with `feat: complete attribution-driven evolution workflow`.
