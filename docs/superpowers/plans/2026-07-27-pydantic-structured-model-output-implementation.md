# Pydantic Structured Model Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all manual LLM response parsing with LangChain structured output backed by strict Pydantic models.

**Architecture:** Bind one LangChain structured-output runnable per response schema. Keep existing result envelopes and attribution fallback behavior while moving validation to the LLM boundary.

**Tech Stack:** Python 3.12, uv, Pydantic 2, LangChain, pytest.

## Global Constraints

- Use `uv` as the only Python environment and dependency manager.
- Add a failing test before each production change.
- Every new public function or method needs an isolated unit test.
- Tests must not call external APIs or require real credentials.
- Run the complete offline integration workflow before completion.

---

### Task 1: Typed prediction and translation output

**Files:**
- Modify: `src/sentiment_agent/llm/base.py`
- Modify: `src/sentiment_agent/llm/langchain_qwen.py`
- Test: `tests/unit/test_prompt_and_llm.py`

**Interfaces:**
- Produces: `TranslationPayload(text: str)` and structured `classify()` / `complete_text()` results.

- [ ] Write tests whose fake chat model only supports `with_structured_output(schema)` and whose runnables return `PredictionPayload` and `TranslationPayload`.
- [ ] Run `uv run pytest tests/unit/test_prompt_and_llm.py -v` and verify the tests fail because the backend invokes the unbound chat model.
- [ ] Add `TranslationPayload`, bind both schemas in `LangChainQwenBackend.__init__`, invoke the bound runnables, validate their returned values, and preserve usage metadata from the raw message.
- [ ] Run `uv run pytest tests/unit/test_prompt_and_llm.py -v` and verify it passes.

### Task 2: Typed attribution output with retry and fallback

**Files:**
- Modify: `src/sentiment_agent/attribution/llm_attributor.py`
- Modify: `src/sentiment_agent/cli.py`
- Test: `tests/unit/test_attribution.py`

**Interfaces:**
- Produces: a structured attribution client result containing `AttributionPayload | None`, raw response text, and parsing error.

- [ ] Replace the string-client test double with a structured client double and add assertions for successful typed output, parse-error retries, and fallback audit data.
- [ ] Run `uv run pytest tests/unit/test_attribution.py -v` and verify failure because the current attributor expects strings and parses JSON itself.
- [ ] Bind `AttributionPayload` using `include_raw=True`, normalize LangChain's response into the typed client result, remove regex/JSON extraction, and retain retry/fallback behavior.
- [ ] Update CLI construction to pass the structured attribution client.
- [ ] Run `uv run pytest tests/unit/test_attribution.py tests/unit/test_cli_v1.py -v` and verify they pass.

### Task 3: Full verification

**Files:**
- Verify all modified source and test files.

- [ ] Run `uv run pytest tests/unit -v`.
- [ ] Run `uv run pytest tests/integration/test_offline_evolution_workflow.py tests/integration/test_generalized_evolution_workflow.py -v`.
- [ ] Run `uv run pytest -v` for the full offline suite.
- [ ] Inspect `git diff --check` and the scoped diff for accidental or unrelated changes.
