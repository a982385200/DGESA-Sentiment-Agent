# Parallel Zero-Shot Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel, auditable zero-shot benchmark for the 2,500 medium test examples.

**Architecture:** A reusable async benchmark module owns bounded concurrency, metrics, and artifacts. A thin script loads model settings and invokes it.

**Tech Stack:** Python 3.12, uv, asyncio, Pydantic 2, LangChain, pytest.

## Global Constraints

- Use `uv` exclusively.
- Write failing tests before production code.
- Tests remain offline and credential-free.
- All LLM responses use the existing Pydantic structured backend.

### Task 1: Benchmark engine

**Files:**
- Create: `src/sentiment_agent/benchmarks/zero_shot.py`
- Create: `src/sentiment_agent/benchmarks/__init__.py`
- Test: `tests/unit/test_zero_shot_benchmark.py`

- [ ] Write an offline test for five concurrent languages, a global concurrency cap, one failed sample, and all required artifacts.
- [ ] Run the test and verify it fails because the module does not exist.
- [ ] Implement bounded classification, per-language artifacts, aggregate metrics, coverage, usage, and failures.
- [ ] Run the test and verify it passes.

### Task 2: Command-line script

**Files:**
- Create: `scripts/run_zero_shot_benchmark.py`
- Test: `tests/unit/test_zero_shot_benchmark.py`

- [ ] Add a failing parser test for dataset, output, config, and concurrency options.
- [ ] Implement the thin script using `load_config` and `LangChainQwenBackend`.
- [ ] Run the benchmark tests and then `uv run python -m pytest -q`.
