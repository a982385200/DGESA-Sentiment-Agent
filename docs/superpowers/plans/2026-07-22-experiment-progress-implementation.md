# Experiment Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default Rich progress reporting and a `--no-progress` CLI switch without changing experiment semantics.

**Architecture:** The runner emits immutable `ProgressEvent` values through an injected reporter protocol. A Rich adapter renders events; a null adapter keeps library use silent and testable.

**Tech Stack:** Python 3.12, Rich, Typer, Pydantic, pytest, uv.

## Global Constraints

- Use `uv` exclusively.
- Add failing tests before production code.
- Do not call APIs or load models in default tests.
- Progress failures must not change experiment behavior.
- Preserve mini-batch snapshot and evaluation isolation semantics.

---

### Task 1: Progress events and runner integration

**Files:**
- Create: `src/sentiment_agent/experiments/progress.py`
- Modify: `src/sentiment_agent/experiments/runner.py`
- Test: `tests/unit/test_progress.py`

**Interfaces:**
- Produces `ProgressEvent`, `ProgressReporter`, `NullProgressReporter`, `RecordingProgressReporter` test fake, and runner reporter injection.

- [ ] Write a failing test asserting train events `[2, 4]`, then dev/test stages, token accumulation, and post-learning experience counts.
- [ ] Run `uv run python -m pytest tests/unit/test_progress.py -v`; expect missing progress module.
- [ ] Implement immutable events, null reporter, safe `_report`, and one update after each completed batch.
- [ ] Run the focused test; expect all pass.
- [ ] Commit with `feat: emit structured experiment progress`.

### Task 2: Rich reporter and CLI switch

**Files:**
- Create: `src/sentiment_agent/reporting/__init__.py`
- Create: `src/sentiment_agent/reporting/progress.py`
- Modify: `src/sentiment_agent/cli.py`
- Modify: `README.md`
- Test: `tests/unit/test_rich_progress.py`
- Test: `tests/unit/test_cli_v1.py`

**Interfaces:**
- Produces `RichProgressReporter`, CLI `--progress/--no-progress` defaulting to enabled.

- [ ] Write failing in-memory Console tests for stage, sample count, batch count, token and experience rendering; add CLI help assertion for `--no-progress`.
- [ ] Run focused tests; expect missing Rich reporter or option.
- [ ] Implement a transient Rich Progress context owned by the reporter, stage reset, close support, and CLI injection/finalization.
- [ ] Document normal and disabled commands.
- [ ] Run focused tests, full `uv run python -m pytest -q`, and config validation; expect success.
- [ ] Commit with `feat: show live experiment progress`.
