# Parameterized Python Experiment Runner and Logging Design

## Goal

Replace the Qwen-specific PowerShell launcher with a reusable Python entry point that can switch datasets and important experiment parameters from PyCharm or the command line, while producing useful console and file logs without exposing secrets or sample content.

## Entry Point and Parameters

`scripts/run_experiment.py` is a thin Typer application. It accepts train and test paths, optional sample limits, experiment name, model endpoint settings, API-key environment variable, local embedding model/path/device/batch size, checkpoints, reflection and cross-lingual flags, strategy, retrieval count, seed, output root, log level, log file, and progress interval.

The runner loads `.env` without printing values, validates parameters through `AppConfig`, and invokes the existing workflow directly. YAML-based CLI commands remain supported.

## Reproducibility

`ExperimentConfig` stores `train_limit`, `test_limit`, and `progress_every`, so limits and logging cadence become part of the configuration hash and saved expanded configuration. Input order is deterministic; limits select the first N validated rows. A timestamped run root avoids accidental output overwrite.

## Workflow Logging

A logging helper configures one console handler and an optional UTF-8 file handler. The workflow logs experiment identity, model and embedding selection, output directory, dataset paths and selected counts, initialization milestones, and completion summary. The experiment runner logs checkpoint start/completion with Accuracy, Macro-F1, and experience count, plus training progress every configured number of samples.

Logs never include API keys, environment values, raw text, prompts, or complete model responses. Exceptions include stack traces in the log and propagate with a nonzero process exit.

## Architecture

The Python launcher constructs an `AppConfig` and calls a new `run_config(config, ...)` workflow function. Existing `run_from_config(path, ...)` remains a compatibility wrapper. This avoids temporary YAML files and ensures PyCharm and CLI execution share the same experiment implementation.

## Testing

Tests are written before production changes. Unit tests cover argument-to-config mapping, `.env` loading without secret logging, limits, logging handlers, checkpoint/progress messages, and backward-compatible YAML execution. The complete offline integration workflow and coverage threshold remain mandatory. A final real five-sample Qwen run verifies the launcher and generated log without unnecessary API cost.
