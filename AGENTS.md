# Development Rules

## Python environment

- Use `uv` as the only Python environment and dependency manager.
- Declare runtime and development dependencies in `pyproject.toml`.
- Commit `uv.lock` and use `uv sync --extra dev` to create or update `.venv`.
- Run Python tools through `uv run`, for example `uv run pytest` and `uv run python -m sentiment_agent.cli`.
- Do not use `pip install`, Conda environment mutation, Poetry, or manually managed virtual environments for this project.
- CI and documented reproduction commands must use the locked environment via `uv sync --frozen --extra dev`.

## Testing

- Follow test-driven development: add a failing test before production code.
- Every public function or method needs an isolated unit test.
- Run the complete offline integration workflow before claiming completion.
- Tests must not call external APIs or require real credentials by default.
