# Development Rules

## Python environment

- Use `uv` as the only Python environment and dependency manager.

## Testing

- Follow test-driven development: add a failing test before production code.
- Every public function or method needs an isolated unit test.
- Run the complete offline integration workflow before claiming completion.
- Tests must not call external APIs or require real credentials by default.
