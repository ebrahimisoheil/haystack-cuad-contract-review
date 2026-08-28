# Contributing

Thank you for improving `haystack-cuad-contract-review`.

## Development setup

~~~bash
uv sync --locked --extra dev
uv run --extra dev pytest
uv build
~~~

Tests run in deterministic mode and use an isolated Witdem telemetry endpoint.
They do not require provider credentials and must not create runs in a local
developer dashboard.

## Pull requests

- Keep provider/model IDs in `model_routing.yaml`, not in pipeline components.
- Keep Haystack responsible for orchestration and LiteLLM responsible for model access.
- Add or update tests for behavior changes.
- Do not commit API keys, downloaded contracts, CUAD subsets, scans, or run output.
- Update the README when commands, environment variables, or compatibility change.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.
