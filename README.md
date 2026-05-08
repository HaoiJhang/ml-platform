# ML Platform Research Prototype

This repository contains a local, single-user research prototype for tabular ML
workflows: data loading, EDA, cleaning, model training, evaluation, result
storage, and LLM-assisted reporting.

## Development

Use `uv` for the Python environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python /usr/bin/python3
UV_CACHE_DIR=.uv-cache uv run pytest
```

Feature work is developed on small `dev/*` branches, committed after tests pass,
and merged back to `main`.
