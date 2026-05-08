# ML Platform Research Prototype

This repository contains a local, single-user research prototype for tabular ML
workflows: CSV upload, EDA, cleaning, model training, evaluation, result
storage, and optional LLM-assisted reporting.

The first version is intentionally small: it does not include authentication,
multi-user workflows, cloud deployment, distributed training, or production
monitoring.

## Development

Use `uv` for the Python environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python /usr/bin/python3
UV_CACHE_DIR=.uv-cache uv run pytest
```

Start the local app with:

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run app.py
```

The included Streamlit config binds the server to `127.0.0.1`, so the app is
intended for local access at `http://localhost:8501` only.

The app writes experiment artifacts under `runs/`. Each completed run gets a
directory containing `config.json`, `eda_summary.json`, `cleaning_log.json`,
`metrics.json`, `feature_importance.json`, `prediction_sample.csv`,
`model.joblib`, and `report.md`. Run metadata is also recorded in
`runs/runs.sqlite`.

LLM report generation is optional. If `OPENAI_API_KEY` is not configured, the
pipeline still completes and writes a local rule-based report.

Feature work is developed on small `dev/*` branches, committed after tests pass,
and merged back to `main`.
