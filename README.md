# ML Platform Research Prototype

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

This repository contains a Streamlit research prototype for tabular ML
workflows: CSV upload, EDA, cleaning, model training, evaluation, result
storage, and optional LLM-assisted reporting.

The first version is intentionally small: it does not include authentication,
multi-user workflows, distributed training, or production monitoring.

## Development

Use `uv` for the Python environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
UV_CACHE_DIR=.uv-cache uv run pytest
```

Start the local app with:

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run app.py
```

The app is available at `http://localhost:8501` by default.

The app writes experiment artifacts under `runs/`. Each completed run gets a
directory containing `config.json`, `eda_summary.json`, `cleaning_log.json`,
`metrics.json`, `feature_importance.json`, `prediction_sample.csv`,
`model.joblib`, and `report.md`. Run metadata is also recorded in
`runs/runs.sqlite`.

LLM report generation is optional. If `OPENAI_API_KEY` is not configured, the
pipeline still completes and writes a local rule-based report.

## Streamlit Community Cloud

Deploy from GitHub with these settings:

- Repository: `HaoiJhang/ml-platform`
- Branch: the branch you want to deploy
- Main file path: `app.py`
- Python version: `3.11`

The repository includes `uv.lock`, which Streamlit Community Cloud recognizes
as the Python dependency file. The project declares Python `>=3.11,<3.12`, so
choose Python 3.11 in the deployment advanced settings instead of the
Community Cloud default.

If you want the hosted app to use an OpenAI-compatible API key without asking
each user to enter one, add secrets in Streamlit Community Cloud instead of
committing them to the repository:

```toml
OPENAI_API_KEY = "..."
OPENAI_BASE_URL = "..."
OPENAI_MODEL = "gpt-4o-mini"
```

For public deployments, the app does not persist API keys entered in the UI.
Local file persistence can be re-enabled for local development only by setting
`ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG=1`.

Feature work is developed on small `dev/*` branches, committed after tests pass,
and merged back to `main`.
