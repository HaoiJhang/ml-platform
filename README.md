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
UV_CACHE_DIR=.uv-cache uv run streamlit run app_beamer_v5.py
```

The app is available at `http://localhost:8501` by default.

The app writes experiment artifacts under `runs/`. Each completed run gets a
directory containing `config.json`, `eda_summary.json`, `cleaning_log.json`,
`metrics.json`, `feature_importance.json`, `prediction_sample.csv`,
`autogluon_predictor/`, `autogluon_predictor.zip`, `leaderboard.json`,
`fit_summary.json`, and `report.md`. Run metadata is also recorded in
`runs/runs.sqlite`.

After a model is trained in the current browser session, the results page also
includes an `Inference` frame. Upload a CSV with the same feature columns used
for training to preview predictions. Classification runs show predicted labels
plus per-class probabilities; regression runs show predicted values. This first
version does not load historical runs for inference and does not write a new
prediction artifact.

LLM report generation is optional. If `LLM_API_KEY` is not configured, the
pipeline still completes and writes a local rule-based report.

## Streamlit Community Cloud

Deploy from GitHub with these settings:

- Repository: `HaoiJhang/ml-platform`
- Branch: the branch you want to deploy
- Main file path: `app_beamer_v5.py`
- Python version: `3.11`

The repository includes `uv.lock`, which Streamlit Community Cloud recognizes
as the Python dependency file. The project declares Python `>=3.11,<3.12`, so
choose Python 3.11 in the deployment advanced settings instead of the
Community Cloud default.

To keep the hosted UI aligned with the local baseline, the project pins
`streamlit==1.50.0` instead of floating to newer widget layouts.

If you want the hosted app to use a compatible LLM API key without asking
each user to enter one, add secrets in Streamlit Community Cloud instead of
committing them to the repository:

```toml
LLM_API_KEY = "..."
LLM_BASE_URL = "..."
LLM_MODEL = "gpt-4o-mini"
```

By default, the app can persist API keys entered in the UI to the local
`.ml_platform.local.json` file when the user enables
`Remember LLM settings on this device`. On shared or public deployments, disable
this behavior with `ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG=0` and prefer host
secrets instead. The app also accepts legacy provider-prefixed variable names for compatibility.

Feature work is developed on small `dev/*` branches, committed after tests pass,
and merged back to `main`.
