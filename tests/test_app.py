from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
from streamlit.testing.v1 import AppTest


def test_local_llm_config_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", raising=False)
    assert app_module._local_llm_config_enabled() is True


def test_local_llm_config_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", "0")
    assert app_module._local_llm_config_enabled() is False


def test_data_flow_selection_does_not_drop_latest_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    run_training.click()
    app.run(timeout=120)

    assert any(subheader.value == "Run results" for subheader in app.subheader)

    data_flow_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Inspect data flow step")
    data_flow_select.set_value(data_flow_select.options[1])
    app.run(timeout=120)

    assert any(subheader.value == "Run results" for subheader in app.subheader)
    selected_again = next(selectbox for selectbox in app.selectbox if selectbox.label == "Inspect data flow step")
    assert selected_again.value == selected_again.options[1]
