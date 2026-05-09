from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from ml_platform.artifacts import artifact_to_dict
from ml_platform.automl import train_model
from ml_platform.cleaning import CleanConfig, clean_and_split
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.llm_report import generate_report_result
from ml_platform.planner import suggest_plan
from ml_platform.config import Settings
from ml_platform.storage import RunStorage
from ml_platform.validation import build_recommendations, validate_postrun, validate_preflight


def test_end_to_end_classification_smoke(tmp_path: Path) -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["dirty_constant"] = 1
    df.loc[df.index[:5], "sepal length (cm)"] = None

    eda = generate_eda_summary(df, target="species")
    cleaned = clean_and_split(df, CleanConfig(target="species", task_type="classification", test_size=0.25))
    trained = train_model(cleaned, time_budget=2)
    metrics, predictions = evaluate_model(trained.model, cleaned, "classification")

    settings = Settings(
        runs_dir=tmp_path,
        data_dir=tmp_path / "data",
        openai_api_key=None,
        openai_base_url=None,
        openai_model="test",
    )
    plan = suggest_plan(df, eda, settings, user_brief="predict species with a quick baseline and optimize f1")
    preflight = validate_preflight(
        df=df,
        target="species",
        task_type="classification",
        excluded_columns=[],
        priority_metric="f1_weighted",
        high_missing_threshold=0.9,
    )
    postrun = validate_postrun(
        metrics=metrics,
        prediction_sample=predictions,
        task_type="classification",
        priority_metric="f1_weighted",
        trainer_name=trained.trainer_name,
        optimization_metric_used=trained.optimization_metric_used,
        feature_importance=trained.feature_importance,
        report_mode="rule_based",
    )
    recommendations = build_recommendations(preflight, postrun)
    report, _ = generate_report_result(
        eda,
        cleaned.cleaning_log,
        metrics,
        trained.feature_importance,
        settings,
        plan_suggestion=plan,
        preflight_validation=preflight,
        postrun_validation=postrun,
        recommendations=recommendations,
    )
    storage = RunStorage(tmp_path)
    run = storage.create_run({"target": "species", "task_type": "classification"})
    storage.save_json(run, "plan.json", artifact_to_dict(plan))
    storage.save_json(run, "eda_summary.json", eda)
    storage.save_json(run, "metrics.json", metrics)
    storage.save_json(run, "validation_pre.json", artifact_to_dict(preflight))
    storage.save_json(run, "validation_post.json", artifact_to_dict(postrun))
    storage.save_json(run, "recommendations.json", artifact_to_dict(recommendations))
    storage.save_text(run, "report.md", report)
    storage.save_predictions(run, predictions)
    storage.save_model(run, trained.model)
    storage.record_run(run, metrics, "completed")

    assert metrics["accuracy"] is not None
    assert metrics["train_accuracy"] is not None
    assert (run.path / "model.joblib").exists()
    assert (run.path / "plan.json").exists()
    assert (run.path / "validation_pre.json").exists()
    assert (run.path / "validation_post.json").exists()
    assert (run.path / "report.md").read_text(encoding="utf-8")
