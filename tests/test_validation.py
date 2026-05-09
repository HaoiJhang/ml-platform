import pandas as pd

from ml_platform.validation import build_recommendations, validate_postrun, validate_preflight


def test_preflight_detects_blockers_and_leakage() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "copy_target": [0, 1, 0, 1, 0, 1],
            "feature": [1, 2, 1, 2, 1, 2],
        }
    )

    validation = validate_preflight(
        df=df,
        target="target",
        task_type="classification",
        excluded_columns=[],
        priority_metric="roc_auc",
        high_missing_threshold=0.9,
    )

    codes = {issue.code for issue in validation.issues}
    assert validation.ok_to_run
    assert "possible_leakage" in codes


def test_postrun_detects_generalization_gap_and_builds_recommendations() -> None:
    postrun = validate_postrun(
        metrics={
            "accuracy": 0.62,
            "f1_weighted": 0.61,
            "precision_weighted": 0.63,
            "recall_weighted": 0.62,
            "roc_auc": 0.70,
            "train_accuracy": 0.93,
            "train_f1_weighted": 0.92,
            "train_precision_weighted": 0.93,
            "train_recall_weighted": 0.93,
            "train_roc_auc": 0.98,
        },
        prediction_sample=pd.DataFrame({"prediction": [0, 1]}),
        task_type="classification",
        priority_metric="recall_weighted",
        trainer_name="flaml",
        optimization_metric_used="accuracy",
        feature_importance=[{"feature": "x", "importance": 1.0}],
        report_mode="rule_based",
    )

    recommendations = build_recommendations(
        preflight=validate_preflight(
            df=pd.DataFrame({"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]}),
            target="target",
            task_type="classification",
            excluded_columns=[],
            priority_metric="recall_weighted",
            high_missing_threshold=0.9,
        ),
        postrun=postrun,
    )

    assert any(issue.code == "generalization_gap" for issue in postrun.issues)
    assert recommendations.next_steps
