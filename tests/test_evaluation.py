import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml_platform.cleaning import CleanConfig, CleanedData
from ml_platform.evaluation import evaluate_model


def _cleaned(task_type: str, y_train: pd.Series, y_test: pd.Series) -> CleanedData:
    X_train = pd.DataFrame({"x": range(len(y_train))})
    X_test = pd.DataFrame({"x": range(len(y_test))})
    identity = FunctionTransformer(validate=False)
    return CleanedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=identity,
        feature_columns=["x"],
        numeric_features=["x"],
        categorical_features=[],
        cleaning_log=[],
        config=CleanConfig(target="target", task_type=task_type),
    )


def test_classification_metrics() -> None:
    cleaned = _cleaned("classification", pd.Series([0, 1, 0, 1]), pd.Series([0, 1]))
    model = Pipeline([("identity", FunctionTransformer(validate=False)), ("model", DummyClassifier(strategy="most_frequent"))])
    model.fit(cleaned.X_train, cleaned.y_train)

    metrics, sample = evaluate_model(model, cleaned, "classification")

    assert set(metrics) == {
        "accuracy",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
        "roc_auc",
        "train_accuracy",
        "train_f1_weighted",
        "train_precision_weighted",
        "train_recall_weighted",
        "train_roc_auc",
    }
    assert len(sample) == 2


def test_regression_metrics() -> None:
    cleaned = _cleaned("regression", pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0]))
    model = Pipeline([("identity", FunctionTransformer(validate=False)), ("model", DummyRegressor(strategy="mean"))])
    model.fit(cleaned.X_train, cleaned.y_train)

    metrics, sample = evaluate_model(model, cleaned, "regression")

    assert set(metrics) == {"rmse", "mae", "r2", "train_rmse", "train_mae", "train_r2"}
    assert len(sample) == 2
