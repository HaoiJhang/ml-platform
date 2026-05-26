from __future__ import annotations

import pandas as pd
import pytest

from ml_platform.artifacts import FeatureEngineeringOperation
from ml_platform.cleaning import CleanConfig, clean_and_split, prepare_for_training
from ml_platform.inference import predict_with_trained_model


class RecordingModel:
    def __init__(self) -> None:
        self.last_input: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> list[str]:
        self.last_input = X.copy()
        return ["yes"] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"no": [0.2] * len(X), "yes": [0.8] * len(X)})


class RegressionModel:
    def predict(self, X: pd.DataFrame) -> list[float]:
        return [float(index + 1) for index in range(len(X))]


def _cleaned_data() -> object:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )
    return clean_and_split(
        df,
        CleanConfig(target="target", task_type="classification", random_state=3),
    )


def test_inference_requires_training_features() -> None:
    cleaned = _cleaned_data()

    with pytest.raises(ValueError, match="missing required feature columns: segment"):
        predict_with_trained_model(
            model=RecordingModel(),
            cleaned=cleaned,
            new_data=pd.DataFrame({"amount": [12.0]}),
            task_type="classification",
        )


def test_inference_ignores_target_for_model_input_and_preserves_extra_columns() -> None:
    cleaned = _cleaned_data()
    model = RecordingModel()
    new_data = pd.DataFrame(
        {
            "amount": [12.0, 22.0],
            "segment": ["a", "z"],
            "target": [1, 0],
            "row_id": ["r1", "r2"],
        }
    )

    result = predict_with_trained_model(
        model=model,
        cleaned=cleaned,
        new_data=new_data,
        task_type="classification",
    )

    assert list(model.last_input.columns) == ["amount", "segment"]
    assert result["row_id"].tolist() == ["r1", "r2"]
    assert result["target"].tolist() == [1, 0]
    assert result["prediction"].tolist() == ["yes", "yes"]
    assert result["probability__no"].tolist() == [0.2, 0.2]
    assert result["probability__yes"].tolist() == [0.8, 0.8]


def test_regression_inference_outputs_prediction_without_probabilities() -> None:
    cleaned = _cleaned_data()

    result = predict_with_trained_model(
        model=RegressionModel(),
        cleaned=cleaned,
        new_data=pd.DataFrame({"amount": [12.0], "segment": ["a"]}),
        task_type="regression",
    )

    assert result["prediction"].tolist() == [1.0]
    assert not any(column.startswith("probability__") for column in result.columns)


def test_inference_reuses_fitted_feature_engineering_transformer() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "a", "b", "b", "c", "c"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            random_state=3,
            feature_engineering_operations=[
                FeatureEngineeringOperation(
                    operation="frequency_encoding", source_column="segment"
                )
            ],
        ),
    )
    prepared = prepare_for_training(cleaned)
    model = RecordingModel()

    predict_with_trained_model(
        model=model,
        cleaned=prepared,
        new_data=pd.DataFrame({"segment": ["a", "unknown"], "amount": [15.0, 70.0]}),
        task_type="classification",
    )

    assert "fe__segment__freq" in model.last_input.columns
    expected_frequency = prepared.fitted_preprocessor.frequency_maps_["segment"]["a"]
    assert model.last_input["fe__segment__freq"].tolist() == [expected_frequency, 0.0]
