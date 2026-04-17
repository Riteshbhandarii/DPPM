import numpy as np
import pandas as pd

from src import random_forest_serving


class FakeTree:
    def __init__(self, outputs):
        self.outputs = np.asarray(outputs, dtype=float)

    def predict(self, transformed_input):
        return self.outputs[: len(transformed_input)]


class FakePreprocessor:
    def transform(self, input_frame):
        return np.asarray(input_frame[["feature_a", "feature_b"]], dtype=float)


class FakeForest:
    def __init__(self):
        self.estimators_ = [
            FakeTree([100.0, 200.0]),
            FakeTree([110.0, 210.0]),
            FakeTree([120.0, 220.0]),
        ]


class FakePipeline:
    def __init__(self):
        self.named_steps = {
            "preprocessor": FakePreprocessor(),
            "model": FakeForest(),
        }

    def predict(self, input_frame):
        return np.asarray(input_frame["feature_a"], dtype=float) + np.asarray(input_frame["feature_b"], dtype=float)


def build_bundle():
    return {
        "metadata": {
            "feature_names": ["feature_a", "feature_b"],
            "held_out_test_metrics": {"test_RMSE": 10.0, "test_MAE": 4.0},
            "trusted_validation_metrics": {"validation_RMSE": 12.0, "validation_MAE": 5.0},
        },
        "model": FakePipeline(),
    }


def test_ensure_feature_frame_reorders_and_drops_extra_columns():
    rows = [{"feature_b": 2, "feature_a": 1, "extra": 999}]

    frame = random_forest_serving.ensure_feature_frame(rows, ["feature_a", "feature_b"])

    assert frame.columns.tolist() == ["feature_a", "feature_b"]
    assert frame.iloc[0].to_dict() == {"feature_a": 1, "feature_b": 2}


def test_ensure_feature_frame_raises_for_missing_columns():
    rows = [{"feature_a": 1}]

    try:
        random_forest_serving.ensure_feature_frame(rows, ["feature_a", "feature_b"])
    except ValueError as exc:
        assert "feature_b" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing feature_b")


def test_bundle_error_scale_prefers_test_rmse():
    metadata = {
        "held_out_test_metrics": {"test_RMSE": 62.5, "test_MAE": 20.0},
        "trusted_validation_metrics": {"validation_RMSE": 70.0, "validation_MAE": 30.0},
    }

    assert random_forest_serving.bundle_error_scale(metadata) == 62.5


def test_bundle_error_scale_falls_back_to_validation_mae():
    metadata = {
        "held_out_test_metrics": {},
        "trusted_validation_metrics": {"validation_MAE": 18.0},
    }

    assert random_forest_serving.bundle_error_scale(metadata) == 18.0


def test_predict_price_ranges_returns_expected_columns():
    bundle = build_bundle()
    rows = [
        {"feature_a": 100.0, "feature_b": 10.0},
        {"feature_a": 200.0, "feature_b": 20.0},
    ]

    predictions = random_forest_serving.predict_price_ranges(bundle, rows)

    assert predictions.columns.tolist() == [
        "predicted_price",
        "price_range_low",
        "price_range_high",
        "range_width",
        "ensemble_range_low",
        "ensemble_range_high",
        "ensemble_range_width",
        "uncertainty_source",
    ]
    assert predictions["predicted_price"].round(2).tolist() == [110.0, 220.0]
    assert all(predictions["price_range_low"] >= 0.0)
    assert all(predictions["range_width"] > predictions["ensemble_range_width"])
    assert set(predictions["uncertainty_source"]) == {"held_out_rmse"}


def test_predict_price_ranges_rejects_invalid_quantiles():
    bundle = build_bundle()

    try:
        random_forest_serving.predict_price_ranges(
            bundle,
            [{"feature_a": 100.0, "feature_b": 10.0}],
            lower_quantile=0.9,
            upper_quantile=0.1,
        )
    except ValueError as exc:
        assert "lower_quantile" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid quantiles")
