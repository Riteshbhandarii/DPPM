import pandas as pd
from fastapi import HTTPException
from unittest.mock import patch

import app.fastapi_app as fastapi_app


def test_health_check_uses_model_bundle_metadata():
    with patch.object(
        fastapi_app,
        "MODEL_BUNDLE",
        {
            "metadata": {
                "model_type": "random_forest",
                "bundle_split": "full_data",
                "config_name": "rf_config",
            }
        },
    ):
        result = fastapi_app.health_check()

    assert result == {
        "status": "ok",
        "model_type": "random_forest",
        "bundle_split": "full_data",
        "config_name": "rf_config",
    }


def test_model_info_returns_metadata():
    metadata = {"model_type": "random_forest", "feature_variant": "demo"}
    with patch.object(fastapi_app, "MODEL_BUNDLE", {"metadata": metadata}):
        assert fastapi_app.model_info() == metadata


def test_predict_returns_prediction_payload():
    def fake_predict_price_ranges(bundle, rows, lower_quantile, upper_quantile):
        assert rows == [{"feature_a": 1}]
        assert lower_quantile == 0.1
        assert upper_quantile == 0.9
        return pd.DataFrame(
            [
                {
                    "predicted_price": 123.456,
                    "price_range_low": 100.0,
                    "price_range_high": 150.0,
                    "range_width": 50.0,
                }
            ]
        )

    with patch.object(
        fastapi_app,
        "MODEL_BUNDLE",
        {
            "metadata": {
                "model_type": "random_forest",
                "bundle_split": "full_data",
                "feature_variant": "demo_variant",
                "config_name": "rf_config",
            }
        },
    ), patch.object(fastapi_app, "predict_price_ranges", fake_predict_price_ranges):
        request = fastapi_app.PredictionRequest(rows=[{"feature_a": 1}])
        result = fastapi_app.predict(request)

    assert result["model_type"] == "random_forest"
    assert result["bundle_split"] == "full_data"
    assert result["feature_variant"] == "demo_variant"
    assert result["config_name"] == "rf_config"
    assert result["predictions"][0]["predicted_price"] == 123.46


def test_predict_translates_value_error_to_http_400():
    def fake_predict_price_ranges(bundle, rows, lower_quantile, upper_quantile):
        raise ValueError("bad input")

    with patch.object(
        fastapi_app,
        "MODEL_BUNDLE",
        {
            "metadata": {
                "model_type": "random_forest",
                "bundle_split": "full_data",
                "feature_variant": "demo_variant",
                "config_name": "rf_config",
            }
        },
    ), patch.object(fastapi_app, "predict_price_ranges", fake_predict_price_ranges):
        request = fastapi_app.PredictionRequest(rows=[{"feature_a": 1}])

        try:
            fastapi_app.predict(request)
        except HTTPException as exc:
            assert exc.status_code == 400
            assert exc.detail == "bad input"
        else:
            raise AssertionError("Expected HTTPException")
