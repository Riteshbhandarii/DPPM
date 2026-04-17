"""Serving helpers for the final random-forest deployment bundle."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def patch_simple_imputer_compatibility(transformer):
    """Patch older saved SimpleImputer instances for newer sklearn versions."""

    if isinstance(transformer, SimpleImputer):
        if not hasattr(transformer, "_fill_dtype"):
            statistics = getattr(transformer, "statistics_", None)
            if statistics is not None:
                transformer._fill_dtype = np.asarray(statistics).dtype
            else:
                transformer._fill_dtype = np.dtype("float64")
        return

    if isinstance(transformer, Pipeline):
        for _, step in transformer.steps:
            patch_simple_imputer_compatibility(step)
        return

    if isinstance(transformer, ColumnTransformer):
        for _, subtransformer, _ in transformer.transformers_:
            if subtransformer in {"drop", "passthrough"}:
                continue
            patch_simple_imputer_compatibility(subtransformer)


def load_random_forest_bundle(bundle_dir):
    """Load the saved model bundle from disk."""

    bundle_path = Path(bundle_dir)
    metadata = json.loads((bundle_path / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(bundle_path / "model.joblib")
    patch_simple_imputer_compatibility(model)

    return {
        "bundle_dir": bundle_path,
        "metadata": metadata,
        "model": model,
    }


def ensure_feature_frame(rows, feature_names):
    """Validate incoming rows and align them to the trained feature order."""

    frame = pd.DataFrame(rows)
    missing_columns = [column for column in feature_names if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Prediction input is missing required fields: " + ", ".join(sorted(missing_columns))
        )

    extra_columns = [column for column in frame.columns if column not in feature_names]
    if extra_columns:
        frame = frame.drop(columns=extra_columns)

    return frame.loc[:, feature_names].copy()


def predict_price_ranges(bundle, rows, lower_quantile=0.10, upper_quantile=0.90):
    """Return point predictions and a model-based range from the tree ensemble."""

    metadata = bundle["metadata"]
    model_pipeline = bundle["model"]
    feature_names = list(metadata["feature_names"])

    if lower_quantile >= upper_quantile:
        raise ValueError("lower_quantile must be smaller than upper_quantile.")

    input_frame = ensure_feature_frame(rows, feature_names)
    point_predictions = model_pipeline.predict(input_frame)

    preprocessor = model_pipeline.named_steps["preprocessor"]
    forest = model_pipeline.named_steps["model"]
    transformed_input = preprocessor.transform(input_frame)

    tree_predictions = np.vstack([tree.predict(transformed_input) for tree in forest.estimators_])
    lower_bounds = np.quantile(tree_predictions, lower_quantile, axis=0)
    upper_bounds = np.quantile(tree_predictions, upper_quantile, axis=0)

    return pd.DataFrame(
        {
            "predicted_price": np.asarray(point_predictions, dtype=float),
            "price_range_low": np.asarray(lower_bounds, dtype=float),
            "price_range_high": np.asarray(upper_bounds, dtype=float),
            "range_width": np.asarray(upper_bounds - lower_bounds, dtype=float),
        }
    )
