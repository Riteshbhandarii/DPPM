"""Local SHAP explanation helpers for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from app.ui_helpers import feature_display_name


def to_dense_float_array(matrix):
    """Convert sparse/dense transformer output into a float ndarray."""

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def raw_feature_name(transformed_name, raw_features):
    """Map transformed column names back to raw feature names."""

    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")

    if transformed_name.startswith("cat__"):
        remainder = transformed_name.removeprefix("cat__")
        matches = [
            feature
            for feature in raw_features
            if remainder == feature or remainder.startswith(feature + "_")
        ]
        if matches:
            return max(matches, key=len)
        return remainder

    return transformed_name


def group_shap_values(shap_values, transformed_feature_names, raw_feature_names, row_index):
    """Aggregate transformed-feature SHAP values back to raw features."""

    grouped_indices = {}
    for idx, transformed_name in enumerate(transformed_feature_names):
        grouped_name = raw_feature_name(transformed_name, raw_feature_names)
        grouped_indices.setdefault(grouped_name, []).append(idx)

    grouped_shap = pd.DataFrame(index=row_index)
    for feature_name, indices in grouped_indices.items():
        grouped_shap[feature_name] = shap_values[:, indices].sum(axis=1)

    return grouped_shap


def local_explanation_table(feature_frame, grouped_shap, row_index=0, top_k=None):
    """Build a sorted local SHAP table for one prediction row."""

    local_df = pd.DataFrame(
        {
            "feature_name": grouped_shap.columns,
            "feature_value": [feature_frame.iloc[row_index][column] for column in grouped_shap.columns],
            "shap_value": grouped_shap.iloc[row_index].to_numpy(),
        }
    )
    local_df["abs_shap_value"] = local_df["shap_value"].abs()
    local_df = local_df.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    if top_k is not None:
        local_df = local_df.head(top_k).reset_index(drop=True)
    return local_df


def compute_local_shap_explanation(bundle, reference_rows, prediction_input, background_size=25):
    """Compute a live local SHAP explanation for the current submitted row."""

    model_pipeline = bundle["model"]
    forest = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    metadata = bundle["metadata"]
    feature_names = list(metadata["feature_names"])

    feature_frame = pd.DataFrame([prediction_input], columns=feature_names)
    background_frame = reference_rows.loc[:, feature_names].sample(
        n=min(background_size, len(reference_rows)),
        random_state=42,
    )

    transformed_matrix = to_dense_float_array(preprocessor.transform(feature_frame))
    background_transformed = to_dense_float_array(preprocessor.transform(background_frame))
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.Explainer(
        forest.predict,
        background_transformed,
        algorithm="permutation",
    )
    explanation = explainer(
        transformed_matrix,
        max_evals=2 * transformed_matrix.shape[1] + 1,
    )

    shap_values = np.asarray(explanation.values, dtype=float)
    base_value = float(np.asarray(explanation.base_values, dtype=float).reshape(-1)[0])
    grouped_shap = group_shap_values(
        shap_values=shap_values,
        transformed_feature_names=transformed_feature_names,
        raw_feature_names=feature_names,
        row_index=feature_frame.index,
    )
    local_df = local_explanation_table(feature_frame, grouped_shap, row_index=0)
    local_df["display_name"] = local_df["feature_name"].map(feature_display_name)
    total_effect = float(grouped_shap.iloc[0].sum())
    predicted_price = float(model_pipeline.predict(feature_frame)[0])
    reconstruction_error = abs((base_value + total_effect) - predicted_price)
    return {
        "base_value": base_value,
        "total_effect": total_effect,
        "predicted_price": predicted_price,
        "reconstruction_error": reconstruction_error,
        "local_df": local_df,
    }
