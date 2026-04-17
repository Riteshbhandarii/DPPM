"""Simple Streamlit UI for the final random-forest price estimator."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Add the repository root so local imports work when the app is launched directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.random_forest_serving import load_random_forest_bundle, predict_price_ranges


# Use the production-ready bundle by default.
BUNDLE_DIR = Path("artifacts/random_forest_final/full_data_bundle")


@st.cache_resource
def load_bundle():
    """Cache the saved model bundle so reruns stay fast."""

    return load_random_forest_bundle(BUNDLE_DIR)


def build_example_label(row):
    """Create a short label for one saved reference row."""

    return (
        f"{row.get('brand', 'Unknown')} | "
        f"{row.get('model', 'Unknown')} | "
        f"{row.get('part_name', 'Unknown part')}"
    )


def initialize_form_state(reference_rows, feature_names):
    """Seed the form state from the first saved reference row."""

    if "input_values" in st.session_state:
        return

    default_row = reference_rows.iloc[0]
    st.session_state["input_values"] = {
        feature_name: default_row.get(feature_name) for feature_name in feature_names
    }


def render_input(feature_name, reference_rows):
    """Render one simple widget based on the reference column dtype."""

    feature_series = reference_rows[feature_name]
    current_value = st.session_state["input_values"].get(feature_name)
    label = feature_name.replace("_", " ").title()

    if pd.api.types.is_bool_dtype(feature_series):
        return st.checkbox(label, value=bool(current_value))

    if pd.api.types.is_numeric_dtype(feature_series):
        numeric_series = pd.to_numeric(feature_series, errors="coerce")
        min_value = float(numeric_series.min()) if numeric_series.notna().any() else 0.0
        max_value = float(numeric_series.max()) if numeric_series.notna().any() else 0.0
        default_value = float(current_value) if pd.notna(current_value) else min_value
        step = 1.0 if float(default_value).is_integer() else 0.1
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=step,
        )

    choices = feature_series.dropna().astype(str).value_counts().index.tolist()
    if choices and len(choices) <= 60:
        default_choice = str(current_value) if pd.notna(current_value) else choices[0]
        index = choices.index(default_choice) if default_choice in choices else 0
        return st.selectbox(label, options=choices, index=index)

    return st.text_input(label, value="" if pd.isna(current_value) else str(current_value))


def main():
    """Render the simple local UI."""

    st.set_page_config(page_title="DPPM Price Estimator", layout="wide")
    bundle = load_bundle()
    metadata = bundle["metadata"]
    reference_rows = pd.read_csv(BUNDLE_DIR / "reference_rows.csv")
    feature_names = list(metadata["feature_names"])
    initialize_form_state(reference_rows, feature_names)

    st.title("DPPM Price Estimator")
    st.caption("Simple Streamlit UI for the final random-forest model.")

    st.subheader("Load example")
    example_labels = reference_rows.apply(build_example_label, axis=1).tolist()
    selected_label = st.selectbox("Reference row", options=example_labels)
    selected_index = example_labels.index(selected_label)
    if st.button("Use example", type="primary"):
        selected_row = reference_rows.iloc[selected_index]
        st.session_state["input_values"] = {
            feature_name: selected_row.get(feature_name) for feature_name in feature_names
        }
        st.rerun()

    st.subheader("Inputs")
    updated_values = {}
    for feature_name in feature_names:
        updated_values[feature_name] = render_input(feature_name, reference_rows)
    st.session_state["input_values"] = updated_values

    st.subheader("Prediction")
    prediction = predict_price_ranges(bundle, [updated_values]).iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Point estimate", f"{prediction['predicted_price']:.2f} EUR")
        st.metric("Lower bound", f"{prediction['price_range_low']:.2f} EUR")
    with col2:
        st.metric("Upper bound", f"{prediction['price_range_high']:.2f} EUR")
        st.metric("Range width", f"{prediction['range_width']:.2f} EUR")

    st.info(
        "The range is based on the spread of the individual tree predictions in the saved random forest."
    )


if __name__ == "__main__":
    main()
