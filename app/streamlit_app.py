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
OPERATOR_FIELDS = [
    "part_name",
    "brand",
    "model",
    "category",
    "subcategory",
    "quality_grade",
    "repair_status",
    "mileage",
    "year_start",
    "year_end",
    "oem_number",
]


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


def derive_year_fields(values):
    """Fill the year helper features expected by the model."""

    year_start = values.get("year_start")
    year_end = values.get("year_end")
    if pd.notna(year_start) and pd.notna(year_end):
        year_start = int(year_start)
        year_end = int(year_end)
        values["year_start"] = year_start
        values["year_end"] = year_end
        values["year_span"] = year_end - year_start
        values["year_mid"] = (year_start + year_end) / 2
    return values


def derive_mileage_fields(values):
    """Fill the mileage helper flag expected by the model."""

    mileage = values.get("mileage")
    values["mileage_missing_flag"] = bool(pd.isna(mileage))
    return values


def build_full_input(reference_rows, visible_values):
    """Start from the closest saved row so hidden aggregate fields stay coherent."""

    candidate_mask = pd.Series(True, index=reference_rows.index)
    for key in ["part_name", "brand", "model", "category", "subcategory"]:
        value = visible_values.get(key)
        if pd.isna(value) or value in {"", None}:
            continue
        candidate_mask &= reference_rows[key].astype(str) == str(value)

    candidate_rows = reference_rows.loc[candidate_mask]
    if candidate_rows.empty:
        for key in ["brand", "model", "category", "part_name"]:
            value = visible_values.get(key)
            if pd.isna(value) or value in {"", None}:
                continue
            narrowed = reference_rows.loc[reference_rows[key].astype(str) == str(value)]
            if not narrowed.empty:
                candidate_rows = narrowed
                break

    if candidate_rows.empty:
        candidate_rows = reference_rows

    base_row = candidate_rows.iloc[0].to_dict()
    base_row.update(visible_values)
    derive_year_fields(base_row)
    derive_mileage_fields(base_row)
    return base_row


def comparable_market_range(reference_rows, visible_values, predicted_price):
    """Estimate a practical price range from similar saved rows."""

    hierarchy = [
        ["part_name", "brand", "model", "category", "subcategory", "quality_grade", "repair_status"],
        ["part_name", "brand", "model", "category", "subcategory"],
        ["part_name", "brand", "model"],
        ["part_name", "brand"],
        ["category", "subcategory"],
    ]

    selected = None
    selected_keys = None
    for keys in hierarchy:
        mask = pd.Series(True, index=reference_rows.index)
        for key in keys:
            value = visible_values.get(key)
            if pd.isna(value) or value in {"", None}:
                continue
            mask &= reference_rows[key].astype(str) == str(value)

        candidate_rows = reference_rows.loc[mask].copy()
        if candidate_rows.empty:
            continue

        mileage = visible_values.get("mileage")
        if pd.notna(mileage) and "mileage" in candidate_rows:
            candidate_rows["mileage_distance"] = (candidate_rows["mileage"] - float(mileage)).abs()
            candidate_rows = candidate_rows.sort_values("mileage_distance")

        if len(candidate_rows) >= 5:
            selected = candidate_rows.head(50).copy()
            selected_keys = keys
            break

    if selected is None:
        return {
            "range_low": max(float(predicted_price) - 75.0, 0.0),
            "range_high": float(predicted_price) + 75.0,
            "range_width": 150.0,
            "comparable_count": 0,
            "range_source": "fallback_minimum_band",
            "matched_on": "",
        }

    prices = pd.to_numeric(selected["price"], errors="coerce").dropna()
    low = float(prices.quantile(0.10))
    high = float(prices.quantile(0.90))

    # Keep a minimum practical width so repeated identical rows do not imply fake certainty.
    minimum_half_width = max(float(predicted_price) * 0.10, 15.0)
    low = min(low, float(predicted_price) - minimum_half_width)
    high = max(high, float(predicted_price) + minimum_half_width)
    low = max(low, 0.0)

    return {
        "range_low": float(low),
        "range_high": float(high),
        "range_width": float(high - low),
        "comparable_count": int(len(prices)),
        "range_source": "comparable_rows",
        "matched_on": ", ".join(selected_keys),
    }


def initialize_form_state(reference_rows):
    """Seed the form state from the first saved reference row."""

    if "input_values" in st.session_state:
        return

    default_row = reference_rows.iloc[0]
    st.session_state["input_values"] = {
        feature_name: default_row.get(feature_name) for feature_name in OPERATOR_FIELDS
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
    initialize_form_state(reference_rows)

    st.title("DPPM Price Estimator")
    st.caption("Operator-facing UI for the final random-forest dismantling-part price model.")

    st.subheader("Load example")
    example_labels = reference_rows.apply(build_example_label, axis=1).tolist()
    selected_label = st.selectbox("Reference row", options=example_labels)
    selected_index = example_labels.index(selected_label)
    if st.button("Use example", type="primary"):
        selected_row = reference_rows.iloc[selected_index]
        st.session_state["input_values"] = {
            feature_name: selected_row.get(feature_name) for feature_name in OPERATOR_FIELDS
        }
        st.rerun()

    st.subheader("Inputs")
    updated_values = {}
    for feature_name in OPERATOR_FIELDS:
        updated_values[feature_name] = render_input(feature_name, reference_rows)
    st.session_state["input_values"] = updated_values

    full_input = build_full_input(reference_rows, updated_values)
    prediction_input = {feature_name: full_input.get(feature_name) for feature_name in feature_names}

    st.subheader("Prediction")
    prediction = predict_price_ranges(bundle, [prediction_input]).iloc[0]
    market_range = comparable_market_range(
        reference_rows=reference_rows,
        visible_values=updated_values,
        predicted_price=float(prediction["predicted_price"]),
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Point estimate", f"{prediction['predicted_price']:.2f} EUR")
        st.metric("Lower bound", f"{market_range['range_low']:.2f} EUR")
    with col2:
        st.metric("Upper bound", f"{market_range['range_high']:.2f} EUR")
        st.metric("Range width", f"{market_range['range_width']:.2f} EUR")

    st.info(
        "The displayed range is based on similar historical rows when available, with a minimum width floor to avoid fake certainty."
    )

    with st.expander("Technical Details"):
        st.write("Hidden model fields are filled from the closest saved reference row and simple derived values.")
        st.metric("Range source", str(market_range["range_source"]))
        st.metric("Comparable rows used", int(market_range["comparable_count"]))
        st.caption(f"Matched on: {market_range['matched_on'] or 'fallback'}")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.metric("Displayed low", f"{market_range['range_low']:.2f} EUR")
            st.metric("Displayed high", f"{market_range['range_high']:.2f} EUR")
            st.metric("Ensemble low", f"{prediction['ensemble_range_low']:.2f} EUR")
            st.metric("Ensemble high", f"{prediction['ensemble_range_high']:.2f} EUR")
        with tech_col2:
            st.metric("Calibrated low", f"{prediction['price_range_low']:.2f} EUR")
            st.metric("Calibrated high", f"{prediction['price_range_high']:.2f} EUR")
            st.metric("Ensemble width", f"{prediction['ensemble_range_width']:.2f} EUR")
            st.metric("Range source", str(prediction["uncertainty_source"]))
        st.dataframe(
            pd.DataFrame({"operator_input": updated_values}).T,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
