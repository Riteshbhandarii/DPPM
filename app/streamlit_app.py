"""Simple Streamlit UI for the final random-forest price estimator."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add the repository root so local imports work when the app is launched directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shap_utils import compute_local_shap_explanation
from app.ui_helpers import (
    OPERATOR_FIELDS,
    QUALITY_GRADE_OPTIONS,
    build_full_input,
    build_option_label_map,
    build_part_option_catalog,
    clean_display_label,
    comparable_market_range,
    choose_option,
    feature_display_name,
    filter_reference_rows,
    format_feature_value,
    initialize_form_state,
    keep_valid_choice,
    sorted_unique_options,
    sync_dependent_state,
    derive_part_type,
)
from src.random_forest_serving import load_random_forest_bundle, predict_price_ranges

# Re-export helper names so the existing tests keep working via app.streamlit_app.
__all__ = [
    "build_full_input",
    "choose_option",
    "clean_display_label",
    "comparable_market_range",
    "feature_display_name",
    "filter_reference_rows",
    "format_feature_value",
    "initialize_form_state",
    "keep_valid_choice",
    "render_operator_form",
    "sorted_unique_options",
    "st",
    "sync_dependent_state",
]

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


def render_operator_form(reference_rows):
    """Render a filtered operator workflow instead of a flat all-fields form."""

    current_values = dict(st.session_state["input_values"])

    st.subheader("Vehicle")
    vehicle_col1, vehicle_col2, vehicle_col3 = st.columns(3)

    with vehicle_col1:
        brand_options = sorted_unique_options(reference_rows["brand"])
        sync_dependent_state(current_values, "brand", brand_options, "brand_select")
        current_values["brand"] = choose_option(
            "Brand",
            brand_options,
            current_values.get("brand"),
            key="brand_select",
            label_map=build_option_label_map(brand_options),
        )

    model_rows = filter_reference_rows(reference_rows, {"brand": current_values.get("brand")})
    with vehicle_col2:
        model_options = sorted_unique_options(model_rows["model"])
        sync_dependent_state(current_values, "model", model_options, "model_select")
        current_values["model"] = choose_option(
            "Model",
            model_options,
            current_values.get("model"),
            key="model_select",
            label_map=build_option_label_map(model_options),
        )

    year_rows = filter_reference_rows(
        reference_rows,
        {"brand": current_values.get("brand"), "model": current_values.get("model")},
    )
    with vehicle_col3:
        year_pairs = (
            year_rows[["year_start", "year_end"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["year_start", "year_end"])
        )
        year_labels = [
            f"{int(row.year_start)}-{int(row.year_end)}" for row in year_pairs.itertuples(index=False)
        ]
        year_options = ["Any"] + year_labels
        default_year = "Any"
        if pd.notna(current_values.get("year_start")) and pd.notna(current_values.get("year_end")):
            candidate_default = f"{int(current_values['year_start'])}-{int(current_values['year_end'])}"
            if candidate_default in year_options:
                default_year = candidate_default
        if "year_range_select" in st.session_state and st.session_state["year_range_select"] not in year_options:
            st.session_state["year_range_select"] = "Any"
        default_year = keep_valid_choice(default_year, year_options)
        selected_year = choose_option("Compatible years", year_options, default_year, key="year_range_select")
        if selected_year and selected_year != "Any":
            year_start, year_end = selected_year.split("-")
            current_values["year_start"] = int(year_start)
            current_values["year_end"] = int(year_end)
        else:
            current_values["year_start"] = None
            current_values["year_end"] = None

    st.subheader("Part")
    part_scope = filter_reference_rows(
        reference_rows,
        {
            "brand": current_values.get("brand"),
            "model": current_values.get("model"),
            "year_start": current_values.get("year_start"),
            "year_end": current_values.get("year_end"),
        },
    )
    part_col1, part_col2 = st.columns(2)

    with part_col1:
        category_options = sorted_unique_options(part_scope["category"])
        sync_dependent_state(current_values, "category", category_options, "category_select")
        current_values["category"] = choose_option(
            "Part group",
            category_options,
            current_values.get("category"),
            key="category_select",
            label_map=build_option_label_map(category_options),
        )

    scoped_rows = filter_reference_rows(part_scope, {"category": current_values.get("category")})
    part_catalog = build_part_option_catalog(scoped_rows)
    current_part_type = derive_part_type(current_values.get("part_name"), current_values.get("subcategory"))
    part_type_options = sorted_unique_options(part_catalog["display_part_type"])

    with part_col2:
        selected_part_type = choose_option(
            "Part type",
            part_type_options,
            current_part_type,
            key="part_type_select",
        )

    filtered_catalog = part_catalog.loc[part_catalog["display_part_type"] == selected_part_type]

    side_options = sorted_unique_options(filtered_catalog["display_side"])
    position_options = sorted_unique_options(filtered_catalog["display_position"])
    show_side_selector = len(side_options) > 1
    show_position_selector = len(position_options) > 1

    if show_side_selector or show_position_selector:
        extra_cols = st.columns(2)
        if show_side_selector:
            with extra_cols[0]:
                selected_side = choose_option(
                    "Side",
                    side_options,
                    side_options[0],
                    key="subcategory_side_select",
                )
                filtered_catalog = filtered_catalog.loc[filtered_catalog["display_side"] == selected_side]
        if show_position_selector:
            with extra_cols[1]:
                refreshed_position_options = sorted_unique_options(filtered_catalog["display_position"])
                selected_position = choose_option(
                    "Position",
                    refreshed_position_options,
                    refreshed_position_options[0],
                    key="subcategory_position_select",
                )
                filtered_catalog = filtered_catalog.loc[
                    filtered_catalog["display_position"] == selected_position
                ]

    variant_options = filtered_catalog["raw_part_name"].tolist()
    current_values["part_name"] = keep_valid_choice(current_values.get("part_name"), variant_options)
    if current_values["part_name"] is None and variant_options:
        current_values["part_name"] = variant_options[0]

    remaining_catalog = filtered_catalog
    if current_values.get("part_name") not in {None, ""}:
        remaining_catalog = remaining_catalog.loc[
            remaining_catalog["raw_part_name"].astype(str) == str(current_values["part_name"])
        ]

    subcategory_options = remaining_catalog["raw_subcategory"].tolist()
    current_values["subcategory"] = keep_valid_choice(current_values.get("subcategory"), subcategory_options)
    if current_values["subcategory"] is None and subcategory_options:
        current_values["subcategory"] = subcategory_options[0]

    st.subheader("Condition And Details")
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        quality_options = QUALITY_GRADE_OPTIONS
        sync_dependent_state(current_values, "quality_grade", quality_options, "quality_grade_select")
        current_values["quality_grade"] = choose_option(
            "Quality grade",
            quality_options,
            current_values.get("quality_grade"),
            key="quality_grade_select",
        )

    repair_options = sorted_unique_options(reference_rows["repair_status"])
    current_values["repair_status"] = repair_options[0] if repair_options else "original_valid"

    mileage_series = pd.to_numeric(reference_rows["mileage"], errors="coerce")
    default_mileage = current_values.get("mileage")
    default_mileage = 0.0 if pd.isna(default_mileage) else float(default_mileage)
    with detail_col2:
        current_values["mileage"] = st.number_input(
            "Vehicle mileage",
            min_value=0.0,
            max_value=float(mileage_series.max()) if mileage_series.notna().any() else 500000.0,
            value=default_mileage,
            step=1000.0,
            key="mileage_input",
        )

    current_values["oem_number"] = st.text_input(
        "OEM number (optional)",
        value="" if pd.isna(current_values.get("oem_number")) else str(current_values.get("oem_number")),
        key="oem_number_input",
    )

    return current_values


def render_prediction_results(bundle, reference_rows, feature_names, submitted_values):
    """Render the prediction summary and local explanation."""

    full_input = build_full_input(reference_rows, submitted_values)
    prediction_input = {feature_name: full_input.get(feature_name) for feature_name in feature_names}

    with st.spinner("Estimating price and building explanation..."):
        prediction = predict_price_ranges(bundle, [prediction_input]).iloc[0]
        market_range = comparable_market_range(
            reference_rows=reference_rows,
            visible_values=submitted_values,
            predicted_price=float(prediction["predicted_price"]),
        )
        shap_explanation = compute_local_shap_explanation(
            bundle=bundle,
            reference_rows=reference_rows,
            prediction_input=prediction_input,
        )

    support_text = (
        "Strong" if market_range["comparable_count"] >= 20 else
        "Moderate" if market_range["comparable_count"] >= 8 else
        "Limited"
    )

    local_shap = shap_explanation["local_df"].copy()
    top_drivers = local_shap.head(6).copy()
    shown_effect = float(top_drivers["shap_value"].sum()) if not top_drivers.empty else 0.0
    remaining_effect = float(shap_explanation["total_effect"] - shown_effect)

    output_col1, output_col2 = st.columns([1.0, 1.0], gap="large")

    with output_col1:
        st.subheader("Prediction")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Estimated price", f"{prediction['predicted_price']:.0f} EUR")
        with metric_col2:
            st.metric(
                "Expected market range",
                f"{market_range['range_low']:.0f}-{market_range['range_high']:.0f} EUR",
            )
        with metric_col3:
            st.metric("Market support", f"{support_text} ({int(market_range['comparable_count'])})")
            st.caption(f"{int(market_range['comparable_count'])} comparable listings found in reference data.")

        st.write(
            f"Estimated price for this part is around {prediction['predicted_price']:.0f} EUR. "
            f"Comparable market references suggest a range of "
            f"{market_range['range_low']:.0f}-{market_range['range_high']:.0f} EUR."
        )
        st.info(
            "Expected market range is derived from matching historical market examples in the saved reference data. "
            "It is a market reference signal, not pure model uncertainty."
        )

    with output_col2:
        st.subheader("Why this price?")
        if top_drivers.empty:
            st.info("Local SHAP explanation returned no active drivers for this prediction.")
        else:
            top_drivers["feature_value"] = [
                format_feature_value(feature_name, value)
                for feature_name, value in zip(top_drivers["feature_name"], top_drivers["feature_value"])
            ]
            st.caption("Main factors that changed this estimate for the current part selection.")
            for row in top_drivers.itertuples(index=False):
                rounded_effect = int(round(abs(float(row.shap_value))))
                if rounded_effect == 0:
                    continue
                direction_text = "raised" if float(row.shap_value) >= 0 else "lowered"
                st.write(
                    f"- **{row.display_name}: {row.feature_value}** "
                    f"{direction_text} the estimate by **{rounded_effect} EUR**"
                )

            rounded_remaining = int(round(abs(remaining_effect)))
            if rounded_remaining > 0:
                remaining_direction = "raised" if remaining_effect >= 0 else "lowered"
                st.caption(
                    f"Other smaller factors combined {remaining_direction} the estimate by {rounded_remaining} EUR."
                )

    st.caption(
        "Decision-support prototype for dismantling-part pricing. Final pricing decisions should still include business judgment."
    )

    with st.expander("Technical Details"):
        st.write("Hidden model fields are filled from the closest saved reference row and simple derived values.")
        st.metric("Market reference source", str(market_range["range_source"]))
        st.metric("Market reference count", int(market_range["comparable_count"]))
        st.caption(f"Market reference match keys: {market_range['matched_on'] or 'fallback'}")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.metric("Ensemble low", f"{prediction['ensemble_range_low']:.2f} EUR")
            st.metric("Ensemble high", f"{prediction['ensemble_range_high']:.2f} EUR")
        with tech_col2:
            st.metric("Calibrated low", f"{prediction['price_range_low']:.2f} EUR")
            st.metric("Calibrated high", f"{prediction['price_range_high']:.2f} EUR")
            st.metric("Ensemble width", f"{prediction['ensemble_range_width']:.2f} EUR")
            st.metric("Model range source", str(prediction["uncertainty_source"]))
        shap_col1, shap_col2, shap_col3 = st.columns(3)
        with shap_col1:
            st.metric("SHAP baseline", f"{shap_explanation['base_value']:.2f} EUR")
        with shap_col2:
            st.metric("SHAP net feature effect", f"{shap_explanation['total_effect']:+.2f} EUR")
        with shap_col3:
            st.metric("SHAP reconstruction error", f"{shap_explanation['reconstruction_error']:.4f} EUR")
        st.dataframe(pd.DataFrame({"operator_input": submitted_values}).T, use_container_width=True)


def main():
    """Render the local Streamlit UI."""

    st.set_page_config(page_title="DPPM Price Estimator", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            min-height: 120px;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            line-height: 1.15;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    bundle = load_bundle()
    metadata = bundle["metadata"]
    reference_rows = pd.read_csv(BUNDLE_DIR / "reference_rows.csv")
    feature_names = list(metadata["feature_names"])
    initialize_form_state(reference_rows)

    st.title("DPPM Price Estimator")
    st.caption("Operator-facing UI for the final random-forest dismantling-part price model.")

    with st.expander("Load Example", expanded=False):
        example_rows = (
            reference_rows[OPERATOR_FIELDS]
            .drop_duplicates(subset=["brand", "model", "category", "subcategory", "part_name"])
            .reset_index(drop=True)
        )
        example_labels = example_rows.apply(build_example_label, axis=1).tolist()
        selected_label = st.selectbox("Reference row", options=example_labels)
        selected_index = example_labels.index(selected_label)
        if st.button("Use example", type="primary"):
            selected_row = example_rows.iloc[selected_index]
            st.session_state["input_values"] = {
                feature_name: selected_row.get(feature_name) for feature_name in OPERATOR_FIELDS
            }
            st.rerun()

    updated_values = render_operator_form(reference_rows)
    st.session_state["input_values"] = updated_values

    if st.button("Estimate Price", type="primary"):
        st.session_state["submitted_input_values"] = dict(updated_values)

    submitted_values = st.session_state.get("submitted_input_values")
    if submitted_values:
        render_prediction_results(bundle, reference_rows, feature_names, submitted_values)


if __name__ == "__main__":
    main()
