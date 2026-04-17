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
QUALITY_GRADE_OPTIONS = ["A1", "A2", "A3", "B1", "B2", "B3", "C", "C1", "C2"]


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


def sorted_unique_options(series):
    """Return stable alphabetical options without duplicates or NaNs."""

    return sorted({str(value) for value in series.dropna().astype(str) if str(value).strip()})


def filter_reference_rows(reference_rows, filters):
    """Filter reference rows by the currently selected visible inputs."""

    filtered = reference_rows
    for key, value in filters.items():
        if value in {None, "", "Any"} or pd.isna(value):
            continue
        filtered = filtered.loc[filtered[key].astype(str) == str(value)]
    return filtered


def choose_option(label, options, current_value, key):
    """Render a selectbox with safe fallback handling."""

    if not options:
        return None
    normalized_current = None if pd.isna(current_value) else str(current_value)
    index = options.index(normalized_current) if normalized_current in options else 0
    return st.selectbox(label, options=options, index=index, key=key)


def keep_valid_choice(current_value, options):
    """Drop stale selections when parent filters change."""

    normalized_current = None if pd.isna(current_value) else str(current_value)
    if normalized_current in options:
        return normalized_current
    return options[0] if options else None


def sync_dependent_state(current_values, field_name, options, session_key):
    """Keep session state and current values aligned with the valid option list."""

    valid_value = keep_valid_choice(current_values.get(field_name), options)
    current_values[field_name] = valid_value
    if session_key in st.session_state:
        session_value = st.session_state.get(session_key)
        if session_value not in options:
            st.session_state[session_key] = valid_value
    return valid_value


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
    st.session_state.setdefault("submitted_input_values", dict(st.session_state["input_values"]))


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
        )

    year_rows = filter_reference_rows(
        reference_rows,
        {
            "brand": current_values.get("brand"),
            "model": current_values.get("model"),
        },
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
        selected_year = choose_option(
            "Compatible years",
            year_options,
            default_year,
            key="year_range_select",
        )
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
    part_col1, part_col2, part_col3 = st.columns(3)

    with part_col1:
        category_options = sorted_unique_options(part_scope["category"])
        sync_dependent_state(current_values, "category", category_options, "category_select")
        current_values["category"] = choose_option(
            "Part group",
            category_options,
            current_values.get("category"),
            key="category_select",
        )

    subcategory_scope = filter_reference_rows(
        part_scope,
        {"category": current_values.get("category")},
    )
    with part_col2:
        subcategory_options = sorted_unique_options(subcategory_scope["subcategory"])
        sync_dependent_state(current_values, "subcategory", subcategory_options, "subcategory_select")
        current_values["subcategory"] = choose_option(
            "Part area",
            subcategory_options,
            current_values.get("subcategory"),
            key="subcategory_select",
        )

    part_name_scope = filter_reference_rows(
        subcategory_scope,
        {"subcategory": current_values.get("subcategory")},
    )
    with part_col3:
        part_name_options = sorted_unique_options(part_name_scope["part_name"])
        sync_dependent_state(current_values, "part_name", part_name_options, "part_name_select")
        current_values["part_name"] = choose_option(
            "Part name",
            part_name_options,
            current_values.get("part_name"),
            key="part_name_select",
        )

    st.subheader("Condition And Details")
    detail_scope = filter_reference_rows(
        part_name_scope,
        {"part_name": current_values.get("part_name")},
    )
    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        quality_options = QUALITY_GRADE_OPTIONS
        sync_dependent_state(current_values, "quality_grade", quality_options, "quality_grade_select")
        current_values["quality_grade"] = choose_option(
            "Quality grade",
            quality_options,
            current_values.get("quality_grade"),
            key="quality_grade_select",
        )

    with detail_col2:
        repair_options = sorted_unique_options(reference_rows["repair_status"])
        if len(repair_options) <= 1:
            current_values["repair_status"] = repair_options[0] if repair_options else "original_valid"
            st.text_input(
                "Repair status",
                value=current_values["repair_status"],
                disabled=True,
                key="repair_status_display",
            )
        else:
            current_values["repair_status"] = choose_option(
                "Repair status",
                repair_options,
                current_values.get("repair_status"),
                key="repair_status_select",
            )

    mileage_series = pd.to_numeric(reference_rows["mileage"], errors="coerce")
    default_mileage = current_values.get("mileage")
    default_mileage = 0.0 if pd.isna(default_mileage) else float(default_mileage)
    with detail_col3:
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

    with st.form("prediction_form"):
        updated_values = render_operator_form(reference_rows)
        submitted = st.form_submit_button("Estimate Price", type="primary")

    st.session_state["input_values"] = updated_values
    if submitted:
        st.session_state["submitted_input_values"] = dict(updated_values)

    submitted_values = st.session_state.get("submitted_input_values")
    if submitted_values:
        full_input = build_full_input(reference_rows, submitted_values)
        prediction_input = {feature_name: full_input.get(feature_name) for feature_name in feature_names}

        prediction = predict_price_ranges(bundle, [prediction_input]).iloc[0]
        market_range = comparable_market_range(
            reference_rows=reference_rows,
            visible_values=submitted_values,
            predicted_price=float(prediction["predicted_price"]),
        )

        st.subheader("Prediction")
        market_col1, market_col2 = st.columns(2)
        with market_col1:
            st.metric("Point estimate", f"{prediction['predicted_price']:.2f} EUR")
            st.metric("Comparable low", f"{market_range['range_low']:.2f} EUR")
            st.metric("Comparable rows", int(market_range["comparable_count"]))
        with market_col2:
            st.metric("Comparable high", f"{market_range['range_high']:.2f} EUR")
            st.metric("Comparable width", f"{market_range['range_width']:.2f} EUR")

        st.info(
            "This block is a separate market lookup based on similar historical rows from the saved reference data. It is not the same thing as model uncertainty."
        )

        with st.expander("Technical Details"):
            st.write("Hidden model fields are filled from the closest saved reference row and simple derived values.")
            st.metric("Comparable source", str(market_range["range_source"]))
            st.metric("Comparable rows used", int(market_range["comparable_count"]))
            st.caption(f"Comparable match keys: {market_range['matched_on'] or 'fallback'}")
            tech_col1, tech_col2 = st.columns(2)
            with tech_col1:
                st.metric("Ensemble low", f"{prediction['ensemble_range_low']:.2f} EUR")
                st.metric("Ensemble high", f"{prediction['ensemble_range_high']:.2f} EUR")
            with tech_col2:
                st.metric("Calibrated low", f"{prediction['price_range_low']:.2f} EUR")
                st.metric("Calibrated high", f"{prediction['price_range_high']:.2f} EUR")
                st.metric("Ensemble width", f"{prediction['ensemble_range_width']:.2f} EUR")
                st.metric("Model range source", str(prediction["uncertainty_source"]))
            st.dataframe(
                pd.DataFrame({"operator_input": submitted_values}).T,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
