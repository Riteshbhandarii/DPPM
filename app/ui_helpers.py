"""Shared UI/domain helpers for the Streamlit pricing workflow."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

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
LOCATION_PATTERNS = {
    "right rear": ("Right", "Rear"),
    "left rear": ("Left", "Rear"),
    "right front": ("Right", "Front"),
    "left front": ("Left", "Front"),
    "either side front": ("Either side", "Front"),
    "either side rear": ("Either side", "Rear"),
    "either side": ("Either side", None),
    "right": ("Right", None),
    "left": ("Left", None),
    "rear": (None, "Rear"),
    "front": (None, "Front"),
    "centre": (None, "Centre"),
}
GENERIC_LOCATION_LABELS = {
    "all",
    "general",
    "any",
    "either side",
    "left",
    "right",
    "front",
    "rear",
    "centre",
}
FEATURE_LABELS = {
    "subcategory": "Part type / placement",
    "part_name": "Specific part label",
    "category": "Part group",
    "year_start": "Compatible year start",
    "year_end": "Compatible year end",
    "year_mid": "Vehicle generation midpoint",
    "year_span": "Compatible year span",
    "oem_number": "OEM number",
    "mileage": "Mileage",
    "model": "Vehicle model",
    "brand": "Vehicle brand",
    "quality_grade": "Quality grade",
}


def clean_display_label(value):
    """Convert noisy source labels into cleaner UI text without changing raw values."""

    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s*-\s*,\s*e-\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-\s*e-\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")

    replacements = {
        "abs": "ABS",
        "ac": "AC",
        "vw": "VW",
        "airbag control unit": "Airbag control unit",
        "airbag krocksensor": "Airbag impact sensor",
        "contact roll airbag": "Airbag contact roll",
        "fuse box / electricity central": "Fuse box / electrical central",
        "gear box / drive axle / middle axle": "Drivetrain",
        "parkeringshjälp frontsensor": "Parking assist front sensor",
        "kamera utvändig": "Exterior camera",
        "dörr styrenhet": "Door control unit",
        "ljusinställning lägesgivare": "Headlight level sensor",
        "komfort styrdon": "Comfort control unit",
    }

    lowered = text.lower()
    if lowered in replacements:
        return replacements[lowered]

    if lowered in LOCATION_PATTERNS:
        side, position = LOCATION_PATTERNS[lowered]
        return " ".join(part for part in [side, position] if part) or "General"

    return text[:1].upper() + text[1:]


def normalize_label_key(value):
    """Normalize cleaned display labels for redundancy checks."""

    if pd.isna(value):
        return ""
    text = clean_display_label(value)
    text = re.sub(r"\s*\[[^\]]+\]\s*$", "", text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def feature_display_name(feature_name):
    """Convert raw feature names into frontend-friendly labels."""

    if pd.isna(feature_name):
        return ""
    feature_name = str(feature_name)
    return FEATURE_LABELS.get(feature_name, clean_display_label(feature_name.replace("_", " ")))


def format_feature_value(feature_name, value):
    """Format feature values for user-facing SHAP explanations."""

    if pd.isna(value):
        return "Missing"
    if feature_name in {"year_start", "year_end"}:
        return str(int(float(value)))
    if feature_name == "year_mid":
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:.1f}"
    if feature_name == "mileage":
        return f"{int(float(value)):,} km".replace(",", " ")
    return clean_display_label(value)


def build_option_label_map(options):
    """Create unique display labels for raw options after cleanup."""

    cleaned_values = {option: clean_display_label(option) for option in options}
    counts = pd.Series(list(cleaned_values.values())).value_counts().to_dict()
    label_map = {}

    for option in options:
        cleaned = cleaned_values[option]
        if counts.get(cleaned, 0) <= 1:
            label_map[option] = cleaned
            continue
        raw_text = re.sub(r"\s+", " ", str(option).strip())
        label_map[option] = f"{cleaned} [{raw_text}]"

    return label_map


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


def choose_option(label, options, current_value, key, label_map=None):
    """Render a selectbox with safe fallback handling."""

    if not options:
        return None
    normalized_current = None if pd.isna(current_value) else str(current_value)
    index = options.index(normalized_current) if normalized_current in options else 0
    if label_map is not None:
        format_func = lambda value: label_map.get(value, value)
        return st.selectbox(label, options=options, index=index, key=key, format_func=format_func)
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


def extract_side_position(*texts):
    """Extract side and position hints from one or more raw labels."""

    side = None
    position = None
    candidate_texts = []
    for text in texts:
        if pd.isna(text):
            continue
        raw = str(text).strip()
        candidate_texts.append(raw.lower())
        candidate_texts.extend(match.lower() for match in re.findall(r"\(([^)]*)\)", raw))

    for candidate in candidate_texts:
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if normalized in LOCATION_PATTERNS:
            parsed_side, parsed_position = LOCATION_PATTERNS[normalized]
            side = side or parsed_side
            position = position or parsed_position

    return side, position


def derive_part_type(part_name, subcategory):
    """Derive a cleaner user-facing part type from raw part fields."""

    cleaned_part_name = clean_display_label(re.sub(r"\([^)]*\)", "", str(part_name))).strip()
    cleaned_subcategory = clean_display_label(subcategory).strip()

    part_core = normalize_label_key(cleaned_part_name)
    sub_core = normalize_label_key(cleaned_subcategory)

    if part_core and sub_core and part_core != sub_core and sub_core not in GENERIC_LOCATION_LABELS:
        return cleaned_part_name
    if part_core and part_core not in GENERIC_LOCATION_LABELS:
        return cleaned_part_name
    if sub_core and sub_core not in GENERIC_LOCATION_LABELS:
        return cleaned_subcategory
    return cleaned_part_name or cleaned_subcategory or "Unknown part"


def build_part_option_catalog(part_scope):
    """Build a UI-facing part catalog from raw subcategory and part-name fields."""

    rows = []
    unique_rows = part_scope[["subcategory", "part_name"]].drop_duplicates()
    for row in unique_rows.itertuples(index=False):
        side, position = extract_side_position(row.part_name, row.subcategory)
        rows.append(
            {
                "raw_subcategory": str(row.subcategory),
                "raw_part_name": str(row.part_name),
                "display_part_type": derive_part_type(row.part_name, row.subcategory),
                "display_side": side,
                "display_position": position,
            }
        )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


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

    values["mileage_missing_flag"] = bool(pd.isna(values.get("mileage")))
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
