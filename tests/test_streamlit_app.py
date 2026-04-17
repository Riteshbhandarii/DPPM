import pandas as pd

import app.streamlit_app as streamlit_app


def build_reference_rows():
    return pd.DataFrame(
        [
            {
                "part_name": "Brake Caliper",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "A2",
                "repair_status": "original_valid",
                "mileage": 100000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-1",
                "price": 200.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 10,
            },
            {
                "part_name": "Brake Caliper",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "A1",
                "repair_status": "original_valid",
                "mileage": 120000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-2",
                "price": 220.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 11,
            },
            {
                "part_name": "Brake Disc",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "front",
                "quality_grade": "B1",
                "repair_status": "original_valid",
                "mileage": 80000.0,
                "year_start": 2012,
                "year_end": 2018,
                "oem_number": "OEM-3",
                "price": 150.0,
                "year_span": 6,
                "year_mid": 2015.0,
                "mileage_missing_flag": False,
                "hidden_metric": 12,
            },
            {
                "part_name": "Contact Roll Airbag",
                "brand": "toyota",
                "model": "corolla",
                "category": "airbag",
                "subcategory": "contact roll airbag",
                "quality_grade": "A2",
                "repair_status": "original_valid",
                "mileage": 90000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-4",
                "price": 180.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 20,
            },
            {
                "part_name": "Brake Caliper",
                "brand": "vw",
                "model": "golf",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "A2",
                "repair_status": "original_valid",
                "mileage": 110000.0,
                "year_start": 2011,
                "year_end": 2016,
                "oem_number": "OEM-5",
                "price": 210.0,
                "year_span": 5,
                "year_mid": 2013.5,
                "mileage_missing_flag": False,
                "hidden_metric": 30,
            },
            {
                "part_name": "Brake Caliper",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "A3",
                "repair_status": "original_valid",
                "mileage": 130000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-6",
                "price": 210.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 13,
            },
            {
                "part_name": "Brake Caliper",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "B1",
                "repair_status": "original_valid",
                "mileage": 140000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-7",
                "price": 205.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 14,
            },
            {
                "part_name": "Brake Caliper",
                "brand": "toyota",
                "model": "corolla",
                "category": "brakes",
                "subcategory": "rear",
                "quality_grade": "B2",
                "repair_status": "original_valid",
                "mileage": 150000.0,
                "year_start": 2010,
                "year_end": 2015,
                "oem_number": "OEM-8",
                "price": 230.0,
                "year_span": 5,
                "year_mid": 2012.5,
                "mileage_missing_flag": False,
                "hidden_metric": 15,
            },
        ]
    )


def test_filter_reference_rows_ignores_any_and_none():
    reference_rows = build_reference_rows()

    filtered = streamlit_app.filter_reference_rows(
        reference_rows,
        {"brand": "toyota", "model": "Any", "year_start": None},
    )

    assert set(filtered["brand"]) == {"toyota"}
    assert len(filtered) == 7


def test_keep_valid_choice_falls_back_to_first_option():
    assert streamlit_app.keep_valid_choice("contact roll airbag", ["front", "rear"]) == "front"
    assert streamlit_app.keep_valid_choice("rear", ["front", "rear"]) == "rear"


def test_sync_dependent_state_resets_invalid_session_value():
    current_values = {"subcategory": "contact roll airbag"}
    streamlit_app.st.session_state.clear()
    streamlit_app.st.session_state["subcategory_select"] = "contact roll airbag"

    value = streamlit_app.sync_dependent_state(
        current_values=current_values,
        field_name="subcategory",
        options=["front", "rear"],
        session_key="subcategory_select",
    )

    assert value == "front"
    assert current_values["subcategory"] == "front"
    assert streamlit_app.st.session_state["subcategory_select"] == "front"


def test_build_full_input_prefers_exact_visible_match_and_derives_fields():
    reference_rows = build_reference_rows()
    visible_values = {
        "part_name": "Brake Caliper",
        "brand": "toyota",
        "model": "corolla",
        "category": "brakes",
        "subcategory": "rear",
        "quality_grade": "A2",
        "repair_status": "original_valid",
        "mileage": 123456.0,
        "year_start": 2010,
        "year_end": 2015,
        "oem_number": "NEW-OEM",
    }

    built = streamlit_app.build_full_input(reference_rows, visible_values)

    assert built["hidden_metric"] == 10
    assert built["oem_number"] == "NEW-OEM"
    assert built["year_span"] == 5
    assert built["year_mid"] == 2012.5
    assert built["mileage_missing_flag"] is False


def test_comparable_market_range_uses_matching_rows():
    reference_rows = build_reference_rows()
    visible_values = {
        "part_name": "Brake Caliper",
        "brand": "toyota",
        "model": "corolla",
        "category": "brakes",
        "subcategory": "rear",
        "quality_grade": "A2",
        "repair_status": "original_valid",
        "mileage": 125000.0,
    }

    market = streamlit_app.comparable_market_range(
        reference_rows=reference_rows,
        visible_values=visible_values,
        predicted_price=215.0,
    )

    assert market["range_source"] == "comparable_rows"
    assert market["comparable_count"] == 5
    assert "part_name" in market["matched_on"]
    assert market["range_low"] <= 215.0 <= market["range_high"]
    assert market["range_width"] >= 30.0


def test_comparable_market_range_falls_back_when_not_enough_matches():
    reference_rows = build_reference_rows().head(4)
    visible_values = {
        "part_name": "Nonexistent Part",
        "brand": "toyota",
        "model": "corolla",
        "category": "brakes",
        "subcategory": "rear",
        "quality_grade": "A2",
        "repair_status": "original_valid",
        "mileage": 125000.0,
    }

    market = streamlit_app.comparable_market_range(
        reference_rows=reference_rows,
        visible_values=visible_values,
        predicted_price=215.0,
    )

    assert market["range_source"] == "fallback_minimum_band"
    assert market["comparable_count"] == 0
    assert market["range_low"] == 140.0
    assert market["range_high"] == 290.0
