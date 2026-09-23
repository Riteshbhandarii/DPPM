"""
Purpose:
Run the frozen Random Forest on the September 2026 listings chosen by hand and
write its prediction next to each asking price.

Inputs:
- results/september_live_validation/september_listings.csv   the chosen listings
- artifacts/random_forest_final/full_data_bundle              the frozen model (not in git)
- datasets/cleaned/clean_master_dataset.csv                   February rows, for part names and registry values
- datasets/traficom_outputs/{model,brand}_summary.csv         registry values for vehicles February never saw

Output:
- the same CSV, with `predicted_eur` filled in

Assumptions:
- The model is not refitted. Listing quality grade is left out, as in every run of the study.
- Registry columns are constant per vehicle, so they are copied from February, or
  from the Traficom summaries for a vehicle February never saw.
- February's own part-name spelling is used where it exists; a different string
  would fall into the encoder's infrequent bucket without any error.

How to run:
    .venv/bin/python scripts/predict_september_listings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.random_forest_serving import ensure_feature_frame, load_random_forest_bundle  # noqa: E402

LISTINGS = ROOT / "results/september_live_validation/september_listings.csv"
BUNDLE = ROOT / "artifacts/random_forest_final/full_data_bundle"
FEBRUARY = ROOT / "datasets/cleaned/clean_master_dataset.csv"
TRAFICOM = ROOT / "datasets/traficom_outputs"


def registry_values(february, brand, model, columns):
    car = february[(february.brand == brand) & (february.model == model)]
    if not car.empty:
        return car.iloc[0][columns]
    models = pd.read_csv(TRAFICOM / "model_summary.csv")
    brands = pd.read_csv(TRAFICOM / "brand_summary.csv")
    row = pd.concat([
        models[(models.brand == brand) & (models.model_family_clean == model)].iloc[0],
        brands[brands.brand == brand].iloc[0],
    ])
    return row[columns]


def features_for(listings, february, brand, model, feature_names):
    car = february[(february.brand == brand) & (february.model == model)]
    part_names = car.groupby("subcategory").part_name.agg(lambda names: names.mode().iat[0])
    frame = pd.DataFrame({
        "part_name": listings.subcategory.map(part_names).fillna(listings.part_label.str.strip() + " -"),
        "quality_grade": np.nan,
        "mileage": listings.mileage,
        "brand": brand,
        "model": model,
        "category": listings.category,
        "subcategory": listings.subcategory,
        "year_start": listings.year_start,
        "year_end": listings.year_end,
        "year_span": listings.year_end - listings.year_start,
        "year_mid": (listings.year_end + listings.year_start) / 2,
        "repair_status": "original_valid",
    })
    registry = [name for name in feature_names if name.startswith(("model_", "brand_"))]
    for column, value in registry_values(february, brand, model, registry).items():
        frame[column] = value
    return ensure_feature_frame(frame, feature_names)


def main():
    listings = pd.read_csv(LISTINGS)
    february = pd.read_csv(FEBRUARY, low_memory=False)
    bundle = load_random_forest_bundle(BUNDLE)
    feature_names = bundle["metadata"]["feature_names"]

    for (brand, model), rows in listings.groupby(["brand", "model"]):
        features = features_for(rows, february, brand, model, feature_names)
        listings.loc[rows.index, "predicted_eur"] = bundle["model"].predict(features).round(2)

    listings.to_csv(LISTINGS, index=False)
    error = (listings.asking_price_eur - listings.predicted_eur).abs() / listings.asking_price_eur * 100
    print(listings.assign(error_pct=error).groupby(["round", "tier"]).error_pct.median().round(1).to_string())


if __name__ == "__main__":
    main()
