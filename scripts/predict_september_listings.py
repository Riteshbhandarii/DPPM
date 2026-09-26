"""
Purpose:
Run the frozen Random Forest on the September 2026 listings chosen for
validation, and put its prediction and the per-part median baseline next to
each asking price.

Inputs:
- results/september_live_validation/september_listings.csv   the chosen listings, as collected
- artifacts/random_forest_final/full_data_bundle              the frozen model (not in git)
- datasets/cleaned/clean_master_dataset.csv                   February rows: part names, registry values
- datasets/splits_strict/{train,validation}_strict.csv        February rows the baseline is fitted on
- datasets/traficom_outputs/{model,brand}_summary.csv         registry values for vehicles February never saw

Output:
- results/september_live_validation/september_predictions.csv
  the listings plus `predicted_eur` and `baseline_eur`
- the thesis results table (median absolute error and predictions within 25%
  of the asking price, per round and listing), printed

Assumptions:
- The model is not refitted. Quality grade is the one read off each listing card, as the
  model was trained with it; a missing grade would be imputed as the training mode (A2).
- Baseline: the subcategory-median heuristic of the thesis, the median price of
  the part over the strict training and validation splits, all three vehicles
  together. It is the same baseline the frozen model was compared with on the
  held-out test set, so the two evaluations use one definition.
- Registry columns are constant per vehicle, so they are copied from February, or
  from the Traficom summaries for a vehicle February never saw.
- February's own part-name spelling is used where it exists; a different string
  would fall into the encoder's infrequent bucket without any error.

How to run:
    .venv/bin/python scripts/predict_september_listings.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.random_forest_serving import ensure_feature_frame, load_random_forest_bundle  # noqa: E402

LISTINGS = ROOT / "results/september_live_validation/september_listings.csv"
PREDICTIONS = ROOT / "results/september_live_validation/september_predictions.csv"
BUNDLE = ROOT / "artifacts/random_forest_final/full_data_bundle"
FEBRUARY = ROOT / "datasets/cleaned/clean_master_dataset.csv"
SPLITS = ROOT / "datasets/splits_strict"
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
        "quality_grade": listings.quality_grade,
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

    fitted = pd.concat([pd.read_csv(SPLITS / f"{name}_strict.csv", low_memory=False)
                        for name in ("train", "validation")])
    medians = fitted.groupby("subcategory").price.median()
    listings["baseline_eur"] = listings.subcategory.map(medians).fillna(fitted.price.median()).round(2)

    listings.to_csv(PREDICTIONS, index=False)
    print(results_table(listings).to_string(index=False))


def results_table(listings):
    """Median absolute error and predictions within 25% of the asking price."""
    price = listings.asking_price_eur
    scored = listings.assign(
        rf_error=(listings.predicted_eur - price).abs(),
        baseline_error=(listings.baseline_eur - price).abs(),
    )
    scored["rf_within_25"] = scored.rf_error <= 0.25 * price
    scored["baseline_within_25"] = scored.baseline_error <= 0.25 * price

    def summary(rows):
        return {
            "n": len(rows),
            "median_ae_rf": round(rows.rf_error.median(), 2),
            "median_ae_baseline": round(rows.baseline_error.median(), 2),
            "within_25_rf": int(rows.rf_within_25.sum()),
            "within_25_baseline": int(rows.baseline_within_25.sum()),
        }

    tiers = ["dearest", "middle", "cheapest"]
    rows = []
    for round_number, group in scored.groupby("round"):
        for tier in tiers:
            rows.append({"round": round_number, "listing": tier, **summary(group[group.tier == tier])})
        rows.append({"round": round_number, "listing": "all", **summary(group)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
