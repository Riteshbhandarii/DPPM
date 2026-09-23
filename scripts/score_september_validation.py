"""Score hand-collected September listings against the frozen random forest.

Pre-registered design lives in the thesis notes; the rules this script encodes:

* nothing is refit, retuned or reselected -- the full-data bundle is loaded as is
* listings whose product_id already sits in the February training data are dropped
* the PRIMARY analysis is the matched draw: per part x generation, the top five
  by price, which reproduces February's page-1 draw on a price-descending page
* the census (every unseen listing) is reported second, as a market-coverage
  figure rather than as the model's score
* headline metrics are median-based; means are fragile at this sample size
* the subcategory-median heuristic is fitted on the same scope as the bundle
* a vehicle absent from February (Round 2) takes its registry block from the
  Traficom summaries and is compared against the all-brand subcategory median,
  the only baseline that exists for it

usage: python scripts/score_september_validation.py <listings.csv> <brand> <model> [generation regex]

The optional regex keeps listings whose generation label matches it. The site
pools some families (Ford Focus with C-Max, Volvo S70 with V70), and the pooled
page is saved as is and filtered here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.random_forest_serving import ensure_feature_frame, load_random_forest_bundle

BUNDLE = ROOT / "artifacts/random_forest_final/full_data_bundle"
FEBRUARY = ROOT / "datasets/cleaned/clean_master_dataset.csv"
TRAFICOM_MODELS = ROOT / "datasets/traficom_outputs/model_summary.csv"
TRAFICOM_BRANDS = ROOT / "datasets/traficom_outputs/brand_summary.csv"
TOP_N = 5
BANDS = [0, 50, 100, 200, 500, 1000, np.inf]
BAND_LABELS = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]

REGISTRY_PREFIXES = ("model_", "brand_")

# The 2026-09-18 Corolla run scored without a quality grade, and every later car
# has to be scored the same way or the cars are not comparable to each other.
# The grade is still parsed into the listing table; it carries 0.2% of the
# model's SHAP importance and moves the headline by 0.2 points either way.
SCORE_WITH_QUALITY_GRADE = False


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    absolute = np.abs(error)
    return {
        "n": len(actual),
        "MAE": absolute.mean(),
        "MdAE": np.median(absolute),
        "RMSE": np.sqrt((error**2).mean()),
        "R2": 1 - (error**2).sum() / ((actual - actual.mean()) ** 2).sum(),
        "MdAPE_pct": np.median(absolute / actual) * 100,
        "median_bias": np.median(predicted - actual),
    }


def show(title, rows):
    print(f"\n{title}")
    frame = pd.DataFrame(rows).set_index("sample")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
        print(frame.to_string())


def traficom_constants(brand, model, registry_columns):
    """The registry block for a vehicle February never saw.

    Brand is matched exactly: ford-cng-technik and ford-tec also carry a
    `focus` family, with 41 and 1 vehicles.
    """
    models = pd.read_csv(TRAFICOM_MODELS)
    brands = pd.read_csv(TRAFICOM_BRANDS)
    model_row = models[(models.brand == brand) & (models.model_family_clean == model)]
    brand_row = brands[brands.brand == brand]
    if len(model_row) != 1 or len(brand_row) != 1:
        raise SystemExit(f"no single Traficom row for {brand}/{model}")
    row = pd.concat([model_row.iloc[0], brand_row.iloc[0]])
    return row[registry_columns]


def build_features(september, february, brand, model, feature_names):
    """September rows in the bundle's feature schema.

    The Traficom registry columns are constant per brand/model, so they are
    copied from any February row for that car rather than recomputed. A car
    February never saw takes them from the Traficom summaries instead.
    """
    car = february[(february.brand == brand) & (february.model == model)]
    registry_columns = [
        name for name in feature_names if name.startswith(REGISTRY_PREFIXES)
    ]
    if car.empty:
        constants = traficom_constants(brand, model, registry_columns)
    else:
        constants = car.iloc[0][registry_columns]
    for column in registry_columns:
        if car[column].nunique(dropna=False) > 1:
            raise SystemExit(f"{column} is not constant for {brand}/{model}")

    # February's part_name strings come from the product page's h1 run through
    # clean_part_name, so they carry per-car residue: Golf listings read
    # "Tank lid - , e-" because the crawler stripped "Golf" out of
    # "Tank lid - VW Golf, e-Golf" on the merged Golf/e-Golf family pages.
    # Rebuilding the string by hand silently mismatches that car, and with
    # onehot_min_frequency=3 a mismatch does not raise -- it drops into the
    # infrequent bucket and kills a 9.6%-SHAP feature without a warning. So
    # take February's own dominant spelling per subcategory instead.
    february_part_names = (
        car.groupby("subcategory").part_name.agg(lambda names: names.mode().iat[0])
    )
    frame = pd.DataFrame(
        {
            "part_name": september.subcategory.map(february_part_names).fillna(
                september.part_label.str.strip() + " -"
            ),
            "quality_grade": september.quality_grade if SCORE_WITH_QUALITY_GRADE else np.nan,
            "mileage": september.mileage,
            "brand": brand,
            "model": model,
            "category": september.category,
            "subcategory": september.subcategory,
            "year_start": september.year_start,
            "year_end": september.year_end,
            "year_span": september.year_end - september.year_start,
            "year_mid": (september.year_end + september.year_start) / 2,
            "repair_status": "original_valid",
        }
    )
    for column in registry_columns:
        frame[column] = constants[column]
    return ensure_feature_frame(frame, feature_names)


def three_per_cell(listings):
    """The reported draw: the dearest, the middle and the cheapest of each cell.

    Ties are broken on product_id so the same pages always yield the same rows;
    16 of the 33 cells hold a tie at the cut, so an explicit rule is required
    for the draw to be reproducible at all. A cell with fewer than three
    listings would put one listing in several tiers, so it is refused.
    """
    rows = []
    for _, cell in listings.groupby(["car", "subcategory"]):
        if len(cell) < 3:
            raise SystemExit(f"cell {cell.car.iat[0]}/{cell.subcategory.iat[0]} has {len(cell)} listings")
        cell = cell.sort_values(
            ["price", "product_id"], ascending=[False, True]
        ).reset_index(drop=True)
        for tier, position in [
            ("expensive", 0),
            ("middle", len(cell) // 2),
            ("cheapest", len(cell) - 1),
        ]:
            rows.append(cell.iloc[position].to_dict() | {"tier": tier})
    drawn = pd.DataFrame(rows)
    for column in ("price", "predicted", "heuristic"):
        drawn[column] = drawn[column].astype(float)
    drawn["ape_model"] = (drawn.price - drawn.predicted).abs() / drawn.price * 100
    drawn["ape_heuristic"] = (drawn.price - drawn.heuristic).abs() / drawn.price * 100
    return drawn


def main(listings_path, brand, model, generation=None):
    september = pd.read_csv(listings_path)
    february = pd.read_csv(FEBRUARY, low_memory=False)
    bundle = load_random_forest_bundle(BUNDLE)
    feature_names = bundle["metadata"]["feature_names"]

    # a cell saved over several pages can repeat a listing at the page boundary
    september = september.drop_duplicates("product_id").reset_index(drop=True)
    print(f"{len(september):4d} listings parsed")
    if generation:
        keep = september.generation.str.contains(generation, regex=True)
        print(f"\ngeneration filter {generation!r}")
        for label, n in september.generation.value_counts().items():
            print(f"  {'keep' if keep[september.generation == label].iat[0] else 'drop'} {n:4d}  {label}")
        september = september[keep].reset_index(drop=True)
        print(f"{len(september):4d} listings kept")
    seen = set(february.product_id.astype(int))
    september["seen_in_february"] = september.product_id.isin(seen)
    print(f"-{september.seen_in_february.sum():3d} already in the February training data")
    unseen = september[~september.seen_in_february].reset_index(drop=True)
    print(f"{len(unseen):4d} unseen listings  <- census")

    features = build_features(unseen, february, brand, model, feature_names)
    unseen["predicted"] = bundle["model"].predict(features)

    # Subcategory-median heuristic, fitted on this car's February rows. Ch4's
    # headline finding is that this flat per-part median beats the tuned forest,
    # so it is the comparator the September study has to clear.
    # A vehicle February never saw has no medians of its own, so it falls back to
    # the all-brand subcategory median. That baseline is weaker (86.0% against
    # 58.0% on the Corolla matched draw), so a Round 2 gap to it is not
    # comparable with a Round 1 gap.
    car = february[(february.brand == brand) & (february.model == model)]
    global_median = float(february.price.median())
    subcategory_medians = (car if len(car) else february).groupby("subcategory").price.median()
    unseen["heuristic"] = unseen.subcategory.map(subcategory_medians).fillna(global_median)

    # did February hold this part x generation cell at all?
    covered_cells = set(
        zip(car.subcategory, car.year_start.astype(int), car.year_end.astype(int))
    )
    unseen["february_covered"] = [
        (row.subcategory, row.year_start, row.year_end) in covered_cells
        for row in unseen.itertuples()
    ]

    # matched draw: per part x generation, the top five by price
    matched = (
        unseen.sort_values("price", ascending=False)
        .groupby(["subcategory", "year_start", "year_end"], sort=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    print(f"{len(matched):4d} generation x top-{TOP_N} by price  <- matched draw (primary)")

    covered = matched[matched.february_covered]
    uncovered = matched[~matched.february_covered]
    rows = [{"sample": "MATCHED DRAW (primary)", **metrics(matched.price, matched.predicted)}]
    if len(car):  # every cell of a vehicle February never saw is uncovered
        rows += [
            {"sample": "- February HAD the cell", **metrics(covered.price, covered.predicted)},
            {"sample": "- cell never seen", **metrics(uncovered.price, uncovered.predicted)},
        ]
    rows.append({"sample": "CENSUS (market coverage)", **metrics(unseen.price, unseen.predicted)})
    show("Random Forest", rows)

    show(
        "Subcategory-median heuristic",
        [
            {"sample": "MATCHED DRAW", **metrics(matched.price, matched.heuristic)},
            {"sample": "CENSUS", **metrics(unseen.price, unseen.heuristic)},
        ],
    )

    print("\nMatched draw by price band")
    band = pd.cut(matched.price, BANDS, labels=BAND_LABELS, right=False)
    band_rows = []
    for label, group in matched.groupby(band, observed=True):
        band_rows.append(
            {
                "band": label,
                "n": len(group),
                "RF_MdAPE": metrics(group.price, group.predicted)["MdAPE_pct"],
                "heuristic_MdAPE": metrics(group.price, group.heuristic)["MdAPE_pct"],
            }
        )
    print(pd.DataFrame(band_rows).to_string(index=False, float_format="{:,.1f}".format))

    print("\nPer part, matched draw")
    part_rows = []
    for part, group in matched.groupby("subcategory"):
        part_rows.append(
            {
                "part": part,
                "n": len(group),
                "median_price": group.price.median(),
                "median_pred": group.predicted.median(),
                "RF_MdAPE": metrics(group.price, group.predicted)["MdAPE_pct"],
                "heur_MdAPE": metrics(group.price, group.heuristic)["MdAPE_pct"],
                "covered": int(group.february_covered.sum()),
            }
        )
    print(
        pd.DataFrame(part_rows)
        .sort_values("RF_MdAPE")
        .to_string(index=False, float_format="{:,.1f}".format)
    )

    # bootstrap the headline and the gap to the heuristic
    rng = np.random.default_rng(32)
    actual = matched.price.to_numpy(dtype=float)
    predicted = matched.predicted.to_numpy(dtype=float)
    heuristic = matched.heuristic.to_numpy(dtype=float)
    draws_rf, draws_gap = [], []
    for _ in range(5000):
        idx = rng.integers(0, len(actual), len(actual))
        rf_mdape = np.median(np.abs(actual[idx] - predicted[idx]) / actual[idx]) * 100
        heur_mdape = np.median(np.abs(actual[idx] - heuristic[idx]) / actual[idx]) * 100
        draws_rf.append(rf_mdape)
        draws_gap.append(rf_mdape - heur_mdape)
    print(
        f"\nBootstrap, 5000 resamples of the matched draw"
        f"\n  RF MdAPE          {np.median(draws_rf):.1f}%"
        f"  95% CI {np.percentile(draws_rf, 2.5):.1f}% to {np.percentile(draws_rf, 97.5):.1f}%"
        f"\n  RF minus heuristic {np.median(draws_gap):+.1f} points"
        f"  95% CI {np.percentile(draws_gap, 2.5):+.1f} to {np.percentile(draws_gap, 97.5):+.1f}"
    )

    unseen["car"] = model
    drawn = three_per_cell(unseen)
    print(f"\nReported draw, dearest / middle / cheapest per cell, n={len(drawn)}")
    tier_rows = []
    for tier in ("expensive", "middle", "cheapest"):
        group = drawn[drawn.tier == tier]
        tier_rows.append(
            {
                "tier": tier,
                "n": len(group),
                "median_price": group.price.median(),
                "RF_MdAPE": group.ape_model.median(),
                "heuristic_MdAPE": group.ape_heuristic.median(),
            }
        )
    print(pd.DataFrame(tier_rows).to_string(index=False, float_format="{:,.1f}".format))
    print(
        f"all {len(drawn)}: RF {drawn.ape_model.median():.1f}%"
        f"  heuristic {drawn.ape_heuristic.median():.1f}%"
        f"  RF closer on {(drawn.ape_model < drawn.ape_heuristic).sum()}"
    )

    out = Path(listings_path).with_name(Path(listings_path).stem + "_scored.csv")
    unseen.to_csv(out, index=False)
    print(f"\nscored rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
