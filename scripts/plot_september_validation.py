"""
Purpose:
Draw the September live-validation figure: actual against predicted price for
every collected listing, and median error by price tier for the model and the
per-part median baseline.

The draw shown is the one the study reports: from each part x car cell, the most
expensive, the middle and the cheapest listing, 33 cells x 3 = 99 listings. Two
of the three points are a cell's extremes by construction, so the figure reports
the three tiers separately and never a pooled error figure.

Inputs:
- results/september_live_validation/<car>_sept.csv, from
  scripts/parse_september_listings.py (kept outside the repo; see .gitignore)
- artifacts/random_forest_final/full_data_bundle
- datasets/cleaned/clean_master_dataset.csv

Outputs:
- results/september_live_validation/september_validation.png

How to run:
  .venv/bin/python scripts/plot_september_validation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_september_validation import BUNDLE, FEBRUARY, build_features
from src.random_forest_serving import load_random_forest_bundle

LISTINGS = Path.home() / "Desktop/validation dataset/parsed"
OUT = ROOT / "results/september_live_validation/september_validation.png"
CARS = [("corolla", "toyota"), ("golf", "vw"), ("octavia", "skoda")]
TIERS = ["expensive", "middle", "cheapest"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8a87", "#e4e3df", "#fcfcfb"
CONTEXT = "#d9d8d3"


def scored_listings():
    """Every unseen September listing, with the model and baseline prediction."""
    february = pd.read_csv(FEBRUARY, low_memory=False)
    seen = set(february.product_id.astype(int))
    bundle = load_random_forest_bundle(BUNDLE)
    feature_names = bundle["metadata"]["feature_names"]

    frames = []
    for model, brand in CARS:
        listings = pd.read_csv(LISTINGS / f"{model}_sept.csv")
        listings = listings[~listings.product_id.isin(seen)].reset_index(drop=True)
        car = february[(february.brand == brand) & (february.model == model)]
        features = build_features(listings, february, brand, model, feature_names)
        listings["predicted"] = bundle["model"].predict(features)
        listings["heuristic"] = listings.subcategory.map(
            car.groupby("subcategory").price.median()
        )
        listings["car"] = model
        frames.append(listings)
    return pd.concat(frames, ignore_index=True)


def three_per_cell(listings):
    """The reported draw: the dearest, the middle and the cheapest of each cell.

    Ties are broken on product_id so the same pages always yield the same rows;
    16 of the 33 cells hold a tie at the cut, so an explicit rule is required
    for the draw to be reproducible at all.
    """
    rows = []
    for _, cell in listings.groupby(["car", "subcategory"]):
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


def main():
    listings = scored_listings()
    drawn = three_per_cell(listings)
    actual = [drawn[drawn.tier == t].price.median() for t in TIERS]
    predicted = [drawn[drawn.tier == t].predicted.median() for t in TIERS]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    figure, axes = plt.subplots(figsize=(10, 5.4))
    positions = np.arange(3)
    height = 0.34

    axes.barh(positions - height / 2 - 0.012, actual, height, color=INK2,
              label="what it actually sells for", zorder=3)
    axes.barh(positions + height / 2 + 0.012, predicted, height, color=BLUE,
              label="what the model predicts", zorder=3)
    for row, value in zip(positions - height / 2 - 0.012, actual):
        axes.text(value + 5, row, f"{value:,.0f} EUR", va="center", fontsize=12, color=INK2)
    for row, value in zip(positions + height / 2 + 0.012, predicted):
        axes.text(value + 5, row, f"{value:,.0f} EUR", va="center", fontsize=12, color=BLUE)

    axes.set_yticks(positions, ["the dearest\nlisting", "the middle\nlisting",
                                "the cheapest\nlisting"], fontsize=12.5)
    axes.invert_yaxis()
    axes.set_xlim(0, 330)
    axes.set_xticks([])
    axes.grid(False)
    for spine in ("top", "right", "left", "bottom"):
        axes.spines[spine].set_visible(False)
    axes.legend(frameon=False, loc="lower right", fontsize=11.5, labelcolor=INK2)

    axes.set_title(
        "The real price drops 9x. The model barely moves.",
        fontsize=16, color=INK, loc="left", pad=16,
    )
    figure.text(
        0.008, 0.035,
        "Median of 33 cells (11 parts x 3 cars), varaosahaku.fi, September 2026. From each cell: "
        "the dearest, the middle and the cheapest listing.",
        fontsize=9.5, color=MUTED,
    )
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    figure.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    print(pd.DataFrame({"tier": TIERS, "actual": actual, "model": predicted}).to_string(
        index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()
