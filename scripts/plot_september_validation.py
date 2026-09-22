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
    floor = float(listings.predicted.min())

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    left.scatter(
        listings.price, listings.predicted, s=9, c=CONTEXT, edgecolors="none",
        zorder=1, label=f"all {len(listings):,} listings (not scored)",
    )
    left.scatter(
        drawn.price, drawn.predicted, s=42, c=BLUE, edgecolors=SURFACE,
        linewidths=1.2, zorder=3, label=f"the {len(drawn)} scored (3 per part x car)",
    )
    limits = [4, 9000]
    left.plot(limits, limits, color=INK2, lw=1.2, ls="--", zorder=2)
    left.axhline(floor, color=ORANGE, lw=2, zorder=2)
    left.annotate(
        f"prediction floor, {floor:.0f} EUR\nthe model cannot go below this",
        xy=(6.5, floor), xytext=(6.5, 11), color=ORANGE, fontsize=9.5,
        va="center", ha="left", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1),
    )
    left.text(
        950, 1750, "a perfect prediction\nwould sit on this line", color=INK2,
        fontsize=9, rotation=32, rotation_mode="anchor", ha="center", va="bottom",
    )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(limits)
    left.set_ylim(limits)
    plain = FuncFormatter(lambda value, _: f"{value:,.0f}")
    left.xaxis.set_major_formatter(plain)
    left.yaxis.set_major_formatter(plain)
    left.set_xlabel("actual asking price, EUR")
    left.set_ylabel("model prediction, EUR")
    left.set_title(
        "Every cheap part is predicted at the floor", fontsize=12.5, color=INK,
        loc="left", pad=10,
    )
    left.grid(True, color=GRID, lw=0.7, zorder=0)
    left.set_axisbelow(True)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)
    left.legend(
        frameon=False, loc="lower right", fontsize=9.5, labelcolor=INK2,
        bbox_to_anchor=(1, 0.02),
    )

    model_error = [drawn[drawn.tier == t].ape_model.median() for t in TIERS]
    heuristic_error = [drawn[drawn.tier == t].ape_heuristic.median() for t in TIERS]
    positions = np.arange(3)
    height = 0.36
    right.barh(
        positions - height / 2 - 0.01, model_error, height, color=BLUE,
        label="random forest", zorder=3,
    )
    right.barh(
        positions + height / 2 + 0.01, heuristic_error, height, color=ORANGE,
        label="per-part median (baseline)", zorder=3,
    )
    for row, value in zip(positions - height / 2 - 0.01, model_error):
        right.text(value + 7, row, f"{value:.0f}%", va="center", fontsize=10, color=INK2)
    for row, value in zip(positions + height / 2 + 0.01, heuristic_error):
        right.text(value + 7, row, f"{value:.0f}%", va="center", fontsize=10, color=INK2)
    right.set_yticks(
        positions,
        [f"{t}\nmedian {drawn[drawn.tier == t].price.median():,.0f} EUR" for t in TIERS],
        fontsize=10,
    )
    right.invert_yaxis()
    right.set_xlim(0, 420)
    right.set_xlabel("median error, % of the actual price")
    right.set_title(
        "Accuracy collapses at the cheap end", fontsize=12.5, color=INK, loc="left", pad=10
    )
    right.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    right.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        right.spines[spine].set_visible(False)
    right.legend(
        frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK2,
        bbox_to_anchor=(1, 0.36),
    )

    figure.suptitle(
        "September live validation: 99 listings, 11 parts, 3 cars",
        fontsize=14, color=INK, x=0.037, ha="left", y=0.985,
    )
    figure.text(
        0.037, 0.015,
        "Saved varaosahaku.fi result pages, 2026-09-17 to 09-22. From each part x car cell: "
        "the most expensive, the middle and the cheapest listing. Listings already in the "
        "February training data removed.",
        fontsize=8.5, color=MUTED,
    )
    figure.tight_layout(rect=[0, 0.045, 1, 0.95])
    figure.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    print(
        pd.DataFrame(
            {"tier": TIERS, "random_forest": model_error, "heuristic": heuristic_error}
        ).to_string(index=False, float_format="{:,.1f}".format)
    )


if __name__ == "__main__":
    main()
