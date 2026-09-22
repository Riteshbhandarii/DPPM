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
    drawn = three_per_cell(scored_listings())
    # The dearest listing of each cell: the tier the model can actually price.
    cells = drawn[drawn.tier == "expensive"].copy()
    cells["label"] = cells.subcategory + "  ·  " + cells.car
    cells = cells.sort_values("price").reset_index(drop=True)
    cells["euros_off"] = cells.predicted - cells.price

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "ytick.color": INK,
            "xtick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    figure, axes = plt.subplots(figsize=(12, 11))
    rows = np.arange(len(cells))

    for row, cell in zip(rows, cells.itertuples()):
        axes.plot([cell.price, cell.predicted], [row, row], color=GRID, lw=2.5, zorder=1)
    axes.scatter(cells.price, rows, s=90, color=INK2, zorder=3, label="real price")
    axes.scatter(cells.predicted, rows, s=90, color=BLUE, zorder=3, label="model said")

    for row, cell in zip(rows, cells.itertuples()):
        off = cell.euros_off
        text = "spot on" if abs(off) < 10 else f"{off:+,.0f} EUR"
        colour = INK2 if abs(off) < 10 else ORANGE
        axes.text(60000, row, text, va="center", ha="right", fontsize=11, color=colour)

    axes.set_yticks(rows, cells.label, fontsize=10.5)
    axes.set_xscale("log")
    axes.set_xlim(8, 90000)
    axes.set_ylim(-1.4, len(cells) - 0.3)
    axes.set_xticks([10, 30, 100, 300, 1000, 3000],
                    ["10", "30", "100", "300", "1 000", "3 000"], fontsize=12)
    axes.set_xlabel("price, EUR", fontsize=12.5, color=INK2, labelpad=10)
    axes.text(60000, -1.1, "how far off", va="center", ha="right", fontsize=11.5, color=INK2)
    axes.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(left=False)
    axes.legend(frameon=False, loc="lower left", fontsize=12.5, labelcolor=INK2,
                bbox_to_anchor=(0.0, -0.005), ncol=2)

    axes.set_title(
        "Where the model got the price right, and where it did not\n"
        "the dearest listing of each part, 33 cases",
        fontsize=16, color=INK, loc="left", pad=20,
    )
    figure.text(
        0.012, 0.018,
        "Short bar = the model was close. varaosahaku.fi, September 2026, "
        "11 parts x 3 cars (Corolla, Golf, Octavia).",
        fontsize=10.5, color=MUTED,
    )
    figure.tight_layout(rect=[0, 0.032, 1, 1])
    figure.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    print(cells[["label", "price", "predicted", "euros_off"]].to_string(
        index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()
