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
    figure, axes = plt.subplots(figsize=(12, 5.8))
    rng = np.random.default_rng(32)

    # One dot per cell, so the spread stays visible: a single median per tier
    # hides that the dearest listings cluster tightly while the cheap tail
    # scatters over two orders of magnitude.
    axes.axvspan(1, 50, color="#e8f0fb", zorder=0)
    axes.axvline(50, color=BLUE, lw=1.2, ls="--", zorder=1)

    for row, tier in enumerate(TIERS):
        errors = drawn[drawn.tier == tier].ape_model.clip(lower=1)
        axes.scatter(
            errors, row + rng.uniform(-0.13, 0.13, len(errors)),
            s=95, color=BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=1.2, zorder=3,
        )
        axes.text(
            3900, row, f"{(errors <= 50).sum()} of {len(errors)}",
            va="center", ha="right", fontsize=17, color=BLUE,
        )

    axes.text(7, -0.72, "USABLE: within 50% of the real price",
              fontsize=13, color=BLUE, va="center")
    axes.text(3900, -0.72, "how many land\nin the usable band", fontsize=12,
              color=INK2, va="center", ha="right")

    axes.set_xscale("log")
    axes.set_xlim(1, 4200)
    axes.set_ylim(2.6, -1.0)
    axes.set_xticks([1, 10, 50, 100, 1000], ["1%", "10%", "50%", "100%", "1000%"],
                    fontsize=13)
    axes.set_yticks(range(3),
                    ["dearest listing\nof each part", "middle listing\nof each part",
                     "cheapest listing\nof each part"], fontsize=13)
    axes.set_xlabel("how far off the model was, % of the real price", fontsize=13,
                    color=INK2, labelpad=10)
    axes.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(left=False)

    axes.set_title(
        "The model works on the dearest part in each group, and only there",
        fontsize=17, color=INK, loc="left", pad=22,
    )
    figure.text(
        0.012, 0.028,
        "One dot = one part on one car, 33 cells (11 parts x 3 cars: Corolla, Golf, Octavia). "
        "varaosahaku.fi, September 2026.",
        fontsize=10.5, color=MUTED,
    )
    figure.tight_layout(rect=[0, 0.055, 1, 1])
    figure.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    for tier in TIERS:
        errors = drawn[drawn.tier == tier].ape_model
        print(f"{tier:10s} median {errors.median():6.0f}%   within 50%: "
              f"{(errors <= 50).sum():2d}/{len(errors)}   within 25%: {(errors <= 25).sum():2d}/{len(errors)}")


if __name__ == "__main__":
    main()
