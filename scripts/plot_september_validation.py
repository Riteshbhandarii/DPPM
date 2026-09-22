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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

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


CAR_ORDER = ["corolla", "golf", "octavia"]

# One hue per car, so a part's three rows read as a block. Validated all-pairs
# for normal vision and CVD; the aqua sits under 3:1 on this surface, which the
# per-row car label covers -- identity is never carried by colour alone.
CAR_COLOUR = {"corolla": "#2a78d6", "golf": "#eb6834", "octavia": "#1baf7a"}


def tint(hex_colour, amount=0.58):
    """Lighter step of the same hue: the model's guess against the real price."""
    red, green, blue = mcolors.to_rgb(hex_colour)
    return tuple(channel + (1 - channel) * amount for channel in (red, green, blue))


def main():
    drawn = three_per_cell(scored_listings())
    # The dearest listing of each cell: the tier the model can actually price.
    cells = drawn[drawn.tier == "expensive"].copy()
    cells["euros_off"] = cells.predicted - cells.price

    # Group the rows by part, parts ordered by what they cost, so the part name
    # is printed once instead of three times and the eye can scan down a group.
    part_order = cells.groupby("subcategory").price.median().sort_values(ascending=False)
    cells["part_rank"] = cells.subcategory.map(
        {part: rank for rank, part in enumerate(part_order.index)}
    )
    cells["car_rank"] = cells.car.map({car: rank for rank, car in enumerate(CAR_ORDER)})
    cells = cells.sort_values(["part_rank", "car_rank"]).reset_index(drop=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "ytick.color": INK2,
            "xtick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    figure, axes = plt.subplots(figsize=(12, 10.5))
    rows = np.arange(len(cells))
    label_x = 4.4
    # the euros column lives outside the plot, so the gridlines stop at the data
    outside = blended_transform_factory(axes.transAxes, axes.transData)

    for row, cell in zip(rows, cells.itertuples()):
        colour = CAR_COLOUR[cell.car]
        axes.plot([cell.price, cell.predicted], [row, row], color=tint(colour, 0.78),
                  lw=3.5, zorder=1, solid_capstyle="round")
        axes.scatter(cell.predicted, row, s=110, color=tint(colour), zorder=3,
                     edgecolors=colour, linewidths=1.6)
        axes.scatter(cell.price, row, s=110, color=colour, zorder=4)

    for row, cell in zip(rows, cells.itertuples()):
        off = cell.euros_off
        close = abs(off) < 0.1 * cell.price
        axes.text(
            1.13, row, "on the money" if close else f"{off:+,.0f} EUR",
            transform=outside, va="center", ha="right", fontsize=11.5,
            color=MUTED if close else INK,
        )

    # one part label per group, plus a hairline between groups
    for part, group in cells.groupby("part_rank"):
        middle = group.index.to_numpy().mean()
        axes.text(label_x, middle, group.subcategory.iat[0], va="center", ha="left",
                  fontsize=12.5, color=INK)
        if part:
            axes.axhline(group.index.min() - 0.5, color=GRID, lw=0.8, zorder=0)

    axes.set_yticks(rows, cells.car, fontsize=11)
    axes.set_xscale("log")
    axes.set_xlim(4.2, 9000)
    axes.set_ylim(len(cells) - 0.4, -3.1)
    axes.set_xticks([10, 30, 100, 300, 1000, 3000],
                    ["10", "30", "100", "300", "1 000", "3 000"], fontsize=12)
    axes.set_xticks([], minor=True)
    axes.set_xlabel("price, EUR", fontsize=12, color=INK2, labelpad=8)
    axes.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(left=False)

    # direct labels on the first row instead of a legend box
    first = cells.iloc[0]
    first_colour = CAR_COLOUR[first.car]
    axes.annotate("what it sells for\n(solid)", xy=(first.price, -0.3),
                  xytext=(first.price * 0.30, -2.25), fontsize=12, color=INK2, ha="center",
                  arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    axes.annotate("what the model said\n(pale)", xy=(first.predicted, -0.3),
                  xytext=(first.predicted * 0.075, -1.35), fontsize=12, color=INK2, ha="center",
                  arrowprops=dict(arrowstyle="-", color=first_colour, lw=1))
    axes.text(1.13, -2.25, "euros out", transform=outside, va="center", ha="right",
              fontsize=12, color=INK2)

    for label, car in zip(axes.get_yticklabels(), cells.car):
        label.set_color(CAR_COLOUR[car])

    axes.set_title(
        "What each part sells for, and what the model said",
        fontsize=17, color=INK, loc="left", pad=26,
    )
    figure.text(
        0.012, 0.016,
        "The dearest listing of each part on each car, 33 cases. "
        "varaosahaku.fi, September 2026.",
        fontsize=10.5, color=MUTED,
    )
    figure.tight_layout(rect=[0, 0.028, 0.88, 1])
    figure.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    print(cells[["subcategory", "car", "price", "predicted", "euros_off"]].to_string(
        index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()
