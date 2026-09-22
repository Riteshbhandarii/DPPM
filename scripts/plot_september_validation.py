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
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_september_validation import BUNDLE, FEBRUARY, build_features
from src.random_forest_serving import load_random_forest_bundle

LISTINGS = Path.home() / "Desktop/validation dataset/parsed"
OUT = ROOT / "results/september_live_validation/september_validation.png"
OUT_VECTOR = OUT.with_suffix(".pdf")

# Built at the thesis text width so the type lands at its intended size on the
# page. Scaling a figure down in Word is what makes labels unreadable in print.
TEXT_WIDTH_INCHES = 6.3
GROUP_GAP = 1.15  # blank rows between parts; a hairline alone let the blocks merge
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


def academic(name):
    """Subcategory label as it should read in a thesis figure."""
    if name.startswith("abs "):
        return "ABS " + name[4:]
    return name[0].upper() + name[1:]


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
    figure, axes = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 8.5))
    # Each part occupies three consecutive rows, then a blank gap, so the eye
    # reads one block per part instead of a continuous 33-row list.
    rows = np.array([
        position + cell.part_rank * GROUP_GAP
        for position, cell in enumerate(cells.itertuples())
    ])
    cells = cells.assign(row=rows)
    label_x = 4.6
    # the euros column lives outside the plot, so the gridlines stop at the data
    outside = blended_transform_factory(axes.transAxes, axes.transData)

    for row, cell in zip(rows, cells.itertuples()):
        colour = CAR_COLOUR[cell.car]
        axes.plot([cell.price, cell.predicted], [row, row], color=tint(colour, 0.78),
                  lw=2, zorder=1, solid_capstyle="round")
        axes.scatter(cell.predicted, row, s=34, color=SURFACE, zorder=3,
                     edgecolors=colour, linewidths=1.3)
        axes.scatter(cell.price, row, s=34, color=colour, zorder=4)

    for row, cell in zip(rows, cells.itertuples()):
        off = cell.euros_off
        close = abs(off) < 0.1 * cell.price
        axes.text(
            1.15, row, "within 10%" if close else f"{off:+,.0f}",
            transform=outside, va="center", ha="right", fontsize=7.5,
            color=MUTED if close else INK,
        )

    # one part label per group; the blank rows do the separating
    for _, group in cells.groupby("part_rank"):
        axes.text(label_x, group.row.mean(), academic(group.subcategory.iat[0]),
                  va="center", ha="left", fontsize=8.5, color=INK)

    axes.set_yticks(rows, [car.capitalize() for car in cells.car], fontsize=7.5)
    axes.tick_params(axis="y", pad=1)
    axes.set_xscale("log")
    axes.set_xlim(4.2, 9000)
    axes.set_ylim(rows.max() + 0.7, -3.0)
    axes.set_xticks([10, 30, 100, 300, 1000, 3000],
                    ["10", "30", "100", "300", "1 000", "3 000"], fontsize=8)
    axes.set_xticks([], minor=True)
    axes.set_xlabel("Asking price, EUR (logarithmic scale)", fontsize=8.5,
                    color=INK2, labelpad=6)
    axes.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(left=False)

    # direct labels on the first row instead of a legend box
    first = cells.iloc[0]
    # One colour key for the cars. The marker shape is explained in words
    # underneath rather than as a second legend: two legends on a 16 cm figure
    # end up on the same line and overlap.
    car_key = [
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               color=CAR_COLOUR[car], label=car.capitalize())
        for car in CAR_ORDER
    ]
    axes.legend(
        handles=car_key, frameon=False, fontsize=8, labelcolor=INK2,
        loc="upper left", bbox_to_anchor=(0.0, 1.075), ncol=3,
        handletextpad=0.4, columnspacing=1.6,
    )
    axes.text(
        0.0, 1.035, "Filled marker: observed price.   Open marker: predicted price.",
        transform=axes.transAxes, fontsize=8, color=INK2, va="top",
    )
    axes.text(1.15, -1.7, "Error, EUR", transform=outside, va="center", ha="right",
              fontsize=8, color=INK2)

    # No in-image title: in the thesis the caption below the figure carries it.
    figure.tight_layout(rect=[0, 0, 0.86, 0.975])
    figure.savefig(OUT, dpi=300)
    figure.savefig(OUT_VECTOR)
    print(f"saved {OUT}\nsaved {OUT_VECTOR}")
    print(cells[["subcategory", "car", "price", "predicted", "euros_off"]].to_string(
        index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()
