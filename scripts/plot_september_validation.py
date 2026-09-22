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


def draw_key(figure):
    """The key, drawn by hand in figure coordinates.

    Two matplotlib legends stacked here end up on the same line at this width
    and overlap, so the marker row and the colour row are placed explicitly.
    """
    box = figure.add_axes([0.10, 0.930, 0.80, 0.058])
    box.set_xlim(0, 1)
    box.set_ylim(0, 1)
    box.set_xticks([])
    box.set_yticks([])
    for side, spine in box.spines.items():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    box.set_facecolor(SURFACE)

    marker_row = [
        ("o", MUTED, MUTED, "Observed price"),
        ("o", SURFACE, MUTED, "Random forest"),
        ("D", SURFACE, MUTED, "Per-part median"),
    ]
    for index, (shape, face, edge, text) in enumerate(marker_row):
        x = 0.035 + index * 0.325
        box.plot(x, 0.70, marker=shape, markersize=5, markerfacecolor=face,
                 markeredgecolor=edge, markeredgewidth=1.2, linestyle="none")
        box.text(x + 0.032, 0.70, text, va="center", fontsize=7.8, color=INK2)

    for index, car in enumerate(CAR_ORDER):
        x = 0.035 + index * 0.325
        box.plot(x, 0.26, marker="o", markersize=5, color=CAR_COLOUR[car],
                 linestyle="none")
        box.text(x + 0.032, 0.26, car.capitalize(), va="center", fontsize=7.8,
                 color=INK2)


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
    label_x = 2.7
    # the euros column lives outside the plot, so the gridlines stop at the data
    outside = blended_transform_factory(axes.transAxes, axes.transData)

    for row, cell in zip(rows, cells.itertuples()):
        colour = CAR_COLOUR[cell.car]
        axes.plot([cell.price, cell.predicted], [row, row], color=tint(colour, 0.78),
                  lw=2, zorder=1, solid_capstyle="round")
        axes.scatter(cell.predicted, row, s=34, color=SURFACE, zorder=3,
                     edgecolors=colour, linewidths=1.3)
        axes.scatter(cell.price, row, s=34, color=colour, zorder=4)
        # the per-part median baseline: grey, because it is not a vehicle-specific
        # method conceptually and must not compete with the car hues
        axes.scatter(cell.heuristic, row, s=26, marker="D", color=SURFACE,
                     edgecolors=MUTED, linewidths=1.1, zorder=2)

    for row, cell in zip(rows, cells.itertuples()):
        off = cell.euros_off
        close = abs(off) < 0.1 * cell.price
        axes.text(
            1.15, row, f"{off:+,.0f}", transform=outside, va="center", ha="right",
            fontsize=7.5, color=MUTED if close else INK,
        )

    # one part label per group; the blank rows do the separating
    for _, group in cells.groupby("part_rank"):
        axes.text(label_x, group.row.mean(), academic(group.subcategory.iat[0]),
                  va="center", ha="left", fontsize=8.2, color=INK)

    axes.set_yticks(rows, [car.capitalize() for car in cells.car], fontsize=7.5)
    axes.tick_params(axis="y", pad=11)
    # the car's colour sits beside its own row rather than in a key at the top
    for row, car in zip(rows, cells.car):
        axes.plot(-0.016, row, marker="o", markersize=4, color=CAR_COLOUR[car],
                  transform=outside, clip_on=False, zorder=5)
    axes.set_xscale("log")
    axes.set_xlim(2.6, 9000)
    axes.set_ylim(rows.max() + 0.7, -1.6)
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
    axes.text(1.15, -1.0, "Error, EUR (random forest)", transform=outside, va="center", ha="right",
              fontsize=8, color=INK2)

    # No in-image title: in the thesis the caption below the figure carries it.
    figure.tight_layout(rect=[0, 0, 0.86, 0.92])
    draw_key(figure)
    figure.savefig(OUT, dpi=300)
    figure.savefig(OUT_VECTOR)
    print(f"saved {OUT}\nsaved {OUT_VECTOR}")
    print(cells[["subcategory", "car", "price", "predicted", "euros_off"]].to_string(
        index=False, float_format="{:,.0f}".format))


if __name__ == "__main__":
    main()
