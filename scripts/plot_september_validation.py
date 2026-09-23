"""
Purpose:
Draw the September live-validation figure: every listing the study reports,
the dearest, the middle and the cheapest of each part x car cell, as three
panels that share their rows. Each point pair is one real listing: its observed
asking price and the frozen random forest's prediction for it.

All three tiers are drawn because the model behaves differently across them,
and a figure of the dearest tier alone reads as a best case. The comparison
with the per-part median baseline is not drawn; it belongs in the tier table
that accompanies the figure.

Round 2 draws the same figure for the three vehicles February never saw, 9
parts x 3 = 27 cells.

Inputs:
- <car>_sept_scored.csv, written by scripts/score_september_validation.py
  (kept outside the repo; see .gitignore)

Outputs:
- results/september_live_validation/september_validation.png  (round 1)
- results/september_live_validation/september_validation_round2.png  (round 2)
  each with a .pdf alongside

How to run:
  .venv/bin/python scripts/plot_september_validation.py [1|2]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_september_validation import three_per_cell

LISTINGS = Path.home() / "Desktop/validation dataset/parsed"
RESULTS = ROOT / "results/september_live_validation"

# Built at the thesis text width so the type lands at its intended size on the
# page. Scaling a figure down in Word is what makes labels unreadable in print.
TEXT_WIDTH_INCHES = 6.3
GROUP_GAP = 1.15  # blank rows between parts; a hairline alone let the blocks merge
CARS = [("corolla", "toyota"), ("golf", "vw"), ("octavia", "skoda")]

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8a87", "#e4e3df", "#fcfcfb"


def scored_listings(cars):
    """Every unseen September listing, as the scorer predicted and baselined it.

    Reading the scorer's own output keeps the figure and the reports on the
    same rows, including Round 2's generation filter and fallback baseline.
    """
    frames = []
    for model in cars:
        listings = pd.read_csv(LISTINGS / f"{model}_sept_scored.csv")
        listings["car"] = model
        frames.append(listings)
    return pd.concat(frames, ignore_index=True)


# One hue per car, so a part's three rows read as a block. Validated all-pairs
# for normal vision and CVD; the aqua sits under 3:1 on this surface, which the
# per-row car label covers -- identity is never carried by colour alone.
# Round 2 reuses the same three hues in the same order: the two figures are
# never shown on one axis, and every row carries its car's name.
HUES = ["#2a78d6", "#eb6834", "#1baf7a"]
ROUNDS = {
    "1": {
        "cars": {"corolla": "Corolla", "golf": "Golf", "octavia": "Octavia"},
        "out": "september_validation.png",
    },
    "2": {
        "cars": {"ford": "Focus", "nissan": "Qashqai", "volvo": "V70"},
        "out": "september_validation_round2.png",
    },
}


def academic(name):
    """Subcategory label as it should read in a thesis figure."""
    if name.startswith("abs "):
        return "ABS " + name[4:]
    return name[0].upper() + name[1:]


def tint(hex_colour, amount=0.58):
    """Lighter step of the same hue: the model's guess against the real price."""
    red, green, blue = mcolors.to_rgb(hex_colour)
    return tuple(channel + (1 - channel) * amount for channel in (red, green, blue))


TIERS = [
    ("expensive", "Dearest listing"),
    ("middle", "Middle listing"),
    ("cheapest", "Cheapest listing"),
]


def draw_key(figure):
    """The key, drawn by hand in figure coordinates so it sits above the panels.

    Only the two markers are keyed. Every row already carries its car's name
    beside a dot in the car's colour, so a colour key would say it twice. The
    per-part median is deliberately not drawn: the figure shows the raw result,
    and the comparison with the baseline is reported in the tier table beside it.
    """
    box = figure.add_axes([0.30, 0.955, 0.46, 0.030])
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
    ]
    for index, (shape, face, edge, text) in enumerate(marker_row):
        x = 0.08 + index * 0.50
        box.plot(x, 0.5, marker=shape, markersize=5, markerfacecolor=face,
                 markeredgecolor=edge, markeredgewidth=1.2, linestyle="none")
        box.text(x + 0.06, 0.5, text, va="center", fontsize=7.8, color=INK2)


def main(round_number="1"):
    setup = ROUNDS[round_number]
    names = setup["cars"]
    colours = dict(zip(names, HUES))
    out = RESULTS / setup["out"]
    drawn = three_per_cell(scored_listings(names))

    # One row per part x car, the same rows in all three panels, so a row can be
    # read straight across from its dearest to its cheapest listing. Parts are
    # ordered by what their dearest listing costs.
    dearest = drawn[drawn.tier == "expensive"]
    part_order = dearest.groupby("subcategory").price.median().sort_values(ascending=False)
    part_rank = {part: rank for rank, part in enumerate(part_order.index)}
    car_rank = {car: rank for rank, car in enumerate(names)}
    layout = (
        dearest[["subcategory", "car"]]
        .assign(
            part_rank=lambda frame: frame.subcategory.map(part_rank),
            car_rank=lambda frame: frame.car.map(car_rank),
        )
        .sort_values(["part_rank", "car_rank"])
        .reset_index(drop=True)
    )
    # Each part occupies three consecutive rows, then a blank gap that also
    # carries the part's name.
    layout["row"] = [
        position + cell.part_rank * GROUP_GAP
        for position, cell in enumerate(layout.itertuples())
    ]
    row_of = {(cell.subcategory, cell.car): cell.row for cell in layout.itertuples()}

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
    # 8.5 in held the 33 rows of round 1; fewer rows keep the same row pitch
    groups = layout.part_rank.nunique()
    height = 8.5 * (len(layout) + groups * GROUP_GAP + 6) / (33 + 11 * GROUP_GAP + 6)
    figure, panels = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH_INCHES, height), sharex=True, sharey=True
    )

    for axes, (tier, title) in zip(panels, TIERS):
        for cell in drawn[drawn.tier == tier].itertuples():
            row = row_of[(cell.subcategory, cell.car)]
            colour = colours[cell.car]
            axes.plot([cell.price, cell.predicted], [row, row], color=tint(colour, 0.78),
                      lw=2, zorder=1, solid_capstyle="round")
            axes.scatter(cell.predicted, row, s=22, color=SURFACE, zorder=3,
                         edgecolors=colour, linewidths=1.1)
            axes.scatter(cell.price, row, s=22, color=colour, zorder=4)
        axes.set_title(title, fontsize=8.5, color=INK, pad=6)
        # One shared log axis across the panels, so a gap in one panel is the
        # same ratio as an equally long gap in another.
        axes.set_xscale("log")
        axes.set_xlim(4, 9000)
        axes.set_xticks([10, 100, 1000], ["10", "100", "1 000"], fontsize=7.5)
        axes.set_xticks([], minor=True)
        axes.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
        axes.set_axisbelow(True)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.tick_params(left=False)

    first = panels[0]
    first.set_yticks(layout.row, [names[car] for car in layout.car], fontsize=7)
    first.tick_params(axis="y", pad=9)
    first.set_ylim(layout.row.max() + 0.7, -1.4)
    beside = blended_transform_factory(first.transAxes, first.transData)
    # the car's colour sits beside its own row rather than in a key at the top
    for cell in layout.itertuples():
        first.plot(-0.035, cell.row, marker="o", markersize=3.5, color=colours[cell.car],
                   transform=beside, clip_on=False, zorder=5)
    # the part's name sits in the gap above its three rows
    for _, group in layout.groupby("part_rank"):
        first.text(-0.62, group.row.min() - 0.95, academic(group.subcategory.iat[0]),
                   transform=beside, va="center", ha="left", fontsize=7.8, color=INK)

    figure.supxlabel("Asking price, EUR (logarithmic scale)", fontsize=8.5,
                     color=INK2, y=0.012)
    # No in-image title: in the thesis the caption below the figure carries it.
    figure.tight_layout(rect=[0.03, 0.02, 1, 0.945], w_pad=0.6)
    draw_key(figure)
    figure.savefig(out, dpi=300)
    figure.savefig(out.with_suffix(".pdf"))
    print(f"saved {out}\nsaved {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
