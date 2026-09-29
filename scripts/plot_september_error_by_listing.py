"""
Purpose:
Draw the September extended-evaluation comparison as a figure: the median
absolute error of the frozen Random Forest and of the subcategory-median
heuristic, for the most expensive, middle and cheapest listing of each part,
on the vehicles in the training data and on the vehicles outside it.

Inputs:
- results/september_live_validation/september_predictions.csv. `baseline_eur`
  there is the thesis heuristic (training + validation subcategory median).

Output:
- results/september_live_validation/september_error_by_listing.png and .pdf

How to run:
  .venv/bin/python scripts/plot_september_error_by_listing.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "september_live_validation"
SOURCE = RESULTS / "september_predictions.csv"

# Same page width and ink as the Figure 10 / 11 script, so the three figures match.
TEXT_WIDTH_INCHES = 6.3
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8a87", "#e4e3df", "#ffffff"
RF_COLOUR, HEURISTIC_COLOUR = "#2a78d6", "#a7a5a0"

GROUPS = [("dearest", "Most\nexpensive"), ("middle", "Middle"), ("cheapest", "Cheapest"), ("all", "All\nlistings")]
PANELS = [(1, "Included in the training data"), (2, "Not included in the training data")]


def load():
    listings = pd.read_csv(SOURCE)
    listings["rf_error"] = (listings.predicted_eur - listings.asking_price_eur).abs()
    listings["heuristic_error"] = (listings.baseline_eur - listings.asking_price_eur).abs()
    return listings


def medians(rows):
    table = rows.groupby("tier")[["rf_error", "heuristic_error"]].median()
    table.loc["all"] = rows[["rf_error", "heuristic_error"]].median()
    return table


def main():
    listings = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 3.0), sharey=True,
                                facecolor=SURFACE)
    width, gap = 0.36, 0.02  # the gap leaves a thin surface line between touching bars
    for panel, (round_number, title) in zip(axes, PANELS):
        rows = listings[listings["round"] == round_number]
        table = medians(rows)
        panel.set_facecolor(SURFACE)
        positions = [index + (0.35 if key == "all" else 0) for index, (key, _) in enumerate(GROUPS)]
        for offset, column, colour in [(-1, "rf_error", RF_COLOUR), (1, "heuristic_error", HEURISTIC_COLOUR)]:
            for x, (key, _) in zip(positions, GROUPS):
                value = table.loc[key, column]
                centre = x + offset * (width + gap) / 2
                panel.bar(centre, value, width=width, color=colour, zorder=3)
                panel.text(centre, value + 2, f"{value:.0f}", ha="center", va="bottom",
                           fontsize=6.8, color=INK2)
        panel.set_xticks(positions, [label for _, label in GROUPS], fontsize=7.5, color=INK2)
        panel.set_title(f"{title} (n = {len(rows)})", fontsize=8.5, color=INK, pad=6)
        panel.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        panel.tick_params(axis="both", length=0, labelsize=7.5, colors=INK2)
        for side in ("top", "right", "left"):
            panel.spines[side].set_visible(False)
        panel.axvline((positions[2] + positions[3]) / 2, color=GRID, linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Median absolute error, EUR", fontsize=8, color=INK2)
    axes[0].set_ylim(0, 125)

    handles = [plt.Rectangle((0, 0), 1, 1, color=RF_COLOUR), plt.Rectangle((0, 0), 1, 1, color=HEURISTIC_COLOUR)]
    figure.legend(handles, ["Random Forest", "Subcategory-median heuristic"], loc="upper center",
                  ncol=2, frameon=False, fontsize=7.8, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    # No in-image title: in the thesis the caption below the figure carries it.
    figure.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.2)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "september_error_by_listing.png"
    figure.savefig(out, dpi=300, facecolor=SURFACE)
    figure.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"saved {out}")
    for round_number, title in PANELS:
        print(title, medians(listings[listings["round"] == round_number]).round(2).to_dict())


if __name__ == "__main__":
    main()
