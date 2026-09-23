# Draft captions

Insert each figure at text width and place its caption beneath it. Renumber to
follow Figure 9 (the prototype demonstration in §4.5). The wording is a factual
draft; the final caption text is the author's.

---

**Figure 10.** Observed and predicted asking price for three listings of each
spare part on each vehicle the model was trained on: the dearest, the middle
and the cheapest listing of every part and vehicle group (11 parts x 3
vehicles, n = 99). Each row is one part on one vehicle and is the same row in
all three panels. Solid markers show the observed listing price on
varaosahaku.fi in September 2026; open markers show the prediction of the
frozen random forest, which was trained on February 2026 data and was not
refitted. Colour identifies the vehicle. The horizontal axis is logarithmic.
Listings already present in the training data were excluded.

**Figure 11.** The same three-panel comparison for three vehicles absent from
the February training data, Ford Focus, Nissan Qashqai and Volvo V70 (9 parts
x 3 vehicles, n = 81). Drawn and read as Figure 10.

---

## Notes for the surrounding text, not the caption

- The figure shows all three tiers. Error figures per tier (MdAPE for the model
  and the baseline) are in each vehicle's scoring report under "Reported draw"
  and go in the tier table beside the figure, not in the figure.
- The per-part median baseline is not drawn in either figure. Its comparison
  with the model goes in the tier table beside the figure.
- Round 2's baseline is the all-brand per-part median, a weaker comparator than
  Round 1's per-vehicle median, so a Round 2 gap to the baseline is not
  comparable with a Round 1 gap. See README.md.
