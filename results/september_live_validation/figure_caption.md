# Draft caption

Insert the figure at text width and place this beneath it. Renumber to follow
Figure 9 (the prototype demonstration in §4.5).

---

**Figure 10.** Observed and predicted asking price for the highest-priced listing
of each spare part on each vehicle (n = 33). Solid markers show the observed
listing price on varaosahaku.fi in September 2026; open markers show the
prediction of the frozen random forest, which was trained on February 2026 data
and was not refitted. Colour identifies the vehicle. The horizontal axis is
logarithmic. Errors are reported in euros; a negative value indicates that the
model predicted below the observed price. Listings already present in the
training data were excluded. Only the highest-priced listing in each part and
vehicle group is shown, because this is the segment the training sample covers;
accuracy falls sharply on the cheaper listings within the same groups, which is
reported in Table X.

---

## Notes for the surrounding text, not the caption

- The figure is the dearest tier only. The companion number for the other two
  tiers must appear nearby or the figure reads as a selected best case.
- "Within 10%" is the label used on the error column for cases where the absolute
  error is under a tenth of the observed price.
- The three vehicles are the same three the model was trained on. The
  out-of-distribution test (Mercedes C-Class, Ford Focus, Nissan Qashqai) is
  separate and is not in this figure.
