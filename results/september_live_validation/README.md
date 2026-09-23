# September live validation

Manual validation study for the thesis: current varaosahaku.fi prices scored
against the frozen random forest, with the per-part median as the comparator.
Nothing here refits, retunes or reselects anything.

## What produced these files

```
scripts/parse_september_listings.py   saved HTML pages  ->  listing table (CSV)
scripts/score_september_validation.py listing table     ->  metrics report
scripts/plot_september_validation.py  listing tables    ->  the figure
```

Run them with the repo's own interpreter. The system `python3` carries an
x86_64 pandas on this arm64 Mac and cannot import it; `.venv` also holds
sklearn 1.7.2, the version the model bundle was pickled with.

```
.venv/bin/python scripts/parse_september_listings.py "<folder of .html>" <out.csv>
.venv/bin/python scripts/score_september_validation.py <out.csv> <brand> <model> [generation regex]
.venv/bin/python scripts/plot_september_validation.py 1   # or 2 for Round 2
```

## Where the data lives, and why it is not here

The saved pages and the parsed listing tables stay outside this repository, in
`~/Desktop/validation dataset/`. They are raw scraped rows carrying seller
names and OEM numbers, and this repository is public. `.gitignore` excludes
`*.csv` under this directory. Only aggregates are committed.

Collection: 33 cells (11 parts x 3 vehicles), 2,584 listings, saved by hand on
2026-09-17 and 2026-09-22. Automated fetching is not an option: the site's
robots.txt is `User-agent: * / Disallow: /`.

## Files

| File | What it is |
|---|---|
| `september_validation.pdf` | the figure, vector, for the thesis document |
| `september_validation.png` | the same figure at 300 dpi, for slides |
| `figure_caption.md` | draft caption and the notes that must travel with it |
| `corolla_scoring_report.txt` | full metric set, Toyota Corolla |
| `golf_scoring_report.txt` | full metric set, VW Golf |
| `octavia_scoring_report.txt` | full metric set, Skoda Octavia |
| `september_validation_round2.pdf` / `.png` | the same figure for Round 2 |
| `ford_scoring_report.txt` | Round 2, Ford Focus (`'^FORD FOCUS \d'`) |
| `nissan_scoring_report.txt` | Round 2, Nissan Qashqai (`'^NISSAN QASHQAI \d'`, Qashqai+2 dropped: its own registry family) |
| `volvo_scoring_report.txt` | Round 2, Volvo V70 (`'^VOLVO V70 \d'`, the mixed `S70/V70/XC 97-00` label dropped) |

## Two conventions that change the numbers

Both were recovered by reproducing the 2026-09-18 Corolla result and are fixed
in code rather than left to be rediscovered:

1. **The baseline is the vehicle's own subcategory median**, not the all-brand
   one. All-brand gives 86.0% MdAPE on the Corolla matched draw where the
   recorded figure is 58.0%.
2. **Scoring runs without the listing quality grade.** Including it moves the
   Corolla headline from 32.77% to 33.02%. The grade is still parsed into the
   listing table; it carries 0.2% of the model's SHAP importance.

## Round 2, vehicles February never saw

Ford Focus, Nissan Qashqai and Volvo V70, 9 parts each. Three things differ
from Round 1 and are handled in the scorer:

1. **Registry block.** There is no February row to copy it from, so it comes
   from `datasets/traficom_outputs/model_summary.csv` and `brand_summary.csv`,
   brand matched exactly. The same lookup run for the three trained vehicles
   reproduces February's 49 values exactly.
2. **Pooled pages.** The site lists Focus with C-Max and Grand C-Max, and V70
   with S70 and V70 XC. The pooled page is saved as is and the scorer's
   generation regex keeps the target car, printing what it kept and dropped.
   Ford: `'^FORD FOCUS \d'`, which also drops `FORD FOCUS C-MAX I`.
3. **Baseline.** No per-vehicle median exists, so the comparator is the
   all-brand subcategory median. That is a weaker baseline than Round 1's, so
   the Round 2 gap to it is not comparable with a Round 1 gap. The comparison
   that is like for like is the same model and the same draw on trained
   against unseen vehicles.

Cells saved over several pages are de-duplicated on `product_id`.

## Reproduction gate

Any change to the parser or the scorer must still reproduce the recorded
Corolla figures before other vehicles are scored:

```
868 parsed / 157 February overlaps dropped / 711 unseen / 254 matched draw
155 covered cells / 99 uncovered
matched-draw MdAPE 32.77%, MdAE 30.33, median bias +13.88
baseline 58.00% / 44.20 / +35.80
```
