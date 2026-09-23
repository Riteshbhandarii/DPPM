# DPPM

**Dismantler Price Prediction Model.** Predicts asking prices for used car spare
parts on Varaosahaku.fi from listing data and Finnish vehicle-registry
(Traficom) summaries. Built as a proof of concept for price review, not as an
automated pricing system.

## Results

**Strict holdout (1,696 listings, run once).** A tuned Random Forest scores
MAE 69.46 EUR and median error 29.37 EUR. A per-subcategory median lookup scores
MAE 66.15 EUR and median error 15.32 EUR on the same rows, so the model ties the
lookup on MAE and loses on median error.

**Live validation on September 2026 listings.** The frozen model, not refitted,
scored on current listings. Each row is one part on one vehicle: the dearest,
the middle and the cheapest listing, observed price against prediction.

Vehicles the model was trained on (Corolla, Golf, Octavia):

![September validation, trained vehicles](results/september_live_validation/september_validation.png)

Vehicles the model never saw (Focus, Qashqai, V70):

![September validation, unseen vehicles](results/september_live_validation/september_validation_round2.png)

Numbers behind both figures: [docs/RESULTS.md](docs/RESULTS.md).

## Quickstart

Python 3.12 (`.python-version`), dependencies pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py        # demo UI
uvicorn app.fastapi_app:app --reload      # API
```

Check that the frozen split and holdout result still reproduce (read-only):

```bash
make verify PYTHON=.venv/bin/python
make test PYTHON=.venv/bin/python
```

## Documentation

| Document | What it covers |
| --- | --- |
| [Results](docs/RESULTS.md) | Holdout, subgroup errors, September live validation |
| [Architecture](docs/ARCHITECTURE.md) | Data flow, components, evaluation design |
| [Pipeline](docs/PIPELINE.md) | Run order, frozen artifacts, how to reproduce |
| [Strict model comparison](docs/STRICT_MODEL_COMPARISON.md) | Model selection, holdout, baseline comparison, SHAP |
| [Evaluation protocol](docs/STRICT_EVALUATION_PROTOCOL.md) | The connected-component split |
| [Leakage audit](docs/LEAKAGE_AUDIT.md) | Every candidate feature and its leakage risk |
| [Design decisions](docs/DESIGN_DECISIONS.md) | Dated decision log |
| [Development](docs/DEVELOPMENT.md) | Repository layout, data files, CI |
| [Roadmap](docs/THESIS_ROADMAP.md) | Build status and issue mapping |

## License

See [LICENSE](LICENSE).
