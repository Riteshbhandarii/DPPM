# Documentation

| Document | What it covers |
| --- | --- |
| [Results](RESULTS.md) | Test-set result, errors by brand and category, September live validation |
| [Architecture](ARCHITECTURE.md) | Data flow, components, evaluation design |
| [Pipeline](PIPELINE.md) | Run order, frozen artifacts, how to reproduce |
| [Model comparison](STRICT_MODEL_COMPARISON.md) | Model selection, test-set result, baseline comparison, SHAP |
| [Evaluation protocol](STRICT_EVALUATION_PROTOCOL.md) | The connected-component split |
| [Leakage audit](LEAKAGE_AUDIT.md) | Every candidate feature and its leakage risk |
| [Design decisions](DESIGN_DECISIONS.md) | Dated decision log |
| [Development](DEVELOPMENT.md) | Setup, repository layout, data files, CI |
| [Roadmap](THESIS_ROADMAP.md) | Build status and issue mapping |

The project uses two splits. The **product_id split** keeps repeated snapshots of
one listing together; its numbers are optimistic and kept only for comparison.
The **connected-component split** also keeps comparable parts together and is
the one every reported result comes from. In code and folder names it appears
as `strict` (`datasets/splits_strict/`, `test_strict.csv`).
