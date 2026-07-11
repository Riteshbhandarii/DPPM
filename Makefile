# DPPM — reproducibility entry points.
# Override the interpreter if you are not in an activated venv, e.g.:
#   make verify PYTHON=.venv/bin/python
PYTHON ?= python

.PHONY: help verify test analyses tuning

help:  ## List targets
	@echo "DPPM make targets:"
	@echo "  make verify    - regenerate the strict split into a temp dir and confirm it"
	@echo "                   reproduces the frozen split byte-identically (seconds, safe)"
	@echo "  make test      - run the test suite (pytest)"
	@echo "  make analyses  - re-run the holdout-safe descriptive analyses (SHAP, registry"
	@echo "                   ablation, learning curve). Several minutes; overwrites only"
	@echo "                   their own artifacts/ dirs, never the frozen split/holdout."
	@echo "  make tuning    - re-run strict model selection. GUARDED: ~37 min AND overwrites"
	@echo "                   the FROZEN selection artifacts. Requires FORCE=1 to run."
	@echo ""
	@echo "  The final holdout is frozen and has no target on purpose — it must never re-run."

verify:  ## Safe reproducibility check (no frozen artifact is modified)
	$(PYTHON) scripts/verify_reproducibility.py

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

analyses:  ## Re-run holdout-safe descriptive analyses (minutes)
	@echo ">> Re-running descriptive analyses (holdout-safe, several minutes)..."
	$(PYTHON) scripts/run_strict_shap.py
	$(PYTHON) scripts/registry_ablation.py
	$(PYTHON) scripts/learning_curve.py

tuning:  ## Re-run strict selection — GUARDED (overwrites frozen artifacts)
	@if [ "$(FORCE)" != "1" ]; then \
		echo "Refusing: this re-runs ~37 min of tuning and OVERWRITES the frozen selection"; \
		echo "artifacts (artifacts/strict_model_tuning/). Run 'make tuning FORCE=1' only if"; \
		echo "you intend to regenerate them."; \
		exit 1; \
	fi
	$(PYTHON) scripts/run_strict_model_tuning.py --model random_forest
	$(PYTHON) scripts/run_strict_model_tuning.py --model ridge
