# AGENTS.md

# DPPM - Agent Operating Instructions

This repository contains the DPPM (Dismantler Price Prediction Model) thesis project.

The primary objective is not maximizing benchmark scores. The primary objective is producing a defensible, reproducible, academically credible Bachelor's thesis and a professional portfolio project.

Agents working in this repository must prioritize clarity, reproducibility, transparency, and thesis defensibility over novelty or metric chasing.

---

# Project Context

DPPM predicts listing prices for used automotive spare parts in Finland using:

- Varaosahaku marketplace listings
- Traficom vehicle registry data
- Supervised machine learning models
- Explainability analysis (SHAP)
- Streamlit and FastAPI prototype interfaces

This project is a proof-of-concept decision-support system.

It is NOT a production pricing authority.

Predictions are intended to support human pricing decisions rather than replace them.

---

# User Working Preferences

The repository owner prefers:

- Practical research code
- Readable code
- Professional code
- Well-documented code
- Conservative evaluation practices
- Reproducible results

The repository owner may return after weeks or months away from the project.

Code should therefore be understandable without needing to reconstruct historical decisions from memory.

Always optimize for future readability.

When Codex creates a pull request, the PR title should describe the work using
the appropriate change type, such as `Fix`, `Add`, `Improve`, `Document`, or
`Refactor`. Do not prefix PR titles with `Codex` or `[codex]`.

---

# Scope Control

IMPORTANT:

Do not perform major work that was not explicitly requested.

Agents must NOT:

- launch large new experiments without approval
- redesign project architecture without approval
- refactor unrelated files
- rename files unnecessarily
- move project structure unnecessarily
- delete historical artifacts without approval
- change thesis methodology without approval
- introduce "nice-to-have" features unrelated to the current task

If additional work appears valuable:

1. Explain why.
2. Recommend it.
3. Wait for approval.

Default behavior should be focused execution.

---

# Coding Philosophy

Prefer:

- clarity over cleverness
- readability over compactness
- maintainability over micro-optimizations

Avoid:

- unnecessary abstractions
- deeply nested logic
- overly clever one-liners
- premature optimization

Research code should remain understandable to both technical and academic readers.

---

# Commenting Standards

Comments should explain:

- WHY something is done
- assumptions
- thesis implications
- evaluation decisions
- leakage prevention decisions
- modeling decisions

Avoid comments that simply repeat the code.

Bad:

```python
# Create dataframe
df = pd.DataFrame(...)
```

Good:

```python
# Registry features are merged after marketplace cleaning
# to avoid introducing preprocessing leakage before splitting.
```

---

# File Documentation Requirements

Complex scripts should contain a top-level docstring describing:

- purpose
- inputs
- outputs
- assumptions
- how to run the script

Example:

```python
"""
Purpose:
Train strict Random Forest models using leakage-controlled features.

Inputs:
- train_grouped.csv
- validation_grouped.csv

Outputs:
- trained model artifact
- evaluation metrics

Notes:
Uses strict feature policy defined in AGENTS.md.
"""
```

---

# Reproducibility Requirements

Final thesis results must be reproducible.

Prefer:

- scripts
- versioned artifacts
- saved metrics

Do not rely exclusively on notebooks for final thesis results.

Notebooks are acceptable for:

- exploration
- diagnostics
- visualization

Final thesis claims should be traceable to reproducible scripts and saved artifacts.

---

# Evaluation Philosophy

The objective is not to maximize R².

The objective is to estimate realistic generalization performance.

Primary metrics:

1. MAE
2. Median Absolute Error
3. RMSE
4. R²

MAE should generally be treated as the primary practical metric because it is directly interpretable in euros.

High R² values should always be interpreted cautiously.

---

# Leakage Policy

Data leakage prevention is a core project requirement.

Agents must evaluate whether a feature would realistically be available at prediction time.

If uncertain:

Assume the feature is leakage-sensitive until proven otherwise.

---

# Forbidden Features

The following must never be used directly as predictive features:

- target price
- transformed target price
- future-derived target information
- product_id

---

# Leakage-Sensitive Features

Treat the following with caution:

- OEM numbers
- listing history variables
- first_seen_date
- last_seen_date
- observed_span_days
- price history variables
- price change variables
- scrape timing variables

These features require explicit justification before inclusion.

---

# Compatibility Bias Policy

A central risk in this project is compatibility leakage.

Examples include:

- identical spare-part families
- highly similar OEM mappings
- near-duplicate compatibility structures
- repeated marketplace identities

Strong performance may reflect compatibility similarity rather than true generalization.

Agents must discuss compatibility leakage whenever new evaluation results are produced.

---

# Accepted Evaluation Hierarchy

Evaluation confidence generally increases in this order:

1. Fixed validation split
2. Product-ID grouped validation
3. Product-ID grouped cross-validation
4. Strict part-identity grouped evaluation
5. Compatibility-family robustness checks

The strictest available evaluation should be considered the primary thesis result.

---

# Feature Categories

## Safe Features

Examples:

- brand
- model
- category
- subcategory
- quality grade
- mileage
- compatibility year range

## Context Features

Examples:

- Traficom aggregates
- population statistics
- market-share estimates
- lifecycle variables

These may be used but should not be interpreted as causal evidence.

## Risk Features

Examples:

- OEM numbers
- listing-history variables
- price-history variables
- future-aware variables

Require explicit justification.

---

# SHAP Interpretation Rules

SHAP explains model behavior.

SHAP does NOT prove:

- causality
- economic mechanisms
- real-world market effects

All SHAP interpretation must be framed as model explanation.

---

# Artifact Management

Every major experiment should save:

- metrics
- configuration
- feature set description
- evaluation strategy

Artifacts should be preserved whenever practical.

Do not overwrite important historical results without documenting why.

---

# Benchmark Preservation

Historical results should not be deleted simply because newer evaluations are stricter.

Instead:

- preserve them
- label them appropriately
- explain their limitations

The optimistic benchmark remains useful as a comparison point.

---

# Thesis Writing Rules

Always prefer:

- transparency
- honesty
- reproducibility
- defensibility

Never:

- fabricate results
- fabricate citations
- exaggerate model capabilities
- hide evaluation limitations

Important limitations should be discussed openly.

---

# Known Thesis Risks

Agents should remain aware of:

- repeated listings
- compatibility leakage
- identity leakage
- duplicate-like observations
- short scrape window
- limited vehicle scope
- marketplace-specific behavior
- subgroup performance variation

These limitations are part of the thesis narrative and should not be ignored.

---

# Issue Priority

When deciding what to work on next:

1. Fix strict-pipeline issues
2. Define and freeze evaluation protocol
3. Validate strict split artifacts
4. Rerun strict model selection
5. Train final strict model
6. Preserve optimistic benchmark
7. Run compatibility robustness diagnostics
8. Run subgroup error analysis
9. Complete leakage audit
10. Perform secondary diagnostics

---

# Testing

When modifying code:

- run relevant tests
- verify outputs
- preserve reproducibility

Avoid making changes that cannot be verified.

---

# Final Principle

Do not be clever.

Be understandable.

Preserve the research trail.

If changing something important:

- explain what changed
- explain why it changed
- explain how it affects the thesis

Future readers should always be able to understand the reasoning behind a decision.
