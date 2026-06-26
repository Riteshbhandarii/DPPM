# Evaluation Protocol Reproducibility

This document explains how to reproduce the protocol-selection study from the cleaned dataset.

## Starting Point

Run all commands from the repository root.

Input dataset:

```text
datasets/cleaned/clean_master_dataset.csv
```

The analyses are structural diagnostics. They do not train machine-learning models.

## 1. Grouping Strategy Analysis

Command:

```bash
python3 scripts/analyze_grouping_strategies.py
```

Expected outputs:

```text
results/grouping_strategy_analysis.md
results/grouping_strategy_analysis/
```

Purpose:

Compare unique group counts, group-size distributions, singleton rates, histograms, and largest groups.

## 2. Fragmentation Analysis

Command:

```bash
python3 scripts/analyze_grouping_fragmentation.py
```

Expected outputs:

```text
results/grouping_strategy_fragmentation.md
results/grouping_strategy_fragmentation/
```

Purpose:

Measure whether OEM and compatibility years create meaningful refinements or excessive fragmentation.

## 3. Product-ID Identity Leakage Analysis

Command:

```bash
python3 scripts/analyze_product_id_identity_leakage.py
```

Expected outputs:

```text
results/product_id_identity_leakage.md
results/product_id_identity_leakage/
```

Purpose:

Estimate how much identity leakage can remain when splitting only by `product_id`.

## 4. Group Price Variability Analysis

Command:

```bash
python3 scripts/analyze_group_price_variability.py
```

Expected outputs:

```text
results/group_price_variability.md
results/group_price_variability/
```

Purpose:

Measure whether candidate identities group economically similar listings by comparing within-group target-price dispersion.

## 5. Connected Component Analysis

Command:

```bash
python3 scripts/analyze_identity_graph_components.py
```

Expected outputs:

```text
results/identity_graph_components.md
results/identity_graph_components/
```

Purpose:

Analyze connected components created by product-id and candidate identity edges.

## 6. Candidate Component Split Balance Analysis

Command:

```bash
python3 scripts/analyze_candidate_component_split_balance.py
```

Expected outputs:

```text
results/candidate_component_split_balance.md
results/candidate_component_split_balance/
```

Purpose:

Simulate component-level 70 / 15 / 15 assignments over at least 100 seeds and verify balance and leakage constraints.

## Validation

Compile every diagnostic script:

```bash
python3 -m py_compile scripts/analyze_grouping_strategies.py
python3 -m py_compile scripts/analyze_grouping_fragmentation.py
python3 -m py_compile scripts/analyze_product_id_identity_leakage.py
python3 -m py_compile scripts/analyze_group_price_variability.py
python3 -m py_compile scripts/analyze_identity_graph_components.py
python3 -m py_compile scripts/analyze_candidate_component_split_balance.py
```

## Dependencies Between Analyses

The scripts can be run independently from the cleaned dataset.

The documentation uses the following interpretation order:

1. Group-size statistics define candidate feasibility.
2. Fragmentation diagnostics compare identity refinements.
3. Product-id leakage diagnostics show why product-id grouping is not enough.
4. Price variability diagnostics evaluate economic coherence.
5. Component diagnostics show the graph split unit.
6. Balance diagnostics show whether the final protocol can produce usable partitions.

## Regenerating Final Evaluation Inputs

This study does not generate final model evaluation splits.

After the protocol is approved, a separate final split-generation script should implement [02_PROTOCOL_SPECIFICATION.md](02_PROTOCOL_SPECIFICATION.md), use diagnostic seed `32`, and save explicit final train/validation/test files only after approval.
