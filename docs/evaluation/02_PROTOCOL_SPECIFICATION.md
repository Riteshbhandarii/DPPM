# Evaluation Protocol Specification

## Input Dataset

```text
datasets/cleaned/clean_master_dataset.csv
```

## Required Columns

```text
product_id
part_name
brand
model
year_start
year_end
price
```

Optional distribution-check columns:

```text
category
subcategory
```

## Canonicalization

Apply the following to every identity field:

1. Convert missing values to `__missing__`.
2. Convert value to string.
3. Apply Unicode normalization with `NFKC`.
4. Apply `casefold()`.
5. Trim leading and trailing whitespace.
6. Collapse repeated internal whitespace to a single space.
7. Replace empty strings with `__missing__`.
8. Do not use fuzzy matching.

## Identity Definition

The final strict identity key is:

```text
canonical_part_name
canonical_brand
canonical_model
canonical_year_start
canonical_year_end
```

Combined as:

```text
canonical(part_name, brand, model, year_start, year_end)
```

## Graph Construction

Each row is a node.

Create an undirected edge between rows when either condition is true:

```text
same canonical product_id
same final identity key
```

## Connected Components

Compute connected components over the graph.

Every connected component must be assigned to exactly one split.

## Split Ratios

Target row proportions:

```text
train: 70%
validation: 15%
test: 15%
```

## Diagnostic Seed

The selected diagnostic seed is:

```text
32
```

The seed was selected by the candidate balance diagnostic and should be used when regenerating the candidate final split, unless a later documented rerun changes the decision.

## Leakage Assertions

Every generated split must pass:

```text
no product_id appears in more than one split
no identity key appears in more than one split
no connected component appears in more than one split
```

Any assertion failure invalidates the split.

## Expected Diagnostic Outputs

The diagnostic script writes:

```text
results/candidate_component_split_balance.md
results/candidate_component_split_balance/
```

Expected diagnostic artifacts include:

```text
diagnostic_component_rows.csv
diagnostic_component_summary.csv
diagnostic_recommended_seed_component_assignments.csv
diagnostic_recommended_seed_row_assignments.csv
simulation_metrics.csv
simulation_price_distributions.csv
simulation_leakage_checks.csv
simulation_distribution_diagnostics.csv
split_size_summary_across_simulations.csv
```

These are diagnostic artifacts only.

## Non-Outputs

The diagnostic script must not create:

```text
train.csv
validation.csv
test.csv
```

Final split generation should be done only by a future explicitly approved split-generation script or command.
