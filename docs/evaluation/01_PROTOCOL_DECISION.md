# Evaluation Protocol Decision

## Background

DPPM originally used simpler evaluation splits while the modeling pipeline was still being developed. The earliest baseline work used ordinary random or fixed train/validation logic. That approach was useful for model development, but it was not sufficiently conservative for final thesis evidence because repeated marketplace observations can appear as near-identical rows.

The next evaluation step grouped rows by `product_id`. This solved a concrete listing-level leakage problem: snapshots of the same marketplace listing could no longer appear in both training and evaluation data. The existing product-id grouped split is therefore preserved as a valid optimistic operational benchmark.

Later diagnostics showed that product-id grouping does not prevent identity leakage. Different `product_id` values can still describe the same or highly comparable spare-part identity. A model evaluated only with product-id grouping can therefore benefit from seeing economically and structurally similar parts in both training and test data.

The final thesis protocol separates two risks:

- Listing-level leakage: repeated observations of the same listing cross splits.
- Identity leakage: comparable part identities cross splits even when listing IDs differ.

The final protocol addresses both risks by splitting connected components built from `product_id` and a strict part identity key.

## Candidate Strategies

### A. `product_id`

Advantages:

- Prevents repeated snapshots of the same listing from crossing splits.
- Preserves relatively small grouping units.
- Existing historical evaluations are already based on this principle.

Disadvantages:

- Does not prevent comparable part identities from crossing splits.
- Identity leakage diagnostics found that many canonical identities are represented by multiple product IDs.

Observed statistics:

- Unique groups: 2,619.
- Average group size: 4.32 rows.
- Median group size: 6 rows.
- Singleton groups: 834, or 31.84%.
- Largest group size: 6 rows.

Decision:

Accepted as an optimistic benchmark only. Rejected as the primary academic evaluation protocol because it controls listing-level leakage but not identity leakage.

### B. `part_name + brand + model`

Advantages:

- Directly targets comparable part identities.
- Produces larger identity groups, making identity leakage visible.
- Has the lowest singleton rate among the candidate identity keys.

Disadvantages:

- Too coarse for some economically distinct parts.
- Higher within-group price variation indicates that this key can combine observations that are not price-similar enough.

Observed statistics:

- Unique groups: 593.
- Average group size: 19.09 rows.
- Median group size: 18 rows.
- Singleton groups: 96, or 16.19%.
- Largest group size: 87 rows.
- Average within-group price variance: 1,443.55.
- Median within-group price variance: 21.51.

Decision:

Rejected as the final identity key because it is useful for identifying broad leakage risk but appears too coarse for final evaluation.

### C. `part_name + brand + model + OEM`

Advantages:

- Reduces typical within-group price dispersion.
- Gives a stricter identity definition than part name, brand, and model alone.

Disadvantages:

- OEM values appear noisy and reused.
- Adding OEM fragments broad identities aggressively.
- The higher singleton rate indicates that this key may be too strict or unstable for final thesis grouping.

Observed statistics:

- Unique groups: 1,624.
- Average group size: 6.97 rows.
- Median group size: 6 rows.
- Singleton groups: 405, or 24.94%.
- Adding OEM split 348 base identity groups and created 309 new singleton groups.
- Average within-group price variance: 418.37.
- Median within-group price variance: 0.20.

Decision:

Rejected as the final identity key. OEM improves apparent price homogeneity, but the fragmentation diagnostics indicate that it may act as a noisy fragmentation key rather than a reliable identity boundary.

### D. `part_name + brand + model + year_start + year_end`

Advantages:

- Captures compatibility-era distinctions without relying on noisy OEM mappings.
- Reduces price dispersion compared with the broad part identity.
- Fragments less aggressively than OEM.
- Aligns with the thesis concern that comparable compatibility families should not cross evaluation splits.

Disadvantages:

- Still creates more groups than the broad identity key.
- Some subgroup imbalance remains because connected components cannot be divided.

Observed statistics:

- Unique groups: 1,032.
- Average group size: 10.97 rows.
- Median group size: 6 rows.
- Singleton groups: 189, or 18.31%.
- Adding compatibility years split 270 base identity groups and created 93 new singleton groups.
- Average within-group price variance: 643.38.
- Median within-group price variance: 0.75.

Decision:

Accepted as the final strict identity key. It is the best compromise between leakage prevention, economic coherence, and avoiding excessive fragmentation.

## Connected Component Methodology

Each dataset row is treated as a graph node.

Edges are created when two rows share:

- the same `product_id`, or
- the same strict identity key.

Connected components are then split as indivisible units. This prevents indirect leakage paths where one row is linked to another through `product_id`, and that second row is linked to further rows through identity.

Example:

```text
row A -- same product_id -- row B -- same identity -- row C

All three rows belong to one connected component.
The component must stay in one split.
```

This approach prevents both repeated-listing leakage and comparable-identity leakage.

## Final Protocol

Input dataset:

```text
datasets/cleaned/clean_master_dataset.csv
```

Final identity:

```text
canonical(part_name, brand, model, year_start, year_end)
```

Graph edges:

```text
same product_id OR same final identity
```

Split unit:

```text
connected component
```

Split proportions:

```text
train: 70%
validation: 15%
test: 15%
```

Diagnostic seed:

```text
32
```

The diagnostic seed is not a final split artifact by itself. It is the seed selected by balance diagnostics as a reproducible candidate for future final split generation.

## Evidence Summary

| Analysis | Key evidence | Decision relevance |
| --- | --- | --- |
| Grouping strategy analysis | `part_name + brand + model + years` produced 1,032 groups and 18.31% singletons. | Candidate has manageable fragmentation. |
| Fragmentation analysis | OEM created 309 new singleton groups; compatibility years created 93. | OEM was rejected as too fragmenting. |
| Product-id leakage analysis | Under `part_name + brand + model`, 404 identities were represented by multiple product IDs. | Product-id-only splitting remained optimistic. |
| Price variability analysis | Broad identity average variance was 1,443.55; year identity average variance was 643.38. | Compatibility years improved economic coherence. |
| Connected component analysis | Final graph produced 886 components, median size 8, maximum size 71. | Component splitting is feasible. |
| Candidate split balance analysis | Seed 32 produced 70.05% / 14.97% / 14.98% rows with zero leakage. | Final protocol can produce balanced partitions. |

Detailed evidence is preserved in [03_ANALYSIS_INDEX.md](03_ANALYSIS_INDEX.md) and the referenced `results/` reports.

## Final Decision

The final evaluation protocol is a connected-component split using:

```text
product_id + canonical(part_name, brand, model, year_start, year_end)
```

This protocol is selected because it provides the best available compromise:

- It controls repeated-listing leakage through `product_id`.
- It controls comparable-identity leakage through the strict identity key.
- It avoids the strongest OEM fragmentation risk.
- It produces feasible 70 / 15 / 15 diagnostic partitions.
- It preserves the original product-id grouped benchmark as an optimistic comparison point.

## Thesis Implications

The original product-id grouped evaluation should remain in the thesis as an optimistic empirical benchmark. It is useful because it shows performance when repeated listing snapshots are controlled.

The connected-component evaluation becomes the primary academic evaluation because it is stricter and better aligned with the risk of identity leakage.

This change should be presented as methodological refinement based on evidence, not as correction of an error. The project evolved from practical model development toward a more defensible final thesis evaluation.

## Future Work

Future projects could improve identity definitions using:

- verified compatibility databases,
- verified OEM mappings,
- richer vehicle metadata,
- manufacturer-specific part families,
- longer marketplace observation windows.

These are future research directions. They are not required changes for the current thesis protocol.
