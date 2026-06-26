# Evaluation Protocol Timeline

## 1. Random or Fixed Split

Early model development used simple random or fixed split logic.

What was learned:

- Useful for fast modeling iteration.
- Not defensible as final thesis evidence because repeated listing snapshots could cross splits.

## 2. Product-ID Grouped Benchmark

The next protocol grouped observations by `product_id`.

What was learned:

- This prevents repeated-listing leakage.
- It remains a useful optimistic benchmark.
- It does not prevent identity leakage across different product IDs.

## 3. Candidate Identity Analyses

Candidate identity keys were compared:

```text
product_id
part_name + brand + model
part_name + brand + model + OEM
part_name + brand + model + year_start + year_end
```

What was learned:

- Broad identities reduce singleton rates but can be too coarse.
- OEM produces stricter groups but fragments heavily.
- Compatibility years give a better balance between identity strictness and group usability.

## 4. Fragmentation Analysis

The analysis measured how many broad identities were split when adding OEM or compatibility years.

What was learned:

- OEM split 348 base groups and created 309 new singleton groups.
- Compatibility years split 270 base groups and created 93 new singleton groups.
- OEM fragmentation was too strong for the final thesis identity rule.

## 5. Price Variability Analysis

Within-group target-price dispersion was measured.

What was learned:

- `part_name + brand + model` had average within-group variance of 1,443.55.
- `part_name + brand + model + year_start + year_end` reduced average variance to 643.38.
- `part_name + brand + model + OEM` reduced variance further, but with too much fragmentation.

## 6. Connected Components

Rows were modeled as graph nodes with edges for shared `product_id` or shared identity.

What was learned:

- Connected components prevent indirect leakage paths.
- The final candidate graph produced 886 components.
- The largest component had 71 rows.

## 7. Balance Diagnostics

Component-level train/validation/test assignments were simulated over 100 seeds.

What was learned:

- Diagnostic seed 32 produced row proportions of 70.05% / 14.97% / 14.98%.
- Product-id, identity-key, and connected-component leakage checks all passed.
- Category-level distribution differences remain worth reviewing before freezing final split files.

## 8. Final Protocol

The selected final protocol is:

```text
connected-component split
edges: same product_id OR same canonical(part_name, brand, model, year_start, year_end)
target proportions: 70 / 15 / 15
diagnostic seed: 32
```

This becomes the primary academic evaluation protocol. The product-id grouped evaluation remains an optimistic historical benchmark.
