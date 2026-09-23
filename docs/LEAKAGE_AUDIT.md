# Feature leakage audit

One row per candidate column: where it comes from, whether it exists when a new
listing is priced, its leakage risk, and whether the frozen models used it.

- Table: [`artifacts/leakage_audit/feature_leakage_audit.csv`](../artifacts/leakage_audit/feature_leakage_audit.csv)
- Built by: `.venv/bin/python scripts/build_leakage_audit.py` (read-only)

The table is generated from the feature constants in `src/tree_modeling.py` and
the frozen tuning summaries, so it records what the tuning actually did rather
than what it was meant to do. Every risk tier cites its basis in the `basis`
column; where no record explains a choice, it says "no recorded decision".
The script refuses to write the table if a column is unclassified or if the
frozen Random Forest uses a forbidden or high-risk feature.

## Summary

| Risk | Candidates | RF winner uses | Ridge finalist uses |
| --- | ---: | ---: | ---: |
| forbidden (target, `product_id`) | 2 | 0 | 0 |
| high (whole-window listing history) | 8 | 0 | 0 |
| medium (dates, OEM, history so far) | 11 | 0 | 2 |
| none (listing attributes, registry, pipeline flags) | 67 | 61 | 63 |
| **total** | **88** | **61** | **65** |

## Findings

1. **The frozen Random Forest uses no feature above "none".** Its 61 features are
   listing attributes and Traficom registry aggregates. OEM number, all dates
   and all listing-history columns are excluded.
2. **The Ridge finalist used two medium-risk history columns**,
   `observations_so_far` and `days_since_first_seen_so_far`. They sit in no
   exclusion set and no record explains why. They are computed from earlier
   snapshots only, so they do not see the future, but they do not exist for a
   part listed for the first time. Ridge was never scored on the holdout, so the
   holdout result does not depend on them; the Ridge cross-validation numbers
   do, and should be read with that in mind.
3. **The registry block adds no information beyond `model`.** All 49 registry
   columns are constant within a vehicle model, so each holds at most three
   distinct values across the dataset (41 hold exactly three). The registry
   ablation in `artifacts/registry_ablation/` measures the same thing from the
   model side.
4. **`repair_status` is constant** (`original_valid` on every row) and so
   carries nothing, although it is in the winner's feature list.
5. **Three offset columns named in the code never reached the data**
   (`first_seen_day_offset`, `last_seen_day_offset`,
   `listing_midpoint_day_offset`). They are listed so the audit shows they could
   not have been used.

Split-level leakage (repeated listings and comparable part identities crossing
splits) is handled by the connected-component split, not by feature choice. See
[STRICT_EVALUATION_PROTOCOL.md](STRICT_EVALUATION_PROTOCOL.md).
