# Strict Evaluation Protocol

## Status

The strict evaluation protocol is not finalized yet.

The old strict setup is useful historical context, but it should not be automatically accepted as the final thesis protocol.

## Historical Strict Setup

The previous strict grouped-CV setup used:

```text
part_name + brand + model + oem_number
```

The generated group column was:

```text
part_identity_group
```

This setup was designed to prevent exact comparable part identities from crossing folds. However, later inspection raised questions about whether `oem_number` is reliable enough to include in the final strict key.

## OEM Number Concerns

Current evidence raises methodological concerns about using `oem_number` in the final strict identity rule:

- The cleaned dataset has only 54 normalized OEM values.
- OEM values are reused across unrelated part identities.
- Adding OEM fragments identity groups substantially.
- OEM appears more useful as a noisy listing attribute than as a trustworthy strict identity boundary.

Because of this, OEM inclusion or exclusion remains a methodological decision to finalize before generating the new strict split.

## Current Candidate Strict Identity

Current candidate rule:

```text
canonical_part_name + canonical_brand + canonical_model
```

This candidate intentionally excludes `oem_number` unless further comparison shows that OEM improves identity validity rather than fragmenting it.

## Canonicalization Requirements

Strict identity generation should canonicalize text before grouping.

Required canonicalization:

- Casefold/lowercase text.
- Apply Unicode normalization.
- Trim leading and trailing whitespace.
- Collapse repeated internal whitespace.
- Normalize punctuation spacing around separators such as hyphens, commas, and parentheses.
- Use a documented missing-value sentinel if missing values appear in future data.

The previous implementation only stripped whitespace, lowercased strings, and replaced missing/empty values with `__missing__`. That is not strong enough for final split generation because repeated listings can differ by small punctuation and spacing variants.

## Split Construction Principle

The new strict split should be generated directly from:

```text
datasets/cleaned/clean_master_dataset.csv
```

It should not be generated from the old grouped split files.

## Product-ID and Identity Leakage

A strict split must prevent both:

1. Repeated-listing leakage through `product_id`
2. Comparable-identity leakage through the strict identity key

A graph or connected-component approach should be considered:

- Create links between rows sharing the same `product_id`.
- Create links between rows sharing the same strict identity.
- Split connected components rather than individual rows.

This would ensure that repeated observations and comparable identities cannot cross train, validation, or test boundaries.

## Next Task

Compare candidate identity rules before generating the new split.

Candidate rules to compare:

| Candidate | Status |
| --- | --- |
| `canonical_part_name + canonical_brand + canonical_model` | Current preferred candidate |
| `canonical_part_name + canonical_brand + canonical_model + canonical_oem_number` | Historical setup; under review |
| Variants including category/subcategory | Only if justified by identity validity checks |

## Evidence Boundary

Old strict results are historical/contextual only. Final thesis evidence must come from the new documented strict split and future final reruns.
