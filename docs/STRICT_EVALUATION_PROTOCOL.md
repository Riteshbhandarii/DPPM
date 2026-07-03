# Strict Evaluation Protocol

## Status

The strict evaluation protocol is finalized for the thesis methodology.

Canonical documentation now lives under [docs/evaluation/](evaluation/):

- [Protocol decision](evaluation/01_PROTOCOL_DECISION.md)
- [Technical specification](evaluation/02_PROTOCOL_SPECIFICATION.md)
- [Analysis index](evaluation/03_ANALYSIS_INDEX.md)
- [Reproducibility guide](evaluation/04_REPRODUCIBILITY.md)
- [Protocol timeline](evaluation/PROTOCOL_TIMELINE.md)

This file is retained as a short compatibility entry point for older documentation links.

## Final Protocol Summary

Input dataset:

```text
datasets/cleaned/clean_master_dataset.csv
```

Final strict identity:

```text
canonical(part_name, brand, model, year_start, year_end)
```

Graph edges:

```text
same product_id OR same final strict identity
```

Split unit:

```text
connected component
```

Target proportions:

```text
train: 70%
validation: 15%
test: 15%
```

Diagnostic seed:

```text
32
```

## Leakage Assertions

Every final split must pass:

- no `product_id` appears in more than one split,
- no strict identity key appears in more than one split,
- no connected component appears in more than one split.

## Evidence Boundary

The earlier product-id grouped evaluation remains an optimistic historical benchmark. The connected-component protocol is the primary academic evaluation protocol.

Detailed reasoning is in [01_PROTOCOL_DECISION.md](evaluation/01_PROTOCOL_DECISION.md).
