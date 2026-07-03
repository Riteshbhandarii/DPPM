# Evaluation Analysis Index

This index links every protocol-selection analysis to its script, report, artifacts, and main conclusion.

| Analysis | Purpose | Script | Report | CSV artifacts | Key conclusion |
| --- | --- | --- | --- | --- | --- |
| Grouping strategy analysis | Compare candidate group sizes and singleton rates. | [scripts/analyze_grouping_strategies.py](../../scripts/analyze_grouping_strategies.py) | [results/grouping_strategy_analysis.md](../../results/grouping_strategy_analysis.md) | [results/grouping_strategy_analysis/](../../results/grouping_strategy_analysis/) | The compatibility-year identity had manageable fragmentation: 1,032 groups and 18.31% singletons. |
| Fragmentation analysis | Measure how OEM and compatibility years split broad identities. | [scripts/analyze_grouping_fragmentation.py](../../scripts/analyze_grouping_fragmentation.py) | [results/grouping_strategy_fragmentation.md](../../results/grouping_strategy_fragmentation.md) | [results/grouping_strategy_fragmentation/](../../results/grouping_strategy_fragmentation/) | OEM fragmented more aggressively, creating 309 new singleton groups versus 93 for compatibility years. |
| Product-id identity leakage analysis | Estimate potential identity leakage under product-id-only splitting. | [scripts/analyze_product_id_identity_leakage.py](../../scripts/analyze_product_id_identity_leakage.py) | [results/product_id_identity_leakage.md](../../results/product_id_identity_leakage.md) | [results/product_id_identity_leakage/](../../results/product_id_identity_leakage/) | Product-id-only splitting leaves a large identity-leakage surface, especially under broad identities. |
| Price variability analysis | Measure within-group target-price dispersion. | [scripts/analyze_group_price_variability.py](../../scripts/analyze_group_price_variability.py) | [results/group_price_variability.md](../../results/group_price_variability.md) | [results/group_price_variability/](../../results/group_price_variability/) | Compatibility years reduced average variance relative to the broad identity while avoiding OEM-level fragmentation. |
| Connected component analysis | Analyze graph components built from product ID and candidate identity edges. | [scripts/analyze_identity_graph_components.py](../../scripts/analyze_identity_graph_components.py) | [results/identity_graph_components.md](../../results/identity_graph_components.md) | [results/identity_graph_components/](../../results/identity_graph_components/) | The final candidate graph produced 886 components, median size 8, and maximum size 71. |
| Candidate split balance analysis | Test whether connected components can produce balanced 70 / 15 / 15 diagnostic partitions. | [scripts/analyze_candidate_component_split_balance.py](../../scripts/analyze_candidate_component_split_balance.py) | [results/candidate_component_split_balance.md](../../results/candidate_component_split_balance.md) | [results/candidate_component_split_balance/](../../results/candidate_component_split_balance/) | Diagnostic seed 32 produced 70.05% / 14.97% / 14.98% row proportions with zero leakage. |

## Reproduction Order

Run analyses in this order:

1. Grouping strategy analysis.
2. Fragmentation analysis.
3. Product-id identity leakage analysis.
4. Price variability analysis.
5. Connected component analysis.
6. Candidate split balance analysis.

The final protocol decision is documented in [01_PROTOCOL_DECISION.md](01_PROTOCOL_DECISION.md). The technical protocol is specified in [02_PROTOCOL_SPECIFICATION.md](02_PROTOCOL_SPECIFICATION.md).
