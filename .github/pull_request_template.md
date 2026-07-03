## What changed?
<!-- A clear, concise description of the changes in this PR. -->

## Why?
<!-- Why is this change necessary? Link any related issues or context. -->

## Risk level
<!-- [Low | Medium | High] - Describe the blast radius. Call out changes to datasets, split generation, frozen split/model artifacts, evaluation scripts, protocol docs, or the serving app. -->

## Thesis evidence / leakage checklist
<!-- See docs/STRICT_EVALUATION_PROTOCOL.md and docs/evaluation/ for the full rules. -->
- [ ] No frozen artifacts (`datasets/splits_strict/`, final model artifacts) modified or regenerated without a documented reason
- [ ] No strict holdout (test) data used in training, tuning, or model selection
- [ ] Leakage assertions still pass, if split or identity logic was touched
- [ ] Reported metrics come from the documented pipeline, not old exploratory runs
- [ ] Protocol/roadmap docs updated, if the methodology changed
- [ ] Not applicable <!-- Explain why thesis evidence is unaffected -->

## Validation
<!-- Required before requesting review. Check the commands that ran, then paste real output below. Explain any command that was not run. -->
- [ ] `python -m pytest tests/` passes
- [ ] Changed scripts run end-to-end with their documented commands, if script behavior changed
- [ ] Reruns are reproducible from a fixed seed/config, if experiments changed
- [ ] CI passes, or skipped with reason below

## Verification output
<!-- Paste exact result lines, not only command names. Examples: "17 passed, 14 warnings in 3.00s", "All leakage assertions passed." -->

## Documentation updated?
<!-- What files changed? List the files or sections updated (README, docs/, AGENTS.md, etc.), or "None". -->

## Checks not run
<!-- List skipped checks and why. -->

## Known limitations
<!-- Are there any edge cases, performance trade-offs, or incomplete features? -->
