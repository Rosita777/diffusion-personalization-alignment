# Diffusion Personalization Target Alignment

Working directory for the Distribution-Aligned Denoising Targets for Diffusion Personalization project.

Core idea: prevent personalization forgetting by constructing denoising targets that are more compatible with the pretrained score / velocity field, rather than only constraining parameter updates.

## Current Status

Current stage: Stage 1 off-priorness measurement pipeline implemented and run for two DreamBooth smoke tests plus a Stage 1.3 clean diagnostic. The first SD 1.5/DreamBooth smoke run is complete under `experiments/off_prior_measurement_v0/smoke_test/` and produced a No-Go under the 4-of-5 subject rule. Stage 1 v2 prior-compatibility ladder is complete under `experiments/off_prior_measurement_v0/ladder_v2/` and also produced a No-Go. Stage 1.3 roundtrip-confound diagnosis is complete under `experiments/off_prior_measurement_v0/ladder_v2_clean/` and produced a stronger No-Go: the raw standard-reference signal does not survive VAE roundtrip subtraction.

Current design inputs:

```text
docs/superpowers/specs/2026-06-21-distribution-aligned-denoising-targets-design.md
docs/superpowers/specs/2026-06-22-prior-compatibility-ladder-design.md
docs/superpowers/specs/2026-06-23-roundtrip-confound-clean-offpriorness-design.md
docs/superpowers/plans/2026-06-22-stage-1-off-priorness-measurement.md
docs/superpowers/plans/2026-06-22-stage-1-v2-prior-compatibility-ladder.md
docs/superpowers/plans/2026-06-25-stage-1-3-clean-offpriorness.md
notes/2026-06-22-dataset-survey.md
```

Immediate next step: revise the off-priorness measurement itself before any personalization fine-tuning. Stage 1.3 found clean standard-reference positives for 0 of 8 subjects and mean clean standard-reference residual `-0.0067`, with roundtrip attribution ratio `1.1502`; the current metric is too confounded for Stage 2.

## Current Research Question

Diffusion personalization methods can forget or distort the pretrained class prior after fine-tuning on a few reference images. Existing anti-forgetting strategies often add prior samples, replay, regularization, or parameter constraints.

This project studies an earlier failure point: the denoising targets induced by reference images may already be off-prior with respect to the pretrained diffusion model's score / velocity field. If so, personalization forgetting can begin before fine-tuning changes the model.

## Working Thesis

Reference-image denoising targets should be measured and corrected before training when they contain prior-harmful components. The correction should be selective, not uniform:

- preserve subject identity where the target deviation is useful;
- align background, low-frequency, or high-off-prior components with the pretrained field when they mainly cause prior drift;
- make alignment timestep-aware, region-aware, frequency-aware, and driven by an off-priorness metric.

## Related Work Map

- DreamBooth: personalization with class-specific prior preservation, analogous to replay or rehearsal.
- Preserve and Personalize: distribution preservation through regularization or constrained optimization.
- MixSD: main inspiration; analyzes target distribution gap and constructs distribution-aligned targets.
- Spectral Progressive Diffusion: motivates timestep- and frequency-aware treatment of denoising signals.

## Current Dataset Decision

Use DreamBooth / DreamBench first because it is the common benchmark for DreamBooth, Preserve and Personalize, and DCO-style personalization work:

- Stage 1 smoke test: 5 DreamBooth subjects with easy / standard / hard reference regimes. The runnable subset was dog, cat, backpack, clock, and vase.
- Stage 1 v2 ladder smoke test: 8 DreamBooth subjects with easy controls, standard references, deterministic hard-reference variants, hard controls, and VAE roundtrip controls. The completed subset is dog, cat, backpack, vase, colorful_sneaker, shiny_sneaker, fancy_boot, and dog7. It uses one reference image per subject because GitHub-hosted large DreamBooth files were unstable in this environment.
- Stage 1.3 clean diagnostic: roundtrip-subtracted analysis of v2 under `ladder_v2_clean`, producing No-Go because the raw standard-reference signal disappears after subtracting VAE roundtrip artifacts.
- Future paper-scale measurement: all 30 DreamBooth subjects only after the off-priorness metric and controls are revised.
- Later expansion: CustomConcept101 or DreamBench++ after the off-priorness signal is validated.

The hard reference regime should be documented as a controlled stress test, not hidden example selection.

## Project Hygiene Rule

Project hygiene is part of every task. Keep documentation, code, experiment records, method notes, references, writing materials, environment files, commands, and helper scripts aligned with the current research state.

Prefer updating or replacing existing documents over adding new overlapping files. When content becomes wrong, stale, duplicated, abandoned, or misleading, update it in place or remove it carefully. Failed experiments and rejected ideas may stay, but they must be clearly marked as inactive so they are not mistaken for the current plan.

Every completed task should leave the project easier to understand, easier to reproduce, less redundant, and less likely to mislead a new researcher.

## Backup And External Advice Workflow

This project uses the GitHub repository `Rosita777/diffusion-personalization-alignment` as the remote backup for code, concise documentation, experiment configs, important metadata, and curated result summaries.

Do not commit local secrets, API tokens, raw dataset caches, model checkpoints, or large generated artifacts. Document large or external assets with manifests and reproduction notes instead.

Claude or another external model may be consulted as a research assistant for important decisions, reviewer-style critique, method alternatives, and writing feedback. Treat external-model advice as input for judgment, not as authority; decisions should be reconciled with the papers, experiments, and current project goals.

## Directory Layout

- `docs/superpowers/specs/`: current design specs and research plans, with status marked inside each file.
- `docs/superpowers/plans/`: implementation or experiment execution plans.
- `notes/`: concise working notes, paper notes, external-review notes, and writing notes that do not yet belong in a spec.
- `experiments/`: experiment configs, logs, summaries, and analysis outputs.
- `scripts/`: runnable code and utilities.
- `data/`: small metadata files or dataset manifests. Large datasets should be documented here but not copied blindly.
- `outputs/`: generated artifacts that are useful for inspection or analysis.
