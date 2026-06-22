# Diffusion Personalization Target Alignment

Working directory for the Distribution-Aligned Denoising Targets for Diffusion Personalization project.

Core idea: prevent personalization forgetting by constructing denoising targets that are more compatible with the pretrained score / velocity field, rather than only constraining parameter updates.

## Current Status

Current stage: Stage 1 off-priorness measurement pipeline implemented and run for a DreamBooth smoke test. Lightweight unit tests pass. The first full SD 1.5/DreamBooth smoke run is complete under `experiments/off_prior_measurement_v0/smoke_test/` and produced a No-Go under the current 4-of-5 subject rule.

Current design inputs:

```text
docs/superpowers/specs/2026-06-21-distribution-aligned-denoising-targets-design.md
docs/superpowers/specs/2026-06-22-prior-compatibility-ladder-design.md
docs/superpowers/plans/2026-06-22-stage-1-off-priorness-measurement.md
notes/2026-06-22-dataset-survey.md
```

Immediate next step: review the Stage 1 v2 prior-compatibility ladder design, then write an implementation plan if the design is approved.

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

- Stage 1 smoke test: 5 DreamBooth subjects with easy / standard / hard reference regimes. The current runnable subset is dog, cat, backpack, clock, and vase because this environment can reliably fetch those smaller DreamBooth files through the GitHub Contents API.
- Main measurement: all 30 DreamBooth subjects.
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
