# Diffusion Personalization Target Alignment

Working directory for the Distribution-Aligned Denoising Targets for Diffusion Personalization project.

Core idea: prevent personalization forgetting by constructing denoising targets that are more compatible with the pretrained score / velocity field, rather than only constraining parameter updates.

## Current Status

Current stage: research design v0 after one external reviewer-style critique pass. No experiment code or training pipeline has been implemented yet.

Current design spec:

```text
docs/superpowers/specs/2026-06-21-distribution-aligned-denoising-targets-design.md
```

Immediate next step: create an implementation plan for the Stage 1 off-priorness measurement experiment with base-error-floor and conditioning controls.

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
