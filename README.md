# Diffusion Personalization Target Alignment

Working directory for the Distribution-Aligned Denoising Targets for Diffusion Personalization project.

Core idea: prevent personalization forgetting by constructing denoising targets that are more compatible with the pretrained score / velocity field, rather than only constraining parameter updates.

## Current Status

Current stage: Stage 1 off-priorness measurement produced repeated No-Go / Pivot signals. Stage 1.5D fine-grained diagnosis found only a weak low-frequency late-timestep clue, so Stage 2A was treated as training validation rather than a claimed method result. Stage 2A now has a real SD 1.5 LoRA DreamBooth loop, stable fp32 LoRA training, tested LF-Late target alignment, and reproducible `vase` evaluation grids for base / vanilla / DADT-LF-Late. The first 200-step `vase` comparison is runnable but not convincing: vanilla and DADT-LF-Late look very similar under the current mild alignment setting. The next research step is a Stage 2B redesign of the alignment strength / timestep schedule and evaluation metrics before expanding to more subjects.

Current design inputs:

```text
docs/superpowers/specs/2026-06-21-distribution-aligned-denoising-targets-design.md
docs/superpowers/specs/2026-06-22-prior-compatibility-ladder-design.md
docs/superpowers/specs/2026-06-23-roundtrip-confound-clean-offpriorness-design.md
docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md
docs/superpowers/plans/2026-06-22-stage-1-off-priorness-measurement.md
docs/superpowers/plans/2026-06-22-stage-1-v2-prior-compatibility-ladder.md
docs/superpowers/plans/2026-06-25-stage-1-3-clean-offpriorness.md
docs/superpowers/plans/2026-06-25-stage-1-4-target-gap-source-decomposition.md
notes/2026-06-22-dataset-survey.md
```

Immediate next step: Stage 2A `vase` 200-step training and evaluation grids have run. Next run a Stage 2B redesign pass before expanding to `dog`; see `experiments/stage2a_lf_late/README.md`.

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
- Stage 1.4 source decomposition: completed comparison among base-generated controls, COCO ordinary real class controls, and DreamBooth references for dog, cat, backpack, and vase. Result: Pivot. The average subject-specific gap is positive only because dog is large; backpack, cat, and vase are negative relative to ordinary real controls, and artifact fraction remains high.
- Stage 1.5A failure diagnosis: repeated source decomposition with four ordinary-real control regimes per class and multiple DreamBooth reference images. Result: stronger Pivot. DreamBooth references are not more off-prior than ordinary real images under the current clean residual metric; all four subject-specific gaps are negative.
- Stage 1.5B prompt-matched diagnosis: repeated the ordinary-real versus base-generated comparison using image-specific prompts. Result: prompt matching reduces the mean real-domain gap by only about 6.8%, so coarse class prompts are not enough to explain the failed signal.
- Stage 1.5C metric ablation: reused Stage 1.5A/B raw measurements to compare `raw_norm`, `projected_artifact_norm`, and `clean_norm`. Result: prompt matching cuts raw and artifact-aligned gaps by about half, but only cuts the clean gap by about 6.8%; DreamBooth subject-specific gaps remain negative in Stage 1.5A across all three scalar views.
- Stage 1.5D fine-grained diagnosis: reused Stage 1.5A/B raw measurements to slice gaps by timestep and DCT clean-frequency bands. Result: clean-norm by timestep remains negative, but low-frequency late timesteps show a tiny class-consistent positive clue. Treat this as a metric-redesign lead, not as a paper-ready personalization-specific result.
- Stage 2A LF-Late training validation: real LoRA training is wired with tested DCT target utilities, LF-Late target alignment, smoke config validation, subject-filtered dry-runs, 200-step `vase` jobs for `vanilla` / `dadt_lf_late`, and base / vanilla / DADT evaluation grids. Result status: runnable training-validation pass, but no method win; vanilla and DADT-LF-Late look very similar under the current mild setting.
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
