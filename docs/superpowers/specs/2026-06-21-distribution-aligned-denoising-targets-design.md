# Distribution-Aligned Denoising Targets for Diffusion Personalization Design

Date: 2026-06-21

Status: v0 research design for review. No implementation has started.

## Purpose

This project studies diffusion personalization forgetting from the target-construction side.

The central hypothesis is:

```text
Personalization forgetting can start before fine-tuning updates the model,
because reference images may induce denoising targets that are off-prior
with respect to the pretrained diffusion score / velocity field.
```

Existing anti-forgetting methods usually act after this point: they add class-prior replay samples, regularize model updates, preserve output distributions, or restrict trainable parameters. This project asks a prior question: whether the reference denoising targets themselves should be measured and corrected before they are used for personalization.

The intended reviewer hook is:

```text
Personalization Forgetting Starts Before Fine-Tuning.
```

## Related Work Positioning

### MixSD

MixSD is the main conceptual inspiration. It argues that supervised fine-tuning can forget because expert targets diverge from the base language model's native autoregressive distribution. Its method constructs distribution-aligned targets by mixing an expert conditional and a naive/self conditional from the base model, rather than training on fixed expert targets.

Our migration:

| MixSD | This project |
| --- | --- |
| expert target sequence | reference-image denoising target `v_ref` |
| naive/self/base conditional | pretrained diffusion prediction `v_base` |
| target distribution gap | target off-priorness against the base score / velocity field |
| mixed contextual self-distillation | selective denoising target alignment |
| reduce SFT forgetting | reduce personalization prior drift |

Important distinction: MixSD motivates the logic, but a direct diffusion copy would be weak. The diffusion-native contribution should use timestep, trajectory, frequency, and optionally region structure.

Source: https://arxiv.org/abs/2605.16865

### DreamBooth

DreamBooth is the classic subject-driven diffusion personalization method. It fine-tunes a pretrained text-to-image diffusion model on a few subject images and uses class-specific prior preservation to retain the semantic prior of the class.

SFT analogy:

```text
DreamBooth prior preservation ~= replay / rehearsal.
```

DreamBooth does not ask whether the reference denoising target itself is off-prior. It mainly adds class-prior examples so the model does not collapse the whole class distribution into the reference subject.

Source: https://arxiv.org/abs/2208.12242

### Preserve and Personalize

Preserve and Personalize is the most dangerous nearest neighbor. It explicitly frames personalization as a distributional drift problem and uses a Lipschitz-based regularization objective to constrain parameter updates during personalization.

SFT analogy:

```text
Preserve and Personalize ~= constrained optimization / KL-style regularization.
```

Our distinction must be crisp:

```text
P&P: constrain the model so it does not drift too far from the pretrained distribution.
Ours: change the personalization supervision target before the model learns off-prior directions.
```

This means P&P is reactive or constraint-level preservation, while this project is proactive or target-level preservation. Experiments must include strong regularization baselines, including timestep-weighted prior preservation, to prevent reviewers from reducing this work to P&P with a different weight schedule.

Sources: https://arxiv.org/abs/2505.19519 and https://openreview.net/forum?id=p4oYf6IbG5

### Spectral Progressive Diffusion

Spectral Progressive Diffusion shows that visual content emerges progressively in frequency space: low-frequency components appear earlier in denoising, while high-frequency details emerge later.

This supports a diffusion-native target alignment design:

```text
low-frequency / high-noise components -> preserve class, layout, background prior
high-frequency / low-noise components -> allow subject-specific identity details
```

The method should therefore avoid one global mixing coefficient. A frequency-aware and timestep-aware correction is much more defensible than uniform velocity averaging.

Source: https://arxiv.org/abs/2605.18736

## Problem Definition

Let `M_0` be the pretrained diffusion model and `M_theta` be the personalized model.

For a reference image latent `z_0`, sampled noise `eps`, timestep `t`, and noisy latent `z_t`, standard diffusion training constructs a denoising target:

```text
v_ref = scheduler_target(z_0, eps, t)
```

Depending on the model parameterization, `v_ref` may represent epsilon, velocity, or another scheduler-specific target. The design should use the model's native target type.

The pretrained model can also predict a base denoising direction at the same noisy state:

```text
v_base = M_0(z_t, t, c_base)
```

The conditioning `c_base` is not fixed yet. Candidate choices:

- class prompt, such as `a photo of a dog`;
- reference prompt with the rare token before personalization, such as `a photo of [V] dog`;
- unconditional or empty prompt;
- a pair of conditional and unconditional predictions used to separate text-conditioned and prior components.

The first research question is:

```text
When and where is v_ref incompatible with v_base?
```

The second research question is:

```text
Can we correct only the prior-harmful components of v_ref while preserving subject identity?
```

## Core Claims To Validate

### Claim 1: Target Off-Priorness Is Measurable

Define an off-priorness score over timestep, frequency band, and optionally spatial region:

```text
O(t, b, r) = normalized_distance(
    component(v_ref, t, b, r),
    component(v_base, t, b, r)
)
```

where:

- `b` is a frequency band, such as low / mid / high DCT or wavelet components;
- `r` is optional spatial region, such as subject mask or background;
- `normalized_distance` can start as cosine distance plus normalized L2 residual.

This score is diagnostic first. The method should not depend on a hand-wavy claim that the target is "bad"; it should show where the target departs from the pretrained field.

### Claim 2: Off-Priorness Is Non-Uniform

The most important figure should show that target off-priorness is structured, not uniform:

```text
timestep x frequency x region -> off-priorness heatmap
```

Expected pattern:

- high-noise / low-frequency deviations are more likely to affect class, layout, and background prior;
- low-noise / high-frequency deviations are more likely to carry subject texture and identity;
- background deviations should often be more prior-harmful than subject-region deviations.

This claim motivates selective correction. If off-priorness is flat everywhere, the method collapses into ordinary distillation or regularization.

### Claim 3: Off-Priorness Predicts Forgetting

The diagnostic must correlate with later behavior:

```text
higher off-priorness before fine-tuning -> more prior drift after fine-tuning
```

Potential dependent variables:

- class-prompt diversity drop;
- prompt adherence drop on non-reference contexts;
- background binding to the reference images;
- class prior FID/KID against base-model generated class samples;
- subject prompt overfitting under long fine-tuning.

This is the claim that can make the paper feel new.

### Claim 4: Selective Target Correction Beats Uniform Mixing

The method should outperform:

```text
v_mixed = alpha * v_ref + (1 - alpha) * v_base
```

at matched subject fidelity. The comparison must be explicit. Otherwise reviewers will see the method as simple diffusion distillation.

## Proposed Method: DADT

Working name:

```text
Distribution-Aligned Denoising Targets (DADT)
```

DADT replaces the raw personalization target with a corrected target:

```text
v_dadt = v_ref - G(t, b, r) * P_prior_harmful(v_ref - v_base)
```

where:

- `G(t, b, r)` is an adaptive gate;
- `P_prior_harmful` selects the component of the target residual that is likely to hurt the prior;
- the remaining residual is preserved so subject identity can still be learned.

For v0 experiments, `P_prior_harmful` can be simple:

```text
P_prior_harmful(delta_v) = component(delta_v, selected frequency band and region)
```

The longer-term version can use a more theory-flavored projection, such as Fisher-sensitive directions, local score-field curvature, or one-step denoise reconstruction residual.

### Gate Design

Avoid a manually chosen global alpha. Use a gate driven by off-priorness:

```text
G(t, b, r) = clip(sigmoid((O(t, b, r) - tau_b) / temp) * S(t, b, r), 0, 1)
```

where:

- `O(t, b, r)` is measured off-priorness;
- `tau_b` is a band-specific threshold;
- `S(t, b, r)` encodes a conservative schedule, such as stronger low-frequency correction at high-noise timesteps.

The gate should be interpreted as:

```text
high gate -> move target toward base field
low gate -> keep reference target
```

### Timestep Design

Initial hypothesis:

- high-noise timesteps should be more prior-preserving because they control global structure and class-level generation;
- low-noise timesteps should be more subject-permissive because they control visible details and identity texture.

This should be treated as a hypothesis, not an assumption. The measurement experiment must verify whether off-priorness and forgetting risk follow this pattern.

### Frequency Design

Initial decomposition:

- low-frequency band: strongest target alignment toward base;
- mid-frequency band: adaptive alignment;
- high-frequency band: weakest alignment unless off-priorness is extreme or appears in background.

DCT is a simple first choice. Wavelet decomposition is a possible later improvement because it keeps more spatial localization.

### Region Design

Region-aware alignment is useful but should not be the main novelty in v0 because it can look like straightforward segmentation engineering.

Initial use:

- optional subject mask from reference images;
- stronger correction in background than foreground;
- report region-aware results as an ablation or extension.

The first paper story should emphasize off-priorness measurement plus timestep/frequency-aware target correction.

## Alternative Method Options

### Option A: Diagnostic-Only Paper

Focus on measuring target off-priorness and showing it predicts forgetting.

Pros:

- clean scientific story;
- low implementation burden;
- strong if correlation is striking.

Cons:

- may be seen as incomplete without a method;
- harder to publish as a main conference method paper.

### Option B: Uniform Target Fusion Baseline

Use fixed alpha or timestep-only alpha:

```text
v_mixed = alpha(t) * v_ref + (1 - alpha(t)) * v_base
```

Pros:

- easy to implement;
- useful as a sanity baseline.

Cons:

- too close to obvious distillation;
- weak novelty.

### Option C: Off-Priorness-Adaptive Timestep/Frequency Target Alignment

Use DADT as described above.

Pros:

- directly connects measurement to method;
- diffusion-native;
- clearly different from replay and regularization.

Cons:

- more moving pieces;
- needs careful ablation to prove each piece matters.

Chosen v0 direction: Option C, with Option A as the first experiment and Option B as a mandatory baseline.

## Minimum Experiment Plan

### Stage 1: Measurement Before Training

Goal: show that reference targets are off-prior in a structured way.

Inputs:

- DreamBooth dataset or a small curated subset;
- CustomConcept101 if available;
- SD 1.5 first, then SDXL if resources allow.

Procedure:

1. Encode reference images into latents.
2. Sample timesteps and noise.
3. Compute raw target `v_ref`.
4. Run pretrained model `M_0` at the same `(z_t, t, c_base)` to obtain `v_base`.
5. Decompose `v_ref - v_base` by timestep and frequency.
6. Produce heatmaps and summary statistics.

Expected artifact:

```text
experiments/off_prior_measurement_v0/
```

with configs, summary CSV, heatmaps, and a short conclusion note.

### Stage 2: Correlation With Forgetting

Goal: test whether off-priorness predicts later prior drift.

Procedure:

1. Fine-tune vanilla DreamBooth or LoRA personalization for each subject.
2. Measure subject fidelity and prior preservation after training.
3. Correlate pre-training off-priorness with post-training prior drift.

Key plot:

```text
pre-training target off-priorness vs. post-training prior drift
```

This plot is more important than early method results because it validates the new explanation.

### Stage 3: DADT Training

Goal: show that aligned targets improve the subject-fidelity / prior-preservation trade-off.

Baselines:

- vanilla DreamBooth;
- DreamBooth with class-specific prior preservation;
- LoRA personalization;
- Custom Diffusion if feasible;
- uniform target fusion;
- timestep-only target fusion;
- timestep-weighted prior preservation loss;
- P&P-style regularization if implementation is available or faithfully reproduced.

Metrics:

- subject fidelity: DINO / CLIP-I against reference images;
- text alignment: CLIP-T or equivalent;
- prior preservation: class-prompt diversity, FID/KID to base-model class samples, prompt adherence on held-out contexts;
- overfitting: similarity to reference backgrounds and poses.

Critical reporting rule:

```text
Report Pareto curves, not only single numbers.
```

The method must improve prior preservation at matched subject fidelity.

### Stage 4: Ablation

Required ablations:

- no off-priorness gate, fixed alpha only;
- timestep gate only;
- frequency gate only;
- timestep + frequency gate;
- random gate with same average strength;
- background/foreground region gate if region-aware alignment is included.

The ablation should answer whether DADT is more than a regularization-strength knob.

## Reviewer Risk Register

### Risk 1: "This is just DreamBooth prior preservation with timestep weights."

Mitigation:

- include timestep-weighted prior preservation as a strong baseline;
- show target correction improves the Pareto frontier at matched subject fidelity.

### Risk 2: "This is just base-model distillation."

Mitigation:

- include uniform fusion and pure distillation baselines;
- show selective correction preserves identity better than global pulling toward the base model;
- emphasize that only high-off-prior, prior-harmful components are corrected.

### Risk 3: "P&P already solves distributional drift."

Mitigation:

- present P&P as regularization-level preservation;
- present DADT as target-level preservation;
- compare target correction vs. equivalent loss penalty when feasible.

### Risk 4: "The off-priorness metric is heuristic."

Mitigation:

- start with simple metrics for v0, but connect them to lower-NLL / Fisher-sensitive directions inspired by MixSD;
- later add a theory-flavored metric such as local Jacobian norm, Fisher-weighted residual, or one-step denoise residual.

### Risk 5: "Target correction weakens personalization."

Mitigation:

- always report subject fidelity at matched prior preservation and prior preservation at matched subject fidelity;
- show high-frequency and subject-region components are preserved more than low-frequency/background components.

### Risk 6: "The effect only appears on easy subjects."

Mitigation:

- test object, animal, style-like, and human-adjacent concepts separately;
- include hard subjects with varied backgrounds and poses;
- report failure cases rather than only selected wins.

## Writing Plan

Potential title:

```text
Personalization Forgetting Starts Before Fine-Tuning:
Distribution-Aligned Denoising Targets for Diffusion Personalization
```

Alternative titles:

- `Are Personalization Targets Off-Prior?`
- `Off-Prior Targets: A Score-Field View of Diffusion Personalization`
- `Personalize Where It Is Safe`

Suggested paper structure:

1. Introduction: personalization forgetting may start at target construction.
2. Related Work: replay, regularization, distillation, MixSD-style target distribution alignment.
3. Diagnostic: define and measure denoising target off-priorness.
4. Method: DADT, with timestep/frequency-aware adaptive target correction.
5. Experiments: measurement, correlation, method comparison, ablation.
6. Limitations: metric design, scheduler dependence, segmentation dependence if used, compute cost.

Core contribution statement:

```text
We identify and measure target off-priorness in diffusion personalization,
show that it predicts prior drift, and introduce a selective target-alignment
method that improves prior preservation without sacrificing subject fidelity.
```

## Current Scope

In scope for v0:

- research framing;
- off-priorness metric design;
- first measurement experiment;
- method baselines and ablation design.

Out of scope for v0:

- final paper claims;
- large-scale multi-backbone experiments;
- heavy human evaluation;
- region-aware segmentation as the main contribution;
- committing raw images, model checkpoints, or large generated outputs.

## Immediate Next Steps

1. Review this spec for conceptual correctness and missing near-neighbor papers.
2. Ask Claude or another external model for reviewer-style critique of the spec.
3. Create an implementation plan for Stage 1 measurement only.
4. Set up environment and choose the first backbone, likely SD 1.5.
5. Run a small measurement smoke test on 3 to 5 subjects before scaling.
