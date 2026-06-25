# Distribution-Aligned Denoising Targets for Diffusion Personalization Design

Date: 2026-06-21

Status: v0 research design plus Stage 1 measurement code. The first SD 1.5/DreamBooth smoke test produced a No-Go on the lightweight subset. Stage 1 v2 also produced a No-Go, with VAE roundtrip controls identified as a major confound. Stage 1.3 confirmed the confound: the raw standard-reference signal disappears after roundtrip subtraction. Stage 1.4 is now specified in `docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md` to separate VAE/projection artifact, ordinary real-image domain gap, and DreamBooth subject-specific gap before any personalization fine-tuning.

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

MixSD is the main conceptual inspiration. It argues that supervised fine-tuning can forget because expert targets diverge from the base language model's native autoregressive distribution. Its method constructs distribution-aligned targets by mixing an expert conditional and a naive/self conditional from the base model, rather than training on fixed expert targets. Its evidence includes lower-NLL supervision under the base model and reduced movement along Fisher-sensitive directions.

Our migration:

| MixSD | This project |
| --- | --- |
| expert target sequence | reference-image denoising target `v_ref` |
| naive/self/base conditional | pretrained diffusion prediction `v_base` |
| target distribution gap | target off-priorness against the base score / velocity field |
| mixed contextual self-distillation | selective denoising target alignment |
| reduce SFT forgetting | reduce personalization prior drift |

Important distinction: MixSD motivates the logic, but a direct diffusion copy would be weak. The diffusion-native contribution should use timestep, trajectory, frequency, and optionally region structure.

Source: https://arxiv.org/abs/2605.16865, verified on 2026-06-21.

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

### Direct Consistency Optimization

Direct Consistency Optimization (DCO) is a very close personalization baseline. It improves robust customization by controlling deviation between a fine-tuned model and the pretrained model, with a Pareto framing over subject/style consistency and prompt fidelity.

This is closer than ordinary prior preservation because it explicitly anchors fine-tuning to pretrained behavior. The key distinction must be:

```text
DCO: modify the fine-tuning objective so the personalized model stays close to the pretrained model.
Ours: measure and edit the reference denoising target before the personalized model learns it.
```

This distinction is not enough by wording alone. DCO must be treated as a priority baseline or, if reproduction is not immediately feasible, as a documented nearest neighbor with a faithful surrogate comparison.

Sources: https://arxiv.org/abs/2402.12004 and https://openreview.net/forum?id=VazkRbCGxt

### Score Distillation Methods

SDS and VSD use pretrained diffusion scores as guidance or distillation signals, especially in text-to-3D optimization. They are not personalization methods in the same setting, but reviewers may connect them to any method that compares a target direction with a pretrained model score.

The distinction should be:

```text
SDS / VSD: use pretrained diffusion scores to guide optimization of another representation or generator.
Ours: modifies training-time personalization supervision on reference-image denoising targets.
```

This project should not claim that using a pretrained score as a reference is new. The novelty must be the diagnosis of reference-target off-priorness in personalization and the selective correction of prior-harmful target components.

Sources: https://arxiv.org/abs/2209.14988 and https://arxiv.org/abs/2305.16213

### Timestep Loss Weighting

Min-SNR and P2 weighting show that diffusion timesteps should not always be treated uniformly during training. This creates a near-neighbor risk for any timestep-aware schedule.

The distinction should be:

```text
Min-SNR / P2: reweight losses across timesteps.
Ours: changes the denoising target components using measured off-priorness.
```

Experiments must separate target correction from simple timestep reweighting. A Min-SNR or P2-style baseline is therefore useful once DADT training begins.

Sources: https://arxiv.org/abs/2303.09556 and https://arxiv.org/abs/2204.00227

### Spectral Progressive Diffusion

Spectral Progressive Diffusion shows that visual content emerges progressively in frequency space: low-frequency components appear earlier in denoising, while high-frequency details emerge later.

This supports a diffusion-native target alignment design:

```text
low-frequency / high-noise components -> preserve class, layout, background prior
high-frequency / low-noise components -> allow subject-specific identity details
```

The method should therefore avoid one global mixing coefficient. A frequency-aware and timestep-aware correction is much more defensible than uniform velocity averaging.

Source: https://arxiv.org/abs/2605.18736, verified on 2026-06-21.

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

The conditioning `c_base` is a core measurement axis, not a minor implementation detail. Stage 1 should report off-priorness under at least three choices:

- class prompt, such as `a photo of a dog`;
- unconditional or empty prompt;
- class-plus-context prompt, such as the class and scene without the rare token;
- a pair of conditional and unconditional predictions used to separate text-conditioned and prior components.

The default v0 report should use the class prompt, while the other variants test whether the conclusion is robust to conditioning.

The first research question is:

```text
When and where is v_ref incompatible with v_base?
```

The second research question is:

```text
Can we correct only the prior-harmful components of v_ref while preserving subject identity?
```

## Core Claims To Validate

### Claim 1: Target Off-Priorness Is Measurable Above The Base Error Floor

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
- `normalized_distance` can start as cosine distance plus normalized L2 residual after per-timestep scale normalization.

This score is diagnostic first. The method should not depend on a hand-wavy claim that the target is "bad"; it should show where the target departs from the pretrained field.

Important caveat:

```text
v_ref - v_base includes both target off-priorness and ordinary pretrained-model prediction error.
```

Therefore Stage 1 must estimate a base error floor by running the same measurement on images generated by the pretrained model under class prompts. A reference target should be called off-prior only when its residual is meaningfully above this base-generated floor.

### Claim 2: Off-Priorness Has Semantically Structured Non-Uniformity

The most important figure should show that target off-priorness is not merely non-uniform, but structured in a way that matches diffusion semantics:

```text
timestep x frequency x region -> off-priorness heatmap
```

Expected pattern:

- high-noise / low-frequency deviations are more likely to affect class, layout, and background prior;
- low-noise / high-frequency deviations are more likely to carry subject texture and identity;
- background deviations should often be more prior-harmful than subject-region deviations.

This claim motivates selective correction. If off-priorness is flat everywhere, or if the structure does not align with timestep/frequency roles, the method collapses into ordinary distillation or regularization.

### Claim 3: Off-Priorness Predicts Forgetting

The diagnostic must correlate with later behavior at the subject level:

```text
higher off-priorness before fine-tuning -> more prior drift after fine-tuning
```

Potential dependent variables:

- class-prompt diversity drop;
- prompt adherence drop on non-reference contexts;
- background binding to the reference images;
- class prior FID/KID against base-model generated class samples;
- subject prompt overfitting under long fine-tuning.

This is the claim that can make the paper feel new. The first planned threshold should be modest but predeclared:

```text
Spearman correlation rho >= 0.5 across at least 20 subjects
```

for at least one prior-drift metric after controlling for subject fidelity. The exact threshold can be revised before experiments start, but the analysis must not be chosen after looking at the result.

### Claim 4: Selective Target Correction Improves The Pareto Frontier

The method should outperform:

```text
v_mixed = alpha * v_ref + (1 - alpha) * v_base
```

on the subject-fidelity / prior-preservation Pareto frontier, not just at a single chosen hyperparameter. The comparison must be explicit. Otherwise reviewers will see the method as simple diffusion distillation.

### Claim 5: Target Correction Preserves Subject Fidelity

The method must not only improve prior preservation. It must also show that target correction does not erase the personalized subject. Report:

- prior preservation at matched subject fidelity;
- subject fidelity at matched prior preservation;
- failure cases where target correction over-preserves the base model and underfits the subject.

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

This is an inductive bias, not a solved detector of prior harm. The spec should not pretend that low-frequency or background residuals are automatically harmful. Stage 1 and Stage 2 should test whether those components actually correlate with forgetting.

The longer-term version can use a more theory-flavored projection, such as Fisher-sensitive directions, local score-field curvature, or one-step denoise reconstruction residual.

### Gate Design

Avoid a manually chosen global alpha. Use a gate driven by off-priorness, while keeping the first version simple enough to audit:

```text
G(t, b, r) = sigmoid((O_norm(t, b, r) - tau_b) / temp) * S(t, b, r)
```

where:

- `O_norm(t, b, r)` is measured off-priorness after subtracting or normalizing by the base error floor;
- `tau_b` is a band-specific threshold;
- `S(t, b, r)` encodes a conservative schedule, such as stronger low-frequency correction at high-noise timesteps.

The gate should be interpreted as:

```text
high gate -> move target toward base field
low gate -> keep reference target
```

The v0 implementation should sweep only a small number of gate strengths. If performance depends on fragile hyperparameters, the method will look like another regularization knob.

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

Important sanity check:

```text
latent-space frequency is not automatically image-space frequency.
```

The first measurement experiment should compare latent-space DCT statistics with decoded image-space frequency statistics on a small subset before making strong claims about low-frequency and high-frequency semantics.

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

Dataset survey note:

```text
notes/2026-06-22-dataset-survey.md
```

The current dataset decision is to start with DreamBooth / DreamBench because it is the shared benchmark for DreamBooth, Preserve and Personalize, and DCO-style personalization evaluation. CustomConcept101 and DreamBench++ are reserved for later scale-up after the diagnostic signal is validated.

### Stage 1: Measurement Before Training

Goal: show that reference targets are off-prior in a structured way, above the pretrained model's ordinary prediction-error floor.

Inputs:

- DreamBooth dataset, starting with a 3-5 subject smoke-test subset;
- easy / standard / hard reference-prior-compatibility regimes around the DreamBooth-style subjects;
- all 30 DreamBooth subjects for the main measurement after the smoke test;
- CustomConcept101 or DreamBench++ only after the DreamBooth diagnostic works;
- SD 1.5 first, then SDXL if resources allow.

Reference regimes:

```text
Easy / in-prior control: base-generated or highly typical class images.
Standard: original DreamBooth reference images.
Hard / off-prior: unusual background, lighting, pose, crop, style, or strong subject-background correlation.
```

The easy regime is a sanity control, not evidence for the phenomenon, because base-generated images are expected to be close to the base field. The hard regime is a declared stress test for reference prior compatibility. It should not be presented as ordinary random sampling or hidden cherry-picking.

The primary analysis should use continuous off-priorness scores and downstream drift scores. Easy / standard / hard bins are useful for readability, but the key plot should be:

```text
pre-training off-priorness vs. downstream prior drift
```

not only a three-bin bar chart.

Procedure:

1. Encode reference images into latents.
2. Sample timesteps and noise.
3. Compute raw target `v_ref`.
4. Run pretrained model `M_0` at the same `(z_t, t, c_base)` to obtain `v_base`.
5. Normalize residuals by timestep scale or SNR so the heatmap does not simply reproduce the scheduler scale.
6. Estimate the base error floor by repeating the measurement on images generated by `M_0` from class prompts.
7. Repeat measurement for `c_base` variants: null prompt, class prompt, and class-plus-context prompt.
8. Compare with an in-distribution control image set, such as base-generated class images or a small LAION-like class sample if available.
9. Run a VAE roundtrip control to detect residuals caused by encode/decode artifacts.
10. Decompose residuals by timestep and frequency, first in latent space and then on a decoded-image sanity subset.
11. Produce heatmaps and summary statistics.

Controls required before interpreting the score as off-priorness:

- **Base-model error floor:** compare reference-image residuals to base-generated image residuals.
- **Conditioning robustness:** check whether the conclusion holds under null, class, and class-plus-context prompts.
- **Timestep scale:** normalize by timestep variance or SNR.
- **Dataset shift:** compare subject reference images with in-distribution class images.
- **VAE artifact:** compare raw reference latents with VAE-roundtripped references.
- **Frequency semantics:** check whether latent-space frequency findings agree with decoded-image sanity checks.

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
- Direct Consistency Optimization or a faithful DCO-style surrogate if the official setup is not feasible;
- uniform target fusion;
- timestep-only target fusion;
- timestep-weighted prior preservation loss;
- Min-SNR or P2-style timestep loss weighting;
- VSD/SDS-style target guidance surrogate if a fair training-time adaptation is possible;
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
- gate-strength sweep to verify robustness to `tau_b`, `temp`, and schedule strength;
- `c_base` choice: null, class prompt, class-plus-context prompt;
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

### Risk 4: "DCO already anchors personalization to the pretrained model."

Mitigation:

- add DCO as a priority near-neighbor and baseline;
- emphasize that DCO controls model updates while DADT changes the supervision target;
- compare against DCO or a faithful surrogate on the same subject-fidelity / prior-preservation frontier.

### Risk 5: "SDS/VSD already use pretrained scores."

Mitigation:

- avoid claiming that pretrained-score references are new;
- position DADT as a training-time personalization target correction method, not a sample or representation optimization method;
- include a VSD/SDS-style surrogate if a fair baseline can be defined.

### Risk 6: "The method is only timestep loss weighting."

Mitigation:

- include Min-SNR or P2-style loss weighting baselines;
- separate loss reweighting from target modification in the ablation.

### Risk 7: "The off-priorness metric is heuristic."

Mitigation:

- start with simple metrics for v0, but connect them to lower-NLL / Fisher-sensitive directions inspired by MixSD;
- later add a theory-flavored metric such as local Jacobian norm, Fisher-weighted residual, or one-step denoise residual.

### Risk 8: "The residual is just base-model prediction error."

Mitigation:

- measure and subtract or normalize by the base-generated image residual floor;
- report results under multiple `c_base` choices;
- avoid interpreting raw `v_ref - v_base` as off-priorness before controls pass.

### Risk 9: "Latent-space frequency has unclear semantics."

Mitigation:

- present latent DCT as a practical measurement, not as guaranteed image-frequency semantics;
- run decoded-image frequency sanity checks on a small subset;
- consider wavelet or decoded-image analysis if latent results are hard to interpret.

### Risk 10: "Target correction weakens personalization."

Mitigation:

- always report subject fidelity at matched prior preservation and prior preservation at matched subject fidelity;
- show high-frequency and subject-region components are preserved more than low-frequency/background components.

### Risk 11: "The effect only appears on easy subjects."

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

1. Use the near-neighbor reading note at `notes/2026-06-21-near-neighbor-reading.md` as the current related-work baseline.
2. Use the dataset survey at `notes/2026-06-22-dataset-survey.md` as the current benchmark-selection baseline.
3. Use the Stage 1 implementation plan at `docs/superpowers/plans/2026-06-22-stage-1-off-priorness-measurement.md` as the current execution plan.
4. Use the implemented code under `scripts/off_prior_measurement/` to run the DreamBooth 5-subject smoke test on a PyTorch/diffusers environment.
5. Generate and inspect `experiments/off_prior_measurement_v0/smoke_test/conclusion.md`.
6. Use the smoke test as a go/no-go check: if reference residuals are not above the base-generated floor, the off-prior framing must be revised before method work.
