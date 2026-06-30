# Stage 2C-1 CFG Residual Gate Design

Date: 2026-06-30

Status: approved for immediate implementation after Stage 2C-0.

## Motivation

Stage 2C-0 residual magnitude gating proved that target-level intervention can make personalized outputs closer to the base model. However, it also moved subject prompts closer to the base model, which means the gate likely suppressed subject-bearing residuals along with off-prior residuals.

The next step should keep the useful part of residual gating while adding a cheap semantic signal. We first considered comparing base predictions under the instance prompt and class prompt, but this is weak because the base model does not know the rare `sks` token. The difference between `a photo of sks vase` and `a photo of a vase` may be tiny or noisy before personalization.

## Method

Stage 2C-1 uses the classifier-free guidance direction as a proxy for generic class information:

```text
v_class = base_unet(x_t, t, class_prompt)
v_null  = base_unet(x_t, t, "")
d_class = v_class - v_null

r = v_ref - v_class
```

`d_class` is the base model's direction for making the noisy latent look like the generic class. If a reference residual `r` is aligned with `d_class`, that residual is more likely to be generic class or background behavior. If `r` is orthogonal or opposite to `d_class`, the class prompt explains it less well, so it is more likely to contain subject-specific information.

The target is:

```text
cos = cosine_channel(r, d_class)
gate = 1 - alpha * relu(cos)
v_target = v_class + gate * r
```

where `alpha` is in `[0, 1]`. Positive alignment with the class direction is suppressed. Orthogonal and negative residuals are preserved.

## Scope

Implement one condition:

```text
dadt_cfg_residual_gate
```

Initial config:

```text
alpha: 0.5
```

Keep SD 1.5, LoRA rank 4, 200 steps, the same `vase` references, and the same Stage 2B/2C eval prompts.

## Diagnostic

Before trusting the method, run a no-training diagnostic over the `vase` references:

```text
cosine(r, d_class)
```

The output should report mean, standard deviation, min, max, and the positive-cosine ratio. If the map is nearly all zero, all positive, or all negative, the gate may not have useful separation.

## Success Criterion

Compared with Stage 2C-0, a promising result should:

```text
1. keep class-prompt distance to base below vanilla;
2. increase subject-prompt distance to base above Stage 2C-0;
3. preserve visually plausible subject-prompt identity better than q75/q90 residual magnitude gates.
```

This is still a triage experiment, not a final paper metric.

## Reviewer Risk

The main risk is that CFG direction is still a heuristic. We should describe it as a cheap proxy for base class semantics, not as a perfect subject/background separator. If the direction works, later experiments should add subject diversity, sensitivity over `alpha`, and stronger subject fidelity metrics such as DINO or CLIP image similarity.
