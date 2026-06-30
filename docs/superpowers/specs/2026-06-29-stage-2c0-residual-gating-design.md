# Stage 2C-0 Residual Gating Design

Date: 2026-06-29

Status: completed on the `vase` subject. This is a useful preservation signal, not a method win yet.

## Purpose

Stage 2B showed that fixed LF-Late target mixing is too blunt. The metric audit suggests the best DADT variants may simply learn slightly less, rather than finding a better subject/prior tradeoff.

Stage 2C-0 adds one minimal method variant:

```text
r = v_ref - v_base
v_target = v_base + g(r) * r
```

where `g(r)` is data-dependent. The goal is to suppress unusually large reference-induced pulls while preserving ordinary subject-learning residuals.

## First Gate

Use a per-sample spatial residual magnitude:

```text
m = mean_channel(abs(v_ref - v_base))
tau = quantile(m, q)
g = keep     where m > tau
g = 1.0      otherwise
```

Then:

```text
v_target = v_base + g * (v_ref - v_base)
```

Initial configs:

```text
dadt_residual_gate_q75_keep05: q=0.75, keep=0.5
dadt_residual_gate_q90_keep05: q=0.90, keep=0.5
```

## Interpretation

This is not global distillation. It only suppresses residuals that are unusually large within the current sample. If it works, it should move the fidelity/preservation tradeoff more cleanly than fixed alpha LF-Late.

## Non-Goals

Do not add CLIP/DINO in this pass.

Do not add masks or segmentation.

Do not expand beyond `vase` until dry-run and tests pass.

## 2026-06-29 Result

Both initial configs trained and evaluated successfully:

```text
residual_gate_q75_keep05
residual_gate_q90_keep05
```

GPU runtime note: the Codex shell can report zero CUDA devices when no GPU id is explicitly selected. Launch training and evaluation with `CUDA_VISIBLE_DEVICES=<gpu_id>`; with an explicit id, PyTorch sees the H800 GPUs correctly.

Metric summary against the Stage 2B base and vanilla eval grids:

| run | kind | distance to base | distance to vanilla | diversity |
| --- | --- | ---: | ---: | ---: |
| vanilla | class | 29.156 | 0.000 | 66.501 |
| vanilla | subject | 31.144 | 0.000 | 86.785 |
| residual_gate_q75_keep05 | class | 21.649 | 22.366 | 59.636 |
| residual_gate_q75_keep05 | subject | 20.586 | 23.435 | 84.428 |
| residual_gate_q90_keep05 | class | 24.068 | 19.940 | 63.844 |
| residual_gate_q90_keep05 | subject | 22.528 | 20.809 | 84.744 |

Interpretation:

```text
Residual gating clearly moves generated images closer to the base model than vanilla,
especially q75. However, it also moves subject prompts closer to base, so the current
gate is probably too conservative rather than a clean subject/prior tradeoff.
```

Next method implication:

```text
Keep residual gating as evidence that target-level control is effective, but redesign
the gate to protect subject-bearing residuals. The next version should avoid suppressing
all large residuals uniformly; likely options are subject/background-aware gating,
timestep-conditioned gating, or a gate based on prompt-paired residual agreement.
```
