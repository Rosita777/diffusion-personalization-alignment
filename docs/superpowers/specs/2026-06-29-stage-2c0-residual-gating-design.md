# Stage 2C-0 Residual Gating Design

Date: 2026-06-29

Status: approved for immediate minimal implementation.

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
