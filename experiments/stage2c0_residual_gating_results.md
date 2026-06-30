# Stage 2C-0 Residual Gating Results

Date: 2026-06-29

Status: completed for the `vase` subject. This is a preservation signal, not a method win yet.

## Purpose

Stage 2C-0 tested whether target-level residual gating can preserve more of the base diffusion prior than vanilla LoRA DreamBooth.

The intervention uses:

```text
r = v_ref - v_base
m = mean_channel(abs(r))
tau = quantile(m, q)
g = keep where m > tau else 1
v_target = v_base + g * r
```

Two variants were run:

| run | residual quantile | keep value |
| --- | ---: | ---: |
| residual_gate_q75_keep05 | 0.75 | 0.50 |
| residual_gate_q90_keep05 | 0.90 | 0.50 |

All runs used SD 1.5, LoRA rank 4, learning rate `0.0001`, 200 steps, seed 0, and the same three `vase` reference images as Stage 2B.

## Runtime Note

GPU access worked only after explicitly selecting devices:

```text
CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=7
```

Without an explicit `CUDA_VISIBLE_DEVICES`, the current Codex shell reported zero CUDA devices even though `nvidia-smi` could see the H800 GPUs.

## Artifacts

Generated LoRA weights and eval images are local artifacts ignored by git:

```text
experiments/stage2c0_residual_gating/vase/residual_gate_q75_keep05/vase/
experiments/stage2c0_residual_gating/vase/residual_gate_q90_keep05/vase/
experiments/stage2c0_residual_gating/vase/eval/residual_gate_q75_keep05/
experiments/stage2c0_residual_gating/vase/eval/residual_gate_q90_keep05/
```

Tracked metric tables:

```text
experiments/stage2c0_residual_gate_metric_audit_summary.csv
experiments/stage2c0_residual_gate_metric_audit_per_image.csv
```

## Metric Summary

Mean absolute pixel difference against the Stage 2B base and vanilla eval grids:

| run | kind | images | distance to base | distance to vanilla | diversity |
| --- | --- | ---: | ---: | ---: | ---: |
| vanilla | class | 4 | 29.156 | 0.000 | 66.501 |
| vanilla | subject | 4 | 31.144 | 0.000 | 86.785 |
| dadt_lf_late_alpha100 | class | 4 | 28.647 | 9.705 | 65.151 |
| dadt_lf_late_alpha100 | subject | 4 | 30.949 | 10.130 | 85.911 |
| dadt_lf_midlate_alpha075 | class | 4 | 28.599 | 15.752 | 66.473 |
| dadt_lf_midlate_alpha075 | subject | 4 | 30.384 | 17.584 | 83.575 |
| residual_gate_q75_keep05 | class | 4 | 21.649 | 22.366 | 59.636 |
| residual_gate_q75_keep05 | subject | 4 | 20.586 | 23.435 | 84.428 |
| residual_gate_q90_keep05 | class | 4 | 24.068 | 19.940 | 63.844 |
| residual_gate_q90_keep05 | subject | 4 | 22.528 | 20.809 | 84.744 |

## Qualitative Read

The residual-gated grids are visually stable and still produce plausible vase images. They do not collapse.

Compared with vanilla, both residual-gated runs are more conservative. They preserve ordinary vase behavior more strongly, but the subject prompt also looks less tied to the specific reference vase. The q75 gate is the strongest preservation setting and also appears most likely to underfit the subject.

## Decision

Stage 2C-0 is useful because it shows that target-level residual control can move the generation distribution much more than LF-Late mixing. However, the current gate suppresses too much of the subject-bearing residual.

Do not scale this exact gate yet.

Recommended next step:

```text
Keep residual gating, but make the gate subject-aware or timestep-aware.
The next version should preserve base-prior background/style residuals while leaving
identity-bearing subject residuals less suppressed.
```
