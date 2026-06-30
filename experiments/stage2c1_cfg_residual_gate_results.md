# Stage 2C-1 CFG Residual Gate Results

Date: 2026-06-30

Status: completed triage run on `vase`.

## Question

Stage 2C-0 showed that residual magnitude gating can reduce class-prompt drift, but it also weakened subject learning. Stage 2C-1 tests whether a CFG-direction gate can suppress residuals that look like generic class behavior while preserving residuals less explained by the class prompt.

## Method

Condition:

```text
dadt_cfg_residual_gate
```

Core target:

```text
r = v_ref - v_class
d_class = v_class - v_null
gate = 1 - alpha * relu(cosine_channel(r, d_class))
v_target = v_class + gate * r
```

Config:

```text
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha05.yaml
alpha: 0.5
subject: vase
max_train_steps: 200
LoRA rank: 4
base model: SD 1.5
```

## Diagnostic

Command output saved to:

```text
experiments/stage2c1_cfg_residual_gate/vase/cfg_gate_alpha05_diagnostic_t500.json
```

At timestep 500 over 3 reference batches:

| metric | value |
| --- | ---: |
| cosine mean | 0.0156 |
| cosine std | 0.5515 |
| cosine min | -0.9997 |
| cosine max | 0.9971 |
| positive ratio | 0.5081 |

Interpretation: the cosine signal is not degenerate. Roughly half the residual map is positively aligned with the class CFG direction and therefore suppressible; the rest is preserved.

## Training

Training summary:

```text
experiments/stage2c1_cfg_residual_gate/vase/cfg_gate_alpha05/vase/training_summary.json
```

Key values:

| metric | value |
| --- | ---: |
| loss_first | 0.00835 |
| loss_last | 0.00629 |
| loss_mean | 0.07452 |
| max_train_steps | 200 |

## Metric Audit

Summary CSV:

```text
experiments/stage2c1_cfg_residual_gate_metric_audit_summary.csv
```

The lower `class` distance-to-base is better for anti-forgetting. For `subject`, higher distance-to-base than Stage 2C-0 is useful here because Stage 2C-0 over-preserved the base model and weakened subject learning.

| run | kind | mean distance to base |
| --- | --- | ---: |
| vanilla | class | 29.156 |
| vanilla | subject | 31.144 |
| residual_gate_q75_keep05 | class | 21.649 |
| residual_gate_q75_keep05 | subject | 20.586 |
| residual_gate_q90_keep05 | class | 24.068 |
| residual_gate_q90_keep05 | subject | 22.528 |
| cfg_gate_alpha05 | class | 26.386 |
| cfg_gate_alpha05 | subject | 31.776 |

## Visual Read

Generated grid:

```text
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha05/grid.png
```

The CFG gate result keeps subject-prompt images closer to vanilla subject strength than the q75/q90 residual magnitude gates. It still reduces class-prompt drift compared with vanilla, although less aggressively than q75/q90.

## Takeaway

Stage 2C-1 is more promising than Stage 2C-0 as a method prototype:

1. It reduces class drift versus vanilla: `29.156 -> 26.386`.
2. It avoids the main Stage 2C-0 failure, where subject learning collapsed toward the base model: q75/q90 subject distance `20.586/22.528`, CFG gate `31.776`.
3. It supports the paper story better than pure magnitude gating because the gate uses a base-model semantic direction, not just residual size.

This is not yet a final result. The next useful checks are an `alpha` sweep, another subject class, and a stronger subject-fidelity metric such as DINO or CLIP image similarity to reference images.
