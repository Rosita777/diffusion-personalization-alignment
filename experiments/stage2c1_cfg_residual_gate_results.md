# Stage 2C-1 CFG Residual Gate Results

Date: 2026-06-30

Status: completed triage run and alpha sweep on `vase`; completed second-subject check on `backpack`.

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

Initial config:

```text
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha05.yaml
alpha: 0.5
subject: vase
max_train_steps: 200
LoRA rank: 4
base model: SD 1.5
```

Sweep configs:

```text
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha025.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha05.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha060.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha065.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha070.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha075.yaml
configs/stage2c1_cfg_residual_gate/vase_cfg_gate_alpha100.yaml
configs/stage2c1_cfg_residual_gate/backpack_vanilla.yaml
configs/stage2c1_cfg_residual_gate/backpack_cfg_gate_alpha05.yaml
configs/stage2c1_cfg_residual_gate/backpack_cfg_gate_alpha065.yaml
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

Initial training summary:

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

Alpha sweep training summaries:

| run | loss_first | loss_last | loss_mean |
| --- | ---: | ---: | ---: |
| cfg_gate_alpha025 | 0.00935 | 0.00700 | 0.08245 |
| cfg_gate_alpha05 | 0.00835 | 0.00629 | 0.07452 |
| cfg_gate_alpha060 | 0.00800 | 0.00610 | 0.07179 |
| cfg_gate_alpha065 | 0.00784 | 0.00598 | 0.07051 |
| cfg_gate_alpha070 | 0.00769 | 0.00591 | 0.06929 |
| cfg_gate_alpha075 | 0.00755 | 0.00578 | 0.06814 |
| cfg_gate_alpha100 | 0.00696 | 0.00541 | 0.06334 |

Backpack training summaries:

| run | condition | loss_first | loss_last | loss_mean |
| --- | --- | ---: | ---: | ---: |
| vanilla | vanilla | 0.03601 | 0.02623 | 0.11996 |
| cfg_gate_alpha05 | dadt_cfg_residual_gate | 0.02821 | 0.02187 | 0.09647 |
| cfg_gate_alpha065 | dadt_cfg_residual_gate | 0.02646 | 0.02089 | 0.09109 |

## Metric Audit

Summary CSV:

```text
experiments/stage2c1_cfg_residual_gate_metric_audit_summary.csv
experiments/stage2c1_cfg_residual_gate_alpha_sweep_summary.csv
experiments/stage2c1_cfg_residual_gate_alpha_fine_sweep_summary.csv
experiments/stage2c1_cfg_residual_gate_backpack_summary.csv
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
| cfg_gate_alpha025 | class | 28.378 |
| cfg_gate_alpha025 | subject | 30.666 |
| cfg_gate_alpha05 | class | 26.386 |
| cfg_gate_alpha05 | subject | 31.776 |
| cfg_gate_alpha060 | class | 19.352 |
| cfg_gate_alpha060 | subject | 31.679 |
| cfg_gate_alpha065 | class | 17.756 |
| cfg_gate_alpha065 | subject | 31.275 |
| cfg_gate_alpha070 | class | 26.337 |
| cfg_gate_alpha070 | subject | 30.230 |
| cfg_gate_alpha075 | class | 22.674 |
| cfg_gate_alpha075 | subject | 32.371 |
| cfg_gate_alpha100 | class | 27.978 |
| cfg_gate_alpha100 | subject | 32.511 |

Backpack second-subject check:

| run | kind | mean distance to base |
| --- | --- | ---: |
| vanilla | class | 51.206 |
| vanilla | subject | 43.764 |
| cfg_gate_alpha05 | class | 46.257 |
| cfg_gate_alpha05 | subject | 44.069 |
| cfg_gate_alpha065 | class | 45.260 |
| cfg_gate_alpha065 | subject | 43.495 |

## Visual Read

Generated grid:

```text
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha025/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha05/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha060/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha065/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha070/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha075/grid.png
experiments/stage2c1_cfg_residual_gate/vase/eval/cfg_gate_alpha100/grid.png
experiments/stage2c1_cfg_residual_gate/backpack/eval/base/grid.png
experiments/stage2c1_cfg_residual_gate/backpack/eval/vanilla/grid.png
experiments/stage2c1_cfg_residual_gate/backpack/eval/cfg_gate_alpha05/grid.png
experiments/stage2c1_cfg_residual_gate/backpack/eval/cfg_gate_alpha065/grid.png
```

The CFG gate result keeps subject-prompt images closer to vanilla subject strength than the q75/q90 residual magnitude gates. It still reduces class-prompt drift compared with vanilla, although less aggressively than q75/q90.

The sweep is not monotonic. `alpha=0.25` is too conservative and leaves class drift close to vanilla. `alpha=0.75` gives the strongest class-drift reduction among CFG gate runs while keeping subject distance high, but one subject image becomes noticeably softer/watercolor-like. `alpha=1.0` does not improve class preservation further and appears less stable. This suggests that the gate is useful, but over-suppression can distort the denoising path rather than simply preserve the base prior.

Fine sweep around `0.5-0.75` found a sharper tradeoff. `alpha=0.60` and `alpha=0.65` produce the lowest class distance to base so far (`19.352` and `17.756`), while subject distance remains around vanilla. However, both settings show a soft/watercolor failure on the first subject prompt. `alpha=0.70` visually recovers some sharpness but loses much of the class-preservation gain. Because the eval set has only 4 class and 4 subject images, this non-monotonicity should be treated as a strong triage signal, not a final tuning conclusion.

Backpack gives a weaker but consistent generalization signal. CFG gate reduces class distance to base from `51.206` to `46.257/45.260`, while subject distance stays close to vanilla (`43.764` versus `44.069/43.495`). Visually, vanilla transfers a strong dark/patch-heavy backpack style into class prompts. CFG gate softens that transfer without removing backpack identity from subject prompts. Unlike vase, `alpha=0.65` did not show an obvious watercolor failure on this small backpack grid.

## Takeaway

Stage 2C-1 is more promising than Stage 2C-0 as a method prototype:

1. It reduces class drift versus vanilla: best fine-sweep point `29.156 -> 17.756`.
2. It avoids the main Stage 2C-0 failure, where subject learning collapsed toward the base model: q75/q90 subject distance `20.586/22.528`, CFG gate `31.776`.
3. It supports the paper story better than pure magnitude gating because the gate uses a base-model semantic direction, not just residual size.
4. Current best metric point is `alpha=0.65`; current safer visual-quality point is still `alpha=0.5`, with `alpha=0.70` as a middle candidate.
5. Backpack supports the same direction: class drift decreases while subject strength is not obviously sacrificed.

This is not yet a final result. The next useful checks are a larger prompt/seed eval set, at least one animal subject, and a stronger subject-fidelity metric such as DINO or CLIP image similarity to reference images.
