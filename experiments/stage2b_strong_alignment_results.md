# Stage 2B Strong Alignment Sweep Results

Date: 2026-06-28

Status: completed for the `vase` subject. This is a decision experiment, not a method win.

## Purpose

Stage 2B tested whether stronger low-frequency late-timestep target alignment can visibly reduce class-prior drift while keeping subject personalization alive.

Stage 2A used a mild setting:

```text
condition: dadt_lf_late
alpha: 0.5
late_timestep_threshold: 800
```

Stage 2B increased the intervention strength and compared four `vase` runs:

```text
vanilla
dadt_lf_late_alpha075
dadt_lf_late_alpha100
dadt_lf_midlate_alpha075
```

All runs used SD 1.5, LoRA rank 4, learning rate `0.0001`, 200 steps, seed 0, and the same three `vase` reference images.

## Configs

```text
configs/stage2b_strong_alignment/vase_vanilla.yaml
configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha075.yaml
configs/stage2b_strong_alignment/vase_dadt_lf_late_alpha100.yaml
configs/stage2b_strong_alignment/vase_dadt_lf_midlate_alpha075.yaml
```

The generated LoRA weights, images, manifests, and logs are local artifacts and are ignored by git:

```text
experiments/stage2b_strong_alignment/
experiments/stage2b_vase_*.log
experiments/stage2b_eval_*.log
```

## Training Summary

All four runs completed and saved `pytorch_lora_weights.safetensors`.

| run | loss_first | loss_last | loss_mean |
| --- | ---: | ---: | ---: |
| vanilla | 0.0105663780 | 0.0078691514 | 0.0919447121 |
| dadt_lf_late_alpha075 | 0.0105663780 | 0.0078839818 | 0.0918944036 |
| dadt_lf_late_alpha100 | 0.0105663780 | 0.0078881448 | 0.0918830776 |
| dadt_lf_midlate_alpha075 | 0.0103434660 | 0.0075057158 | 0.0917546061 |

No `Traceback`, non-finite loss, or LoRA-loading error was found in the Stage 2B train/eval logs.

## Evaluation Artifacts

Each eval run generated 8 images and `grid.png`:

```text
experiments/stage2b_strong_alignment/vase/eval/base/grid.png
experiments/stage2b_strong_alignment/vase/eval/vanilla/grid.png
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha075/grid.png
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha100/grid.png
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_midlate_alpha075/grid.png
```

Prompt set:

```text
subject: a photo of sks vase on a wooden table
subject: a photo of sks vase in a room
class:   a photo of a vase
class:   a vase on a wooden table
```

## Lightweight Drift Proxy

Mean absolute pixel difference against the base run at matched seeds:

| run | subject prompts | class prompts |
| --- | ---: | ---: |
| vanilla | 31.144 | 29.156 |
| dadt_lf_late_alpha075 | 31.613 | 29.770 |
| dadt_lf_late_alpha100 | 30.949 | 28.647 |
| dadt_lf_midlate_alpha075 | 30.384 | 28.599 |

This proxy is only for triage. It is not a paper metric. The class-prompt drift is slightly lower for `alpha100` and `midlate_alpha075`, but the gap is small.

## Qualitative Observation

Base generates ordinary `vase` images with broad class behavior.

Vanilla learns visible personalized cues: darker ceramic/wood-like materials and stronger subject-specific shapes. It also shifts class prompts toward related colors, materials, and compositions.

The stronger DADT variants still learn subject-prompt behavior, so the target edit does not break training. However, they do not clearly preserve the class prompts better than vanilla. `alpha100` and `midlate_alpha075` are marginally less drifted by the pixel proxy, but the grids remain visually close to vanilla.

## Decision

Stage 2B is a weak redesign signal, not a go signal.

What we can claim internally:

```text
1. The stronger DADT sweep is runnable and stable.
2. Personalization-induced class drift is visible in the grids.
3. Simple LF-Late low-frequency target alignment does not yet produce a strong qualitative advantage.
```

What we should not claim:

```text
DADT-LF-Late clearly beats vanilla DreamBooth.
```

Recommended next step:

```text
Do not expand this exact LF-Late recipe to all subjects yet.
Use Stage 2B as evidence that the target-construction axis is testable,
but redesign the intervention before spending more GPU time.
```

Promising redesign directions:

```text
1. Spatial gating: preserve background/class regions more, let subject regions learn more.
2. Prompt-paired target alignment: compare subject prompt and class prompt targets under matched noise.
3. Timestep schedule redesign: decide preservation strength from measured off-priorness instead of a fixed threshold.
4. Better metric: separate subject fidelity, class prior preservation, and prompt consistency with CLIP/DINO-style proxies.
```
