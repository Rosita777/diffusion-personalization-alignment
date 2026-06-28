# Stage 2A LF-Late Training Validation Design

Date: 2026-06-28

Status: real LoRA training loop wired after Stage 1.5D. The LF-Late target utilities, config validation, subject-filtered dry-run entrypoints, and one-step `vase` smoke jobs for `vanilla` / `dadt_lf_late` are complete; the full 200-step comparison has not been executed yet.

## Purpose

Stage 1.5A through Stage 1.5D did not support the broad claim that DreamBooth reference targets are globally more off-prior than ordinary real images. The only useful clue is narrower:

```text
DreamBooth references show a tiny, more class-consistent positive signal
in low-frequency late-timestep components.
```

Stage 2A tests whether that clue has training value.

The question is deliberately small:

```text
If we align only low-frequency late-timestep personalization targets
toward the pretrained model, does prior drift decrease without killing
subject fidelity?
```

This is a smoke validation, not a paper-scale experiment.

## Plain-Language Hypothesis

The original story was too broad:

```text
Reference images are globally off-prior.
```

The revised Stage 2A story is more precise:

```text
Some reference-image supervision components are more likely to hurt the
pretrained prior than to teach subject identity.
```

The first component to test is:

```text
low frequency + late timestep
```

The intuition is:

- low frequency can affect layout, shape, coarse color, and background binding;
- late timesteps in the current measurement showed the most class-consistent weak positive clue;
- high frequency should be preserved more because it may carry subject texture and identity.

## Subjects

Use two smoke subjects first:

```text
vase
dog
```

Rationale:

- `vase` produced the strongest Stage 1.5D low-frequency pockets, so it is the best chance to see whether the signal has value.
- `dog` is a classic personalization class and is easier to explain to reviewers.

Do not expand to all DreamBooth subjects until Stage 2A passes.

## Training Conditions

Stage 2A compares two conditions:

```text
vanilla_lora_dreambooth
dadt_lf_late_lora_dreambooth
```

Both should use the same:

- base model;
- subject reference images;
- class prior images or class prompts;
- LoRA rank;
- optimizer;
- learning rate;
- random seed;
- number of training steps;
- validation prompts.

The only intended difference is target construction on selected timesteps and frequency components.

## Target Correction

Standard diffusion personalization trains on:

```text
loss = || model(z_t, t, subject_prompt) - v_ref ||^2
```

Stage 2A changes the target only for selected components:

```text
v_target = v_ref
```

For late timesteps and low-frequency components:

```text
low(v_target) = (1 - alpha) * low(v_ref) + alpha * low(v_base)
```

where:

```text
v_base = frozen_pretrained_model(z_t, t, class_prompt)
```

Default smoke settings:

```text
late timestep rule: scheduler timesteps corresponding to the low-noise end
frequency band: latent DCT low band
alpha: 0.25 and 0.50 if budget allows; otherwise alpha = 0.50 first
```

All mid-frequency and high-frequency components stay unchanged in the first smoke test:

```text
mid(v_target) = mid(v_ref)
high(v_target) = high(v_ref)
```

This is not generic distillation because:

- it does not replace all targets with the base model prediction;
- it applies only to a measured weak-risk component;
- it preserves the subject target outside that component.

## Required Baselines And Controls

Minimum:

```text
vanilla_lora_dreambooth
dadt_lf_late_lora_dreambooth
```

If the smoke signal is positive, add:

```text
timestep_weight_only_lora
uniform_target_mix_lora
```

These later controls separate the proposed target editing from simple timestep weighting or ordinary distillation.

## Evaluation

Stage 2A should report generated images and lightweight metrics.

### Subject Fidelity

Use subject prompts such as:

```text
a photo of sks dog
a photo of sks dog in a park
a photo of sks vase on a wooden table
a photo of sks vase in a room
```

Evaluate:

- visual subject similarity by inspection first;
- CLIP image similarity to reference images if already available or cheap to add;
- failure notes such as identity loss, texture loss, or underfitting.

### Prior Preservation

Use class prompts without the rare token:

```text
a photo of a dog
a dog in a park
a photo of a vase
a vase on a wooden table
```

Evaluate:

- whether class generations collapse toward the personalized subject;
- whether background or layout copies reference images;
- CLIP text-image score if cheap;
- image diversity and qualitative prior drift notes.

### Base Comparison

Generate the same prompts from the frozen base model for side-by-side comparison. This prevents over-interpreting poor generations caused by prompts or seeds.

## Go / No-Go Criteria

Stage 2A is Go only if:

```text
dadt_lf_late_lora reduces visible prior drift
and subject fidelity is not clearly worse than vanilla LoRA.
```

Concrete smoke rule:

- for at least one of the two subjects, DADT-LF-Late has visibly less class-prompt collapse than vanilla LoRA;
- for neither subject does DADT-LF-Late obviously destroy subject identity;
- the result is not explained by fewer effective training steps or a broken loss scale.

Stage 2A is No-Go if:

- DADT-LF-Late and vanilla look the same;
- DADT-LF-Late loses subject identity;
- DADT-LF-Late only improves by underfitting;
- training is too unstable to compare.

## Implementation Shape

Add a new training branch under:

```text
scripts/personalization_training/
configs/stage2a_lf_late/
experiments/stage2a_lf_late/
tests/personalization_training/
```

The implementation should be small and testable:

```text
dct_target.py
```

Implements low/mid/high DCT decomposition and reconstruction for latent tensors.

```text
target_alignment.py
```

Implements the LF-Late target editing rule.

```text
train_lora_dreambooth.py
```

Runs vanilla and DADT-LF-Late LoRA DreamBooth smoke training.

```text
generate_eval_grid.py
```

Generates subject and class prompt grids for visual comparison.

```text
write_stage2a_report.py
```

Writes a concise result README with commands, settings, images, and decision.

## Testing Requirements

Unit tests should run without loading Stable Diffusion:

- DCT split and reconstruction preserve tensor shape and approximately reconstruct the input.
- LF-Late correction changes only low-frequency components when the timestep is selected.
- LF-Late correction leaves the target unchanged when the timestep is not selected.
- Alpha `0.0` returns the reference target.
- Alpha `1.0` replaces only the selected low-frequency component with the base prediction.
- Config parsing rejects missing subject image paths and invalid alpha values.

GPU training is not a unit test requirement.

## Project Hygiene

Do not commit:

```text
token.txt
model checkpoints
LoRA weight files
raw generated image grids if they are large
raw dataset caches
```

Commit:

```text
configs
small scripts
tests
curated result README
small CSV summaries
small contact sheets if file size is reasonable
```

If generated images or LoRA weights are too large, record their local paths and exact commands in the result README.

## Expected Outcomes

Positive smoke outcome:

```text
The DADT story survives in a narrower form:
selective target alignment on measured low-frequency late-timestep components
can reduce prior drift while preserving subject learning.
```

Negative smoke outcome:

```text
The personalization-specific target-correction story is probably too weak.
Pivot toward real-image-to-diffusion-prior target alignment, or stop this branch.
```

Either outcome is useful because Stage 2A is designed to be small, decisive, and hard to overfit.
