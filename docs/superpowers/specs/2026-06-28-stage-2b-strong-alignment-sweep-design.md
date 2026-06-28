# Stage 2B Strong Alignment Sweep Design

Date: 2026-06-28

Status: design approved for the next implementation pass. Do not treat this as a method result yet.

## Purpose

Stage 2A proved that the training and evaluation machinery works, but the current DADT-LF-Late setting is too mild:

```text
condition: dadt_lf_late
alpha: 0.5
late_timestep_threshold: 800
low_radius: 2
mid_radius: 4
```

The first 200-step `vase` comparison produced stable LoRA weights and evaluation grids, but `vanilla` and `dadt_lf_late` looked very similar. Stage 2B is a targeted stress test:

```text
Can stronger target alignment visibly reduce class-prior drift
without destroying subject personalization?
```

The goal is not to claim a paper result. The goal is to decide whether the LF-Late DADT family has enough signal to justify broader experiments.

## Plain-Language Story

Stage 2A only pulled the training target slightly toward the base model, and only at very late denoising steps. That is like barely nudging the student toward the old teacher. If the final model looks the same as vanilla DreamBooth, we cannot know whether the idea is bad or the intervention is just too weak.

Stage 2B deliberately turns the knob harder:

- pull low-frequency targets more strongly toward the pretrained model;
- optionally start that pull earlier in the denoising trajectory;
- check whether ordinary class prompts stay cleaner while `sks vase` still learns something.

If a stronger pull changes class behavior but hurts subject fidelity, that is still useful: it means the target-alignment axis is real, but needs smarter gating. If a stronger pull changes nothing, LF-Late is likely too weak as a method family.

## Sweep Matrix

Use the same `vase` reference images, prompts, SD 1.5 base model, LoRA rank, learning rate, seed, and 200 training steps as Stage 2A.

Run four conditions:

```text
vanilla
dadt_lf_late_alpha075
dadt_lf_late_alpha100
dadt_lf_midlate_alpha075
```

Condition details:

```text
vanilla:
  training target: reference target

dadt_lf_late_alpha075:
  training target: low-frequency reference/base blend
  alpha: 0.75
  late_timestep_threshold: 800

dadt_lf_late_alpha100:
  training target: low-frequency base target at late timesteps
  alpha: 1.0
  late_timestep_threshold: 800

dadt_lf_midlate_alpha075:
  training target: low-frequency reference/base blend
  alpha: 0.75
  late_timestep_threshold: 500
```

Keep:

```text
low_radius: 2
mid_radius: 4
max_train_steps: 200
lora_rank: 4
learning_rate: 0.0001
train_batch_size: 1
seed: 0
subject_id: vase
```

## Implementation Shape

Stage 2B needs run identity to be separated from method identity.

The logical method condition can remain:

```text
vanilla
dadt_lf_late
```

But each sweep variant needs a distinct output name, otherwise multiple DADT runs overwrite each other. Add a run label such as:

```text
run_name: dadt_lf_late_alpha075
```

Training outputs should become:

```text
experiments/stage2b_strong_alignment/vase/<run_name>/vase/
```

Evaluation outputs should become:

```text
experiments/stage2b_strong_alignment/vase/eval/<run_name>/
```

This keeps the method implementation simple while making the experiment reproducible.

## Evaluation

Generate the same prompt set for each run:

Subject prompts:

```text
a photo of sks vase on a wooden table
a photo of sks vase in a room
```

Class prompts:

```text
a photo of a vase
a vase on a wooden table
```

For each run, save:

```text
individual generated images
grid.png
manifest.json
```

Primary qualitative checks:

- subject prompts: does `sks vase` become more like a consistent object?
- class prompts: does ordinary `vase` remain broad and clean, or does it collapse toward the personalized subject?
- DADT variants: do they preserve class prompt diversity more than vanilla?

Lightweight metric checks:

- class drift proxy: compare class-prompt generations against the base run at the same seeds. Less drift means the class prior is more preserved.
- subject shift proxy: compare subject-prompt generations against base and vanilla at the same seeds. Stronger subject shift means personalization is doing something.
- optional CLIP/DINO proxy if available locally: image-image similarity to reference images for subject fidelity, and image-text similarity to class prompts for class consistency.

The pixel or CLIP proxies are only triage metrics. They should not be treated as final paper metrics.

## Go / No-Go Criteria

Go to broader subjects if at least one DADT variant:

```text
1. visibly preserves class-prompt generations better than vanilla;
2. still learns a recognizable subject-prompt behavior;
3. shows a measurable class-drift reduction under the same seeds.
```

Redesign rather than expand if:

```text
DADT preserves class prompts but destroys subject learning.
```

That outcome means target alignment is real but too blunt; the next design should add spatial, token, or timestep-specific gating.

No-Go / pivot if:

```text
all stronger DADT variants remain visually and metrically indistinguishable from vanilla.
```

That outcome means LF-Late target alignment is probably too weak to carry the paper, and the project should return to metric redesign or a different target-construction strategy.

## Non-Goals

Do not expand to all DreamBooth subjects in Stage 2B.

Do not add foreground/background masks yet. That is scientifically interesting, but it adds segmentation complexity before we know whether the alignment axis matters.

Do not claim Stage 2B as a paper result. It is a decision experiment.

## Expected Next Implementation Tasks

1. Add run-label support so variants do not overwrite each other.
2. Add Stage 2B configs for the four sweep runs.
3. Run the four `vase` trainings, preferably on separate available GPUs.
4. Generate evaluation grids for base and all four runs.
5. Write a Stage 2B result note with qualitative grids, metric proxies, and a Go / Redesign / No-Go decision.
