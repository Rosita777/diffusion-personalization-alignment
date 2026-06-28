# Stage 2A LF-Late Training Validation

Date: 2026-06-28

Status: real LoRA training loop wired. One-step and 200-step `vase` jobs have executed for both `vanilla` and `dadt_lf_late`; initial evaluation grids have been generated. This is still a training-validation result, not a method win.

## Purpose

Stage 2A tests whether the weak Stage 1.5D clue has training value:

```text
Can low-frequency late-timestep target alignment reduce prior drift
without visibly hurting subject fidelity?
```

The current implementation includes:

- latent DCT frequency utilities;
- LF-Late target alignment utilities;
- Stage 2A YAML config validation;
- smoke config for `vase` and `dog`;
- lightweight training/evaluation/report entrypoints with `--dry-run`;
- real diffusers UNet LoRA training with subject filtering and condition-specific output directories.

The current runs validate that the training chain loads SD 1.5, attaches LoRA, computes the vanilla/DADT targets, backpropagates stably, saves LoRA weights, and reloads them for qualitative evaluation. The first 200-step `vase` comparison shows that the method is runnable, but the qualitative vanilla/DADT gap is weak.

## Smoke Config

```text
configs/stage2a_lf_late/smoke_vase_dog.yaml
```

The config uses locally cached DreamBooth reference images:

```text
data/cache/off_prior_measurement_v0/dreambooth_dataset/dataset/vase/
data/cache/off_prior_measurement_v0/dreambooth_dataset/dataset/dog/
```

The default model path is the local SD 1.5 cache:

```text
data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
```

## Dry-Run Commands

Validate vanilla LoRA setup:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition vanilla \
  --subject-id vase \
  --dry-run
```

Validate DADT-LF-Late setup:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition dadt_lf_late \
  --subject-id vase \
  --dry-run
```

## Training Runs

Executed on 2026-06-28 to validate runtime wiring:

```bash
CUDA_VISIBLE_DEVICES=4 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition vanilla \
  --subject-id vase \
  --max-train-steps 1

CUDA_VISIBLE_DEVICES=4 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition dadt_lf_late \
  --subject-id vase \
  --max-train-steps 1
```

Both commands saved local LoRA artifacts under:

```text
experiments/stage2a_lf_late/smoke_vase_dog/vanilla/vase/
experiments/stage2a_lf_late/smoke_vase_dog/dadt_lf_late/vase/
```

The LoRA weight files and generated `training_summary.json` files are local generated artifacts and are ignored by git. Re-run the commands above if those smoke artifacts are needed.

The first full `vase` comparison was executed on 2026-06-28:

```bash
CUDA_VISIBLE_DEVICES=5 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition vanilla \
  --subject-id vase

CUDA_VISIBLE_DEVICES=0 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.train_lora_dreambooth \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --condition dadt_lf_late \
  --subject-id vase
```

Observed summaries:

```text
vanilla:     loss_first=0.0105663780, loss_last=0.0078691514, loss_mean=0.0919447121
dadt_lf_late: loss_first=0.0105663780, loss_last=0.0078699570, loss_mean=0.0919028508
```

Important runtime fix: LoRA trainable parameters must be cast to float32 after attaching adapters. Keeping trainable LoRA parameters in fp16 caused NaN loss after a few steps. The training loop now casts only trainable parameters to float32 and fails fast on non-finite loss.

## Evaluation Grids

Generated on 2026-06-28:

```bash
CUDA_VISIBLE_DEVICES=1 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --subject-id vase \
  --output-dir experiments/stage2a_lf_late/eval_vase/base

CUDA_VISIBLE_DEVICES=1 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2a_lf_late/smoke_vase_dog/vanilla/vase \
  --output-dir experiments/stage2a_lf_late/eval_vase/vanilla

CUDA_VISIBLE_DEVICES=2 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.generate_eval_grid \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --subject-id vase \
  --weights-dir experiments/stage2a_lf_late/smoke_vase_dog/dadt_lf_late/vase \
  --output-dir experiments/stage2a_lf_late/eval_vase/dadt_lf_late
```

The eval grid directory is ignored by git because it contains generated images and manifests. Re-run the commands above to reproduce it.

Loading note: training saves UNet LoRA adapters as `pytorch_lora_weights.safetensors`. Evaluation must load them through `pipe.unet.load_lora_adapter(..., prefix=None, weight_name="pytorch_lora_weights.safetensors")`; direct `pipe.load_lora_weights(...)` silently misses the UNet keys in this setup. The local environment also has an incompatible AutoAWQ install, so evaluation disables the AWQ LoRA dispatcher before loading adapters, matching the training loop workaround.

Qualitative observation from the first 200-step grids:

```text
base: ordinary vase images, no subject identity because sks is unknown.
vanilla: visibly changes the subject-prompt generations and also shifts class-prompt generations.
dadt_lf_late: very close to vanilla under the current mild alpha=0.5 / late-threshold setup.
```

This supports "the Stage 2A machinery is runnable" but not "DADT-LF-Late is better." The next scientific step is to strengthen or redesign the alignment condition before expanding to more subjects.

Write a pending report:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.personalization_training.write_stage2a_report \
  --config configs/stage2a_lf_late/smoke_vase_dog.yaml \
  --output experiments/stage2a_lf_late/README.generated.md
```

## Current Decision

Do not interpret Stage 2A as a method result yet. The current runs confirm that:

```text
the LF-Late target edit is testable;
the smoke config resolves local subject image paths;
the entrypoints can validate planned runs;
the training loop can load Stable Diffusion, train LoRA stably, and save weights;
the evaluation script can reload the saved LoRA weights and produce reproducible grids.
```

Next step: run a targeted Stage 2B design pass. Candidate changes are stronger alpha / timestep schedules, prompt-paired class preservation during eval, and a metric that separately tracks subject prompt fidelity and class prompt drift before expanding to `dog`.
