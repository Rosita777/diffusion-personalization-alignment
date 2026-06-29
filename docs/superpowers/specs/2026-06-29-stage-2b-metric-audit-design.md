# Stage 2B Metric Audit Design

Date: 2026-06-29

Status: approved for implementation after the Stage 2B weak-redesign result. This is an evaluation pass, not a new method.

## Purpose

Stage 2B showed that stronger LF-Late target alignment is visually close to vanilla DreamBooth. Before designing Stage 2C, we need a small, reproducible metric audit over the existing Stage 2B generated images.

Plainly:

```text
Do the numbers agree with our eyes that Stage 2B did not create a strong advantage?
```

This pass must not retrain models or generate new images. It only reads the already-generated evaluation manifests and PNG files.

## Inputs

Use the existing Stage 2B eval directories:

```text
experiments/stage2b_strong_alignment/vase/eval/base/
experiments/stage2b_strong_alignment/vase/eval/vanilla/
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha075/
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_late_alpha100/
experiments/stage2b_strong_alignment/vase/eval/dadt_lf_midlate_alpha075/
```

Each run contains:

```text
manifest.json
individual prompt images
grid.png
```

The manifest records `kind`, `text`, `prompt_index`, `image_index`, `seed`, and `path`. Matched-seed comparison should use those fields, not filename parsing.

## Metrics

This audit is intentionally lightweight and dependency-light.

### Class Prior Drift Proxy

For each non-base run, compare class-prompt images against the base run at matched prompt and seed:

```text
mean_abs_rgb(run_image, base_image)
```

Lower means the class prompt stayed closer to the frozen base model.

### Subject Shift Proxy

For each non-base run, compare subject-prompt images against the base run at matched prompt and seed:

```text
mean_abs_rgb(run_image, base_image)
```

This does not prove subject fidelity. It only checks whether personalization changed the subject-token generations at all.

### Difference From Vanilla

For each DADT run, compare images against the vanilla run at matched prompt and seed:

```text
mean_abs_rgb(run_image, vanilla_image)
```

If this is very small, the DADT variant is behaviorally close to vanilla under the sampled prompts.

### Diversity Proxy

For each run and prompt kind, compute pairwise mean absolute RGB distance among generated images:

```text
mean_abs_rgb(image_i, image_j), i < j
```

This is a crude diversity proxy. It should be interpreted only within the same prompt set.

### Optional Reference Histogram Proxy

If reference images are provided, compute average color-histogram cosine similarity between generated subject images and reference images. This is a weak proxy for material/color resemblance, not an identity metric.

The first implementation may omit this metric if it adds complexity. The audit should not pretend this replaces CLIP-I or DINO.

## Outputs

Write small, trackable summaries:

```text
experiments/stage2b_metric_audit_summary.csv
experiments/stage2b_metric_audit_per_image.csv
```

Update:

```text
experiments/stage2b_strong_alignment_results.md
```

with a short metric-audit section and a clear decision.

Generated images, logs, LoRA weights, and raw large artifacts remain ignored by git.

## Decision Logic

If DADT variants have clearly lower class-prior drift than vanilla while keeping subject shift comparable, Stage 2C can build on DADT.

If DADT variants are close to vanilla on both class drift and subject shift, Stage 2C should not be another fixed alpha/timestep rule. It should move to measurement-driven residual gating or spatial/prompt-paired target construction.

If DADT improves class drift only by removing subject shift, then it is merely weakening personalization. That needs to be treated as a failed tradeoff, not a win.

## Non-Goals

Do not run new training.

Do not generate new diffusion images.

Do not download CLIP, DINO, or FID models in this pass.

Do not claim paper-level metrics from pixel proxies.

Do not read or use any token file.

## Expected Implementation

Add a focused script:

```text
scripts/personalization_training/audit_eval_metrics.py
```

The script should:

1. load a base manifest and one or more run manifests;
2. match records by `kind`, `text`, `prompt_index`, `image_index`, and `seed`;
3. compute mean absolute RGB differences;
4. write per-image and summary CSV files;
5. expose a CLI with `--base-dir`, `--run-dir`, `--output-summary`, and `--output-per-image`.

Add tests:

```text
tests/personalization_training/test_audit_eval_metrics.py
```

The tests should use tiny generated PNGs and manifests so they run without GPUs or diffusion dependencies.
