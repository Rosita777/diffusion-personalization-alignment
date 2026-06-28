# Stage 1.5C Metric Ablation

Date: 2026-06-27

Status: executed. This is a diagnosis run, not a personalization training run.

## Purpose

Stage 1.5B showed that prompt matching only slightly reduces the projection-cleaned ordinary-real gap. Stage 1.5C asks a narrower question:

```text
Is the remaining gap coming from the metric's artifact/projection component,
or from the projection-cleaned residual itself?
```

The run reuses existing raw measurements and does not rerun SD 1.5 inference.

## Inputs

Stage 1.5C reads:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15a/measurements/raw_source_decomp_metrics.csv
experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/measurements/raw_source_decomp_metrics.csv
```

Stage 1.5A uses class-only conditioning and includes base-generated controls, ordinary real controls, and DreamBooth references. Stage 1.5B uses prompt-matched conditioning and includes base-generated controls plus ordinary real controls only.

## Metrics

The ablation compares three scalar views of the same residual:

```text
raw_norm
projected_artifact_norm = raw_norm * artifact_fraction
clean_norm
```

`raw_norm` is the total measured target gap. `projected_artifact_norm` estimates the part aligned with the VAE roundtrip/projection artifact direction. `clean_norm` is the projection-cleaned score used by the Stage 1.4 and Stage 1.5A/B conclusions.

## Commands

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.source_decomp_metric_ablation \
  --raw-metrics experiments/off_prior_measurement_v0/source_decomp_stage15a/measurements/raw_source_decomp_metrics.csv \
  --output-dir experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation \
  --label stage15a_class

/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.source_decomp_metric_ablation \
  --raw-metrics experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/measurements/raw_source_decomp_metrics.csv \
  --output-dir experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation \
  --label stage15b_prompt_matched

/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.source_decomp_metric_ablation \
  --output-dir experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation \
  --combine-gap-summaries \
  experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation/summaries/metric_gap_summary_stage15a_class.csv \
  experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation/summaries/metric_gap_summary_stage15b_prompt_matched.csv
```

## Mean Gap Comparison

```text
experiment_label          metric                   real_domain_gap  subject_specific_gap
stage15a_class            raw_norm                        0.039491             -0.033744
stage15a_class            projected_artifact_norm          0.037432             -0.032819
stage15a_class            clean_norm                      0.013899             -0.008663
stage15b_prompt_matched   raw_norm                        0.021300                   n/a
stage15b_prompt_matched   projected_artifact_norm          0.019200                   n/a
stage15b_prompt_matched   clean_norm                      0.012955                   n/a
```

Prompt matching reduces the mean raw real-domain gap from `0.039491` to `0.021300`, about a 46.1% drop. It reduces the projected artifact gap from `0.037432` to `0.019200`, about a 48.7% drop. But it reduces the clean gap only from `0.013899` to `0.012955`, about a 6.8% drop.

## Interpretation

The simple story is:

```text
Prompt mismatch mostly lives in the raw / artifact-aligned direction.
After projection cleaning, the ordinary-real gap mostly remains.
```

This means coarse class prompts are not the main reason Stage 1.5A failed.

The harder problem is that DreamBooth references are still not more off-prior than ordinary real controls under any of these scalar views in Stage 1.5A:

```text
raw subject-specific gap:                -0.033744
projected artifact subject-specific gap: -0.032819
clean subject-specific gap:              -0.008663
```

So the current scalar residual family does not support the original claim that DreamBooth reference images have an extra personalization-specific target gap beyond ordinary real images.

## Decision

Do not start Stage 2 personalization fine-tuning from this metric.

The next step should be one of:

```text
1. Redesign the evidence around trajectory-level or vector-structured behavior, not only scalar residual norms.
2. Add a stronger VAE/projection calibration baseline before making personalization claims.
3. Pivot the paper story from personalization-specific target off-priorness to real-image-to-diffusion-prior target alignment.
```

The current evidence is useful because it localizes a failure mode: prompt mismatch explains a large part of the raw/artifact gap, but the cleaned scalar gap behaves like a generic real-image/projection-domain gap rather than a DreamBooth-specific signal.
