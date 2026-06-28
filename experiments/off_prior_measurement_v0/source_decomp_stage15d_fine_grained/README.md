# Stage 1.5D Fine-Grained Gap Diagnosis

Date: 2026-06-27

Status: executed. This is an offline diagnosis run, not a personalization training run.

## Purpose

Stage 1.5D asks whether the current average scalar metric is hiding a weaker DreamBooth-specific signal in finer axes:

```text
timestep
frequency band
class
```

The run reuses existing Stage 1.5A/B raw metrics. It does not rerun Stable Diffusion or train any personalization model.

## Inputs

```text
experiments/off_prior_measurement_v0/source_decomp_stage15a/measurements/raw_source_decomp_metrics.csv
experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/measurements/raw_source_decomp_metrics.csv
```

Stage 1.5A is the main file because it contains DreamBooth references. Stage 1.5B is used only as a prompt-matched ordinary-real control because it has no DreamBooth reference rows.

## Commands

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.source_decomp_fine_grained \
  --raw-metrics experiments/off_prior_measurement_v0/source_decomp_stage15a/measurements/raw_source_decomp_metrics.csv \
  --output-dir experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained \
  --label stage15a_class

/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.source_decomp_fine_grained \
  --raw-metrics experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/measurements/raw_source_decomp_metrics.csv \
  --output-dir experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained \
  --label stage15b_prompt_matched
```

## Clean-Norm Timestep Result

This is the direct follow-up to the failed average `clean_norm` metric.

```text
timestep  positive DreamBooth classes  mean subject-specific gap
50        1 / 4                        -0.015660
200       0 / 4                        -0.013493
500       0 / 4                        -0.010486
800       0 / 4                        -0.002869
999       1 / 4                        -0.000809
```

Interpretation: splitting by timestep does not rescue the clean scalar metric. DreamBooth references are still usually below ordinary real controls.

## Frequency Result

The frequency view is more interesting:

```text
frequency band  positive cells  total cells  mean subject-specific gap
high            0               20           -0.002032
low             12              20            0.001171
mid             5               20           -0.001377
```

Low frequency is the only band with a positive mean DreamBooth-over-ordinary-real gap. High frequency is consistently negative.

The strongest positive pockets are:

```text
class  timestep  band  subject-specific gap
vase   50        low   0.020408
vase   200       low   0.010732
vase   50        mid   0.003508
vase   500       low   0.003040
dog    50        low   0.001386
```

These early large positives are mostly driven by `vase`, so they are not robust enough to revive the original average-gap story.

## Late Low-Frequency Pattern

The late low-frequency cells are small but more class-consistent:

```text
timestep  band  positive DreamBooth classes  mean subject-specific gap
800       low   4 / 4                        0.000453
999       low   3 / 4                        0.000087
```

This is the most useful finding from Stage 1.5D. In plain language:

```text
The total clean score does not show DreamBooth-specific off-priorness,
but a tiny low-frequency late-timestep signal may exist.
```

This should be treated as a weak clue, not as a paper-ready result.

## Prompt-Matched Control Check

Stage 1.5B remains a generic real-domain control. Its prompt-matched real-domain gap is positive at all timesteps:

```text
timestep  mean real-domain gap  positive classes
50        0.031958              4 / 4
200       0.018146              4 / 4
500       0.010193              4 / 4
800       0.003385              4 / 4
999       0.001094              4 / 4
```

This means the real-image-versus-base-generated gap is still broad and robust. Any DreamBooth-specific claim must separate itself from this generic real-image gap.

## Decision

Stage 1.5D does not produce a Go for Stage 2 personalization training.

The original scalar metric remains weak because:

```text
clean_norm by timestep is still negative for almost all classes;
high-frequency and mid-frequency subject-specific gaps are negative on average;
large positive low-frequency pockets are class-concentrated, especially in vase.
```

However, Stage 1.5D gives a better next direction than Stage 1.5C:

```text
Focus the next metric redesign on low-frequency, late-timestep, vector-structured behavior.
```

The next experiment should not be another average scalar-gap run. It should test whether this low-frequency late-timestep component corresponds to subject/background binding, layout drift, or a genuine personalization target component.
