# Stage 1.5D Fine-Grained Gap Diagnosis Design

Date: 2026-06-27

Status: executed. This offline diagnosis reused existing Stage 1.5A/B raw metrics and found a weak low-frequency late-timestep clue, but no Go signal for Stage 2 personalization training.

## Purpose

Stage 1.5C showed that the current average scalar gap does not support a personalization-specific off-priorness claim. Stage 1.5D checks whether a weaker but useful signal is hidden inside finer axes:

```text
timestep
frequency band
class
source group
```

The goal is not to train a model. The goal is to decide whether the current metric has any structured DreamBooth-specific signal worth rescuing.

## Inputs

Use existing raw measurement CSV files:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15a/measurements/raw_source_decomp_metrics.csv
experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/measurements/raw_source_decomp_metrics.csv
```

Stage 1.5A is the main file because it contains DreamBooth references. Stage 1.5B is used only as a prompt-matched real-domain control because it does not contain DreamBooth references.

## Outputs

Write outputs under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/
```

Required summary files:

```text
summaries/source_timestep_summary_<label>.csv
summaries/gap_by_timestep_<label>.csv
summaries/frequency_gap_summary_<label>.csv
summaries/signal_candidates_<label>.csv
```

The candidate table ranks the largest positive DreamBooth-over-ordinary-real pockets across:

```text
clean_norm by timestep
dct_clean_low by timestep
dct_clean_mid by timestep
dct_clean_high by timestep
```

## Decision Rule

This stage is a rescue check for the metric.

Promising signal:

```text
At least 3 of 4 smoke classes show positive subject_specific_gap
on the same interpretable axis, such as the same timestep range or frequency band.
```

Weak / pivot signal:

```text
Positive pockets are isolated to one class, one timestep, or one frequency band,
or the mean subject_specific_gap stays negative across all interpretable axes.
```

If Stage 1.5D is weak, do not proceed to Stage 2 personalization training from this metric. The next work should redesign the metric around trajectory-level or vector-structured evidence.

## Executed Result

Results are recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/
```

Clean-norm by timestep remains weak:

```text
timestep  positive DreamBooth classes  mean subject-specific gap
50        1 / 4                        -0.015660
200       0 / 4                        -0.013493
500       0 / 4                        -0.010486
800       0 / 4                        -0.002869
999       1 / 4                        -0.000809
```

Frequency decomposition shows the only positive mean signal in low frequency:

```text
frequency band  positive cells  total cells  mean subject-specific gap
high            0               20           -0.002032
low             12              20            0.001171
mid             5               20           -0.001377
```

The strongest positives are mostly `vase`, so they are not robust. The useful clue is late low frequency:

```text
timestep  band  positive DreamBooth classes  mean subject-specific gap
800       low   4 / 4                        0.000453
999       low   3 / 4                        0.000087
```

Decision: no Stage 2 training from this metric. The next metric redesign should focus on low-frequency, late-timestep, vector-structured behavior rather than average scalar norms.
