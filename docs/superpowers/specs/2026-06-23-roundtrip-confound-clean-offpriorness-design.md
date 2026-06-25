# Stage 1.3 Roundtrip Confound And Clean Off-Priorness Design

Date: 2026-06-23

Status: implemented and run. The completed Stage 1.3 diagnostic produced a No-Go; the raw v2 standard-reference signal did not survive VAE roundtrip subtraction.

Completed run note, 2026-06-25:

- Experiment directory: `experiments/off_prior_measurement_v0/ladder_v2_clean/`.
- Source experiment: `experiments/off_prior_measurement_v0/ladder_v2/`.
- Selected conditioning: `class`.
- Clean standard-reference positive subjects: 0 of 8.
- Clean hard-reference not below standard subjects: 0 of 8.
- Mean raw standard-reference residual: 0.0117.
- Mean clean standard-reference residual: -0.0067.
- Mean standard-reference roundtrip attribution ratio: 1.1502.
- Go / No-Go result: No-Go.
- Next research action: Stage 1.4 target-gap source decomposition, specified in `docs/superpowers/specs/2026-06-25-target-gap-source-decomposition-design.md`.

## Purpose

Stage 1 and Stage 1 v2 both produced No-Go results. The v2 ladder was still useful because it narrowed the failure mode:

```text
standard DreamBooth references are modestly above easy controls,
but deterministic hard-reference variants are not above standard references,
and VAE roundtrip controls are high enough to be a confound.
```

Stage 1.3 therefore diagnoses whether the measured target gap is a real reference-image prior mismatch or an artifact of VAE encode-decode, image resizing, and preprocessing.

The central question is:

```text
After subtracting reconstruction/preprocessing artifacts, is there still
a reference-specific off-prior target signal?
```

This stage should run before any personalization fine-tuning. If the metric is confounded, Stage 2 would only measure whether a noisy score correlates with forgetting.

## Diagnosis From V2

The completed v2 run under `experiments/off_prior_measurement_v0/ladder_v2/` found:

- hard-reference positive subjects: 3 of 8;
- hard greater than standard: 0 of 8;
- standard greater than easy: 5 of 8;
- roundtrip sanity check passed: false;
- mean easy-control floor-adjusted residual: -0.0050;
- mean standard-reference floor-adjusted residual: 0.0117;
- mean hard-reference floor-adjusted residual: 0.0013;
- strongest timestep: 50;
- strongest latent DCT band: low.

The most important warning is not only that hard references failed. It is that VAE roundtrip controls can be as high as, or higher than, standard references. That means the current residual:

```text
distance(v_ref, v_base) - easy_control_floor
```

mixes at least two effects:

```text
reference content / prior mismatch
VAE and preprocessing reconstruction mismatch
```

Stage 1.3 tries to separate them.

## Scope

In scope:

- Reuse the existing v2 experiment outputs and manifests.
- Add post-hoc summaries that compare each original reference against its VAE roundtrip version.
- Define clean off-priorness scores that subtract or regress out roundtrip artifacts.
- Add figures for raw versus clean scores across subject, timestep, frequency band, and reference regime.
- Write an updated conclusion that decides whether the metric is usable for Stage 2.

Out of scope:

- Personalization fine-tuning.
- New DreamBooth downloads.
- New hard-reference generation.
- SAM masks or foreground/background segmentation.
- Any claim that downstream forgetting is proven.

## Data Inputs

Primary input:

```text
experiments/off_prior_measurement_v0/ladder_v2/summaries/scored_metrics.csv
```

Required columns already exist:

```text
subject_id
image_id
source_group
reference_regime
hardness_axis
source_standard_image
variant_id
conditioning_key
timestep
noise_seed
normalized_l2
floor_adjusted_l2
cosine_distance
low_band_energy
mid_band_energy
high_band_energy
```

The implementation must preserve string identifiers such as `image_id="00"` and `conditioning_key="null"` when reading CSV files.

## Clean Off-Priorness Scores

Stage 1.3 should report three related scores, not one magic number.

### Raw Score

The existing score remains the starting point:

```text
O_raw(x) = floor_adjusted_l2(x)
```

This is comparable with previous Stage 1 outputs but still confounded.

### Pairwise Roundtrip-Subtracted Score

For a standard reference image and its matched VAE roundtrip control:

```text
O_clean_pair(ref) = O_raw(ref) - O_raw(roundtrip(ref))
```

For a hard-reference variant derived from a standard reference:

```text
O_clean_pair(hard) = O_raw(hard) - O_raw(roundtrip(source_standard_image))
```

This asks whether the reference or hard variant is more off-prior than the reconstruction artifact of its source image.

### Subject-Level Roundtrip-Subtracted Score

When exact pairing is unavailable or fragile, use a subject-level baseline:

```text
O_clean_subject(x) = O_raw(x) - median_subject_roundtrip(
    subject_id,
    conditioning_key,
    timestep,
    noise_seed
)
```

This is less precise but more robust to manifest mismatch.

### Roundtrip Attribution Ratio

Report how much of the raw signal is explained by roundtrip artifacts:

```text
roundtrip_ratio = abs(roundtrip_baseline) / (abs(raw_reference_score) + eps)
```

This ratio should not be used as the main score, but it helps communicate confound strength.

## Primary Comparisons

The main table should answer:

```text
raw standard > easy?
clean standard > 0?
raw hard > raw standard?
clean hard > clean standard?
roundtrip explains most of raw standard?
```

The preferred conditioning is still `class`, with `class_context` as a robustness check and `null` as a diagnostic axis.

The comparison should be reported at:

- subject level;
- reference-regime level;
- timestep level;
- frequency-band level;
- hardness-axis level for hard-reference variants.

## Expected Figures

Create a compact figure set:

```text
raw_vs_clean_ladder.png
roundtrip_attribution_by_subject.png
clean_timestep_curves.png
clean_frequency_heatmap.png
```

The most important figure is `raw_vs_clean_ladder.png`, showing easy, standard, hard, and roundtrip scores before and after roundtrip subtraction.

## Go / No-Go Criteria

Stage 1.3 is a diagnostic gate.

### Go To Stage 2 If

Proceed to correlation-with-forgetting only if all of these hold under `class` or `class_context` conditioning:

- clean standard-reference score is positive for at least 5 of 8 subjects;
- clean standard-reference mean is positive after roundtrip subtraction;
- clean hard-reference score is not systematically below clean standard-reference score;
- roundtrip attribution ratio is below 0.75 for the standard-reference mean;
- the signal is not concentrated in only one broken subject.

### Revise Metric If

Do not proceed to Stage 2 if:

- clean standard-reference score is near zero or negative;
- roundtrip attribution explains most of the raw standard-reference signal;
- clean scores are dominated by one timestep or one artifact-prone frequency band;
- hard-reference variants remain inverted after cleaning.

### Method Implication

If Stage 1.3 is Go, the project can move to Stage 2 and ask whether clean off-priorness predicts personalization forgetting.

If Stage 1.3 is No-Go, the project should revise the measurement itself before any DADT training. Likely revisions are:

- measure in image-pixel or VAE-latent reconstruction space separately from UNet target residuals;
- use natural hard references rather than PIL-style deterministic transforms;
- add foreground/background masks before comparing subject identity and prior-harmful components;
- compare multiple base models or VAE variants to isolate model-specific artifacts.

## Implementation Shape

This should be implemented as analysis code on top of existing outputs:

```text
scripts/off_prior_measurement/clean_offprior.py
scripts/off_prior_measurement/plot_clean_offprior.py
scripts/off_prior_measurement/write_clean_conclusion.py
```

Expected experiment output directory:

```text
experiments/off_prior_measurement_v0/ladder_v2_clean/
```

The clean analysis should copy or record the source experiment path:

```text
source_experiment: experiments/off_prior_measurement_v0/ladder_v2
```

This avoids modifying the completed v2 result while making the diagnostic analysis easy to reproduce.

## Testing Requirements

Add focused tests for:

- exact reference-to-roundtrip pairing;
- subject-level fallback baseline;
- string preservation for `image_id` and `conditioning_key`;
- clean score computation for standard and hard references;
- conclusion logic for Go and No-Go cases;
- figure generation from a tiny synthetic scored-metrics table.

No GPU is required for Stage 1.3 because it reuses existing v2 measurements.

## Research Risk

The best possible outcome is not necessarily a clean positive result. A strong negative result is still valuable because it prevents us from building a DADT method on top of a flawed measurement.

The key reviewer-facing lesson could become:

```text
Reference-target off-priorness is measurable only after controlling for
VAE reconstruction artifacts.
```

If the clean signal survives, this becomes a stronger foundation for the MixSD-to-diffusion story. If it disappears, the project should pivot to better target decomposition before proposing target alignment.

## Spec Self-Review

- No placeholders remain.
- The scope is limited to post-hoc analysis and conclusion writing.
- The Go / No-Go criteria are explicit.
- The spec does not claim downstream forgetting is proven.
- The design preserves the completed v2 result instead of rewriting it.
