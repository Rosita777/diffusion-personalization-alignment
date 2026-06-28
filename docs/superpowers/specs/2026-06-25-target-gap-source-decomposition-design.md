# Stage 1.4 Target-Gap Source Decomposition Design

Date: 2026-06-25

Status: executed. The Stage 1.4 smoke run produced a Pivot decision, the Stage 1.5A failure-diagnosis rerun produced a stronger Pivot after adding better ordinary-real controls and multiple DreamBooth references, the Stage 1.5B prompt-matched control diagnosis found that coarse class prompts are not the main cause of the remaining ordinary-real gap, the Stage 1.5C metric ablation found that prompt matching mostly reduces the raw / artifact-aligned gap rather than the projection-cleaned gap, and the Stage 1.5D fine-grained diagnosis found only a weak low-frequency late-timestep clue.

## Purpose

Stage 1.3 showed that the current off-priorness metric is dominated by VAE roundtrip artifacts. The raw DreamBooth reference signal disappeared after subtracting the matched roundtrip residual:

```text
clean standard-reference positive subjects: 0 of 8
mean raw standard-reference residual: 0.0117
mean clean standard-reference residual: -0.0067
standard-reference roundtrip attribution ratio: 1.1502
```

This means the project should not proceed to personalization fine-tuning from the current metric. Stage 1.4 changes the question from:

```text
Are DreamBooth reference targets off-prior?
```

to:

```text
Which source creates the target gap: VAE/projection artifact,
ordinary real-image domain gap, or DreamBooth subject-specific mismatch?
```

The goal is to recover a scientifically interpretable measurement before designing DADT target correction.

## Implementation Status

The Stage 1.4 code path is implemented under `scripts/off_prior_measurement/`:

```text
source_decomp_manifest.py
source_decomp_measure.py
source_decomp_summarize.py
source_decomp_plot.py
source_decomp_conclusion.py
```

The smoke config is `configs/off_prior_measurement_v0/source_decomp_v1.yaml`, and the ordinary-real manifest is `data/manifests/ordinary_real_controls_v1.yaml`. The ordinary-real controls use four COCO 2017 validation images cached locally under `data/cache/off_prior_measurement_v0/ordinary_real_controls/coco2017/`. Raw images are not committed.

The Stage 1.5A diagnosis config is `configs/off_prior_measurement_v0/source_decomp_stage15a.yaml`, and the expanded ordinary-real manifest is `data/manifests/ordinary_real_controls_stage15a.yaml`. It uses four COCO ordinary-real regimes per class and all locally available DreamBooth references for dog, cat, backpack, and vase. Results are recorded under `experiments/off_prior_measurement_v0/source_decomp_stage15a/`.

The Stage 1.5B prompt-matched diagnosis config is `configs/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched.yaml`, and the prompt-matched ordinary-real manifest is `data/manifests/ordinary_real_controls_stage15b_prompt_matched.yaml`. It compares ordinary-real controls with base SD 1.5 images generated from image-specific prompts. Results are recorded under `experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/`.

The Stage 1.5C metric ablation reuses Stage 1.5A and Stage 1.5B raw measurements through `scripts/off_prior_measurement/source_decomp_metric_ablation.py`. It compares `raw_norm`, `projected_artifact_norm = raw_norm * artifact_fraction`, and `clean_norm`. Results are recorded under `experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation/`.

The Stage 1.5D fine-grained diagnosis reuses Stage 1.5A and Stage 1.5B raw measurements through `scripts/off_prior_measurement/source_decomp_fine_grained.py`. It slices the gap by timestep and latent DCT clean-frequency bands. Results are recorded under `experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/`.

## Execution Result

The executed Stage 1.4 smoke run is recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_v1/
```

The manifest contains dog, cat, backpack, and vase only, because those are the classes with ordinary-real controls. Measurement was run with 5 timesteps and 5 noise seeds, producing 1975 raw rows.

Selected conditioning: `class`.

```text
class      real_domain_gap  subject_specific_gap
backpack          0.017963             -0.009677
cat               0.008754             -0.012851
dog               0.010689              0.117667
vase              0.016680             -0.006093
```

Go / Pivot summary:

```text
subject-specific positive classes: 1 of 4
mean real-domain gap: 0.0135
mean subject-specific gap: 0.0223
mean DreamBooth artifact fraction: 0.8961
decision: Pivot
```

Interpretation: the positive mean subject-specific gap is not robust because it is driven by dog. Backpack, cat, and vase are below ordinary real controls after projection cleaning. Artifact fraction remains high, so the current measurement should not be used to justify Stage 2 personalization fine-tuning.

## Stage 1.5A Failure-Diagnosis Result

The follow-up run is recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15a/
```

Stage 1.5A tests whether the Stage 1.4 result was caused by weak matching or too few reference images. It expands ordinary-real controls to four regimes per class and uses all locally available DreamBooth references. Measurement was run with 5 timesteps and 5 noise seeds, producing 1450 raw rows.

Selected conditioning: `class`.

```text
class      real_domain_gap  subject_specific_gap
backpack          0.016128             -0.010552
cat               0.010957             -0.010829
dog               0.012643             -0.009571
vase              0.015867             -0.003702
```

Go / Pivot summary:

```text
subject-specific positive classes: 0 of 4
mean real-domain gap: 0.0139
mean subject-specific gap: -0.0087
mean DreamBooth artifact fraction: 0.9809
decision: Pivot
```

Interpretation: better data matching did not rescue the personalization-specific target-gap story. Under the current clean residual metric, ordinary real images are more off-prior than DreamBooth references for all four smoke classes. The Stage 1.4 dog signal should be treated as unstable, and the next step should be metric redesign or a paper-story pivot rather than personalization fine-tuning.

## Stage 1.5B Prompt-Matched Control Result

The prompt-matched follow-up run is recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched/
```

Stage 1.5B tests whether Stage 1.5A failed mainly because ordinary-real images were conditioned with coarse class prompts. It does not include DreamBooth reference rows, so it should be read only as a base-generated versus ordinary-real control-gap diagnosis.

Selected conditioning: `prompt_matched`.

```text
class      class-only gap  prompt-matched gap  delta
backpack        0.016128            0.015855  -0.000273
cat             0.010957            0.008615  -0.002343
dog             0.012643            0.012796   0.000153
vase            0.015867            0.014554  -0.001313
mean            0.013899            0.012955  -0.000944
```

Interpretation: prompt matching reduces the mean real-domain gap by only about 6.8%. The gap remains positive for all four smoke classes. Coarse class prompts contribute a little, but they do not explain the failed Stage 1.5A signal. The next diagnosis should target the residual metric and projection pipeline itself rather than only improving text prompts.

## Stage 1.5C Metric-Ablation Result

The metric ablation is recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15c_metric_ablation/
```

Stage 1.5C reuses Stage 1.5A and Stage 1.5B raw measurements and separates three scalar views:

```text
raw_norm
projected_artifact_norm = raw_norm * artifact_fraction
clean_norm
```

Mean gap comparison:

```text
experiment_label          metric                   real_domain_gap  subject_specific_gap
stage15a_class            raw_norm                        0.039491             -0.033744
stage15a_class            projected_artifact_norm          0.037432             -0.032819
stage15a_class            clean_norm                      0.013899             -0.008663
stage15b_prompt_matched   raw_norm                        0.021300                   n/a
stage15b_prompt_matched   projected_artifact_norm          0.019200                   n/a
stage15b_prompt_matched   clean_norm                      0.012955                   n/a
```

Interpretation: prompt matching cuts the raw real-domain gap by about 46.1% and the artifact-aligned gap by about 48.7%, but it cuts the clean gap by only about 6.8%. This means prompt mismatch mostly affects the artifact/projection direction. After projection cleaning, the remaining ordinary-real gap is still present, and DreamBooth references remain below ordinary real controls in Stage 1.5A. The current scalar residual family therefore behaves like a generic real-image/projection-domain measurement, not a reliable personalization-specific off-priorness measurement.

Decision: do not start Stage 2 personalization fine-tuning from this metric. The next work should either redesign the evidence around trajectory-level or vector-structured behavior, add stronger VAE/projection calibration, or pivot the paper story toward real-image-to-diffusion-prior target alignment.

## Stage 1.5D Fine-Grained Result

The fine-grained diagnosis is recorded under:

```text
experiments/off_prior_measurement_v0/source_decomp_stage15d_fine_grained/
```

Stage 1.5D checks whether the weak average scalar result hides a DreamBooth-specific signal in timestep or frequency axes.

Clean-norm by timestep remains weak:

```text
timestep  positive DreamBooth classes  mean subject-specific gap
50        1 / 4                        -0.015660
200       0 / 4                        -0.013493
500       0 / 4                        -0.010486
800       0 / 4                        -0.002869
999       1 / 4                        -0.000809
```

Frequency decomposition finds the only positive mean signal in low frequency:

```text
frequency band  positive cells  total cells  mean subject-specific gap
high            0               20           -0.002032
low             12              20            0.001171
mid             5               20           -0.001377
```

The strongest low-frequency positives are mostly class-concentrated in `vase`, so they are not enough to revive the original average-gap story. The most useful clue is late low frequency:

```text
timestep  band  positive DreamBooth classes  mean subject-specific gap
800       low   4 / 4                        0.000453
999       low   3 / 4                        0.000087
```

Decision: no Stage 2 training from this metric. The next metric redesign should focus on low-frequency, late-timestep, vector-structured behavior rather than average scalar norms.

## Revised Hypothesis

The observed residual should be treated as a mixture:

```text
r_total = r_vae_projection + r_real_image_domain + r_subject_specific + r_noise
```

where:

- `r_vae_projection`: artifact from VAE encode-decode, resize, crop, and preprocessing;
- `r_real_image_domain`: gap between ordinary real images and SD 1.5's generated-image prior;
- `r_subject_specific`: extra gap caused by rare subject identity, unusual reference composition, or subject-background binding;
- `r_noise`: measurement noise from seed, timestep, conditioning, and finite samples.

The DADT story only needs `r_subject_specific` or a clearly localized prior-harmful component. If the measured gap is mostly `r_vae_projection` or generic `r_real_image_domain`, the paper should pivot before proposing personalization-specific target alignment.

## Experimental Groups

Stage 1.4 should compare four image sources for matched class prompts, timesteps, and noise seeds.

### Base-Generated Controls

Images generated by the base SD 1.5 model from plain class prompts:

```text
a photo of a dog
a photo of a backpack
a photo of a vase
```

This is the in-prior floor.

### Ordinary Real Class Controls

Natural real images of the same class that are not personalization subjects. Examples:

```text
COCO / OpenImages / ImageNet dog
COCO / OpenImages / ImageNet backpack
COCO / OpenImages / ImageNet vase
```

This group answers whether the gap is simply a real-image-to-SD-prior domain gap. The implementation should support a local manifest first, because large dataset downloads are not reliable in this environment.

### DreamBooth Standard References

Original DreamBooth subject reference images. This group is the personalization setting.

### Natural Hard DreamBooth References

Naturally difficult DreamBooth images or subjects, not PIL-style synthetic hard variants. Hardness labels should be recorded as metadata:

```text
unusual_pose
unusual_crop
cluttered_background
rare_appearance
subject_background_binding
class_ambiguous_view
```

The v2 deterministic hard variants should be treated as a failed stress test, not as the default Stage 1.4 hard regime.

## Paired VAE Artifact Decomposition

Scalar subtraction was too blunt in Stage 1.3. Stage 1.4 should compute vector-level residual decomposition during measurement.

For an image `x`:

```text
r_x = v_target(x) - v_base(x)
```

For its VAE roundtrip version `rt(x)` under the same class prompt, timestep, and noise seed:

```text
r_rt = v_target(rt(x)) - v_base(rt(x))
```

Then compute how much of `r_x` lies in the roundtrip-artifact direction:

```text
artifact_coeff = <r_x, r_rt> / (||r_rt||^2 + eps)
r_artifact = artifact_coeff * r_rt
r_clean = r_x - r_artifact
artifact_cosine = cosine(r_x, r_rt)
artifact_fraction = ||r_artifact|| / (||r_x|| + eps)
clean_fraction = ||r_clean|| / (||r_x|| + eps)
```

This is more informative than:

```text
score(x) - score(rt(x))
```

because it asks whether the residual direction itself is explained by the roundtrip artifact.

## Source-Gap Metrics

Report both raw and projection-cleaned scores:

```text
raw_norm = ||r_x|| / ||v_target(x)||
artifact_fraction = ||r_artifact|| / ||r_x||
clean_norm = ||r_clean|| / ||v_target(x)||
artifact_cosine = cosine(r_x, r_rt)
```

Then estimate source gaps:

```text
real_domain_gap = mean(clean_norm ordinary_real) - mean(clean_norm base_generated)
subject_specific_gap = mean(clean_norm dreambooth_reference) - mean(clean_norm ordinary_real_same_class)
natural_hard_gap = mean(clean_norm natural_hard_reference) - mean(clean_norm standard_dreambooth_reference)
```

The main scientific question is `subject_specific_gap`, not the absolute raw residual.

## Timestep And Frequency Axes

Keep the existing timestep axis:

```text
50, 200, 500, 800, 999
```

Keep the existing latent DCT bands:

```text
low, mid, high
```

But Stage 1.4 should report them for clean residuals as well:

```text
DCT(r_clean)
DCT(r_artifact)
```

This keeps the Spectral Progressive Diffusion connection alive without relying on a confounded scalar metric.

## Conditioning

Primary conditioning:

```text
class
```

Robustness conditioning:

```text
class_context
null
```

Stage 1.4 should not declare success if the effect appears only under `null` conditioning.

## Smoke Run

The first smoke run should be small and explicit:

```text
subjects/classes: dog, cat, backpack, vase
images per source: 1 to 2
noise seeds: 0 to 4
timesteps: 50, 200, 500, 800, 999
conditioning: class, class_context
```

This keeps the run cheap while testing the new decomposition. If ordinary real controls are not locally available, implementation should stop with a clear manifest error rather than silently using DreamBooth images as ordinary controls.

## Data Manifest Contract

Use one manifest for all source groups:

```text
subject_id
class_name
image_id
image_path
roundtrip_image_path
source_group
reference_regime
hardness_axis
conditioning_key
conditioning_prompt
source_dataset
source_license_note
```

Allowed `source_group` values:

```text
base_generated_control
ordinary_real_control
dreambooth_reference
natural_hard_reference
```

Each real image should have a roundtrip image path or be marked for roundtrip generation before measurement.

## Go / No-Go Criteria

Stage 1.4 is a measurement-recovery gate.

### Go To Stage 2 If

Proceed to forgetting correlation only if all hold under `class` or `class_context`:

- `subject_specific_gap > 0` for at least 3 of 4 smoke classes;
- mean `subject_specific_gap` is positive;
- `natural_hard_gap > 0` for at least 3 of 4 smoke classes when natural hard references are available;
- mean artifact fraction for DreamBooth references is below 0.75;
- the effect is not concentrated in only one class or one timestep.

### Pivot If

Do not proceed to Stage 2 if:

- ordinary real controls are as large as DreamBooth references after projection cleaning;
- DreamBooth references are not above ordinary real images;
- roundtrip artifact fraction remains above 0.75;
- the signal appears only under `null` conditioning;
- the result depends on synthetic hard variants rather than natural references.

## Interpretation Paths

### If DreamBooth Exceeds Ordinary Real Images

The original personalization-specific story survives:

```text
DreamBooth subject references contain target-gap components beyond
generic real-image domain gap.
```

Then Stage 2 should test whether clean `subject_specific_gap` predicts prior drift after personalization.

### If DreamBooth Matches Ordinary Real Images

The original story should pivot:

```text
The main issue is real-image-to-diffusion-prior projection alignment,
not personalization-specific target off-priorness.
```

This could still be a paper direction, but it becomes broader than personalization and must be positioned differently.

### If Everything Is Artifact-Dominated

Do not build DADT yet. First redesign the measurement around VAE-free or multi-VAE controls, or compare against image-space and latent-space reconstruction baselines.

## Implementation Shape

Add a new Stage 1.4 pipeline rather than modifying completed v2 results:

```text
configs/off_prior_measurement_v0/source_decomp_v1.yaml
data/manifests/ordinary_real_controls_v1.yaml
scripts/off_prior_measurement/source_decomp_manifest.py
scripts/off_prior_measurement/source_decomp_measure.py
scripts/off_prior_measurement/source_decomp_summarize.py
scripts/off_prior_measurement/source_decomp_plot.py
scripts/off_prior_measurement/source_decomp_conclusion.py
experiments/off_prior_measurement_v0/source_decomp_v1/
```

The measurement code should compute paired residual projection metrics online. It should not save full residual tensors by default because that would create large artifacts. A debug flag may save a tiny tensor sample for verification.

## Testing Requirements

Add tests for:

- manifest validation rejects missing ordinary real controls;
- roundtrip path pairing is exact and preserves string IDs;
- projection decomposition returns zero clean residual when `r_x` equals `r_rt`;
- projection decomposition returns full clean residual when `r_x` is orthogonal to `r_rt`;
- source-gap summary computes `real_domain_gap`, `subject_specific_gap`, and `natural_hard_gap`;
- conclusion logic handles Go and Pivot cases.

GPU tests are not required. Use tiny NumPy arrays and fake backends for unit tests.

## Project Hygiene

Stage 1.4 must not overwrite:

```text
experiments/off_prior_measurement_v0/ladder_v2/
experiments/off_prior_measurement_v0/ladder_v2_clean/
```

It should write only to:

```text
experiments/off_prior_measurement_v0/source_decomp_v1/
```

Large ordinary-real datasets should stay outside Git. Only small manifests, commands, summaries, figures, and conclusions should be committed.

## Spec Self-Review

- No placeholders remain.
- The scope is a measurement-recovery gate, not personalization fine-tuning.
- The design separates VAE artifact, ordinary real-image domain gap, and DreamBooth subject-specific gap.
- The first smoke run is small enough for the current environment.
- The Go / No-Go criteria are explicit and do not depend on synthetic hard variants.
