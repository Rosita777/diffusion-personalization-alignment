# Off-Prior Measurement V0 Source Decomposition Stage 1.5A

Status: complete. This failure-diagnosis run produced a stronger Pivot decision.

Purpose: check whether the Stage 1.4 Pivot was caused by weak controls, single ordinary-real images, or too few DreamBooth reference images. Stage 1.5A keeps the same source-decomposition metric but improves the data matching before deciding whether the current target-gap story is reliable.

## Data

The run uses four classes: dog, cat, backpack, and vase.

DreamBooth references use all locally available images for these classes:

```text
dog:      5 reference images
cat:      5 reference images
backpack: 6 reference images
vase:     6 reference images
```

Ordinary-real controls are COCO 2017 validation images listed in `data/manifests/ordinary_real_controls_stage15a.yaml`. Each class has four control regimes:

```text
clean object
matched close-up or matched context
cluttered scene
background-heavy scene
```

Raw COCO images are cached under `data/cache/off_prior_measurement_v0/ordinary_real_controls/stage15a/` and are not committed.

## Run Size

```text
manifest rows: 58
raw metric rows: 1450
timesteps: 50, 200, 500, 800, 999
noise seeds: 0, 1, 2, 3, 4
conditioning: class prompt only
source groups: base_generated_control, ordinary_real_control, dreambooth_reference
```

## Run Order

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/source_decomp_stage15a.yaml
export EXP=experiments/off_prior_measurement_v0/source_decomp_stage15a

$PYTHON -m scripts.off_prior_measurement.source_decomp_manifest \
  --reference-manifest $EXP/manifests/reference_manifest.csv \
  --control-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_stage15a.yaml \
  --roundtrip-root data/cache/off_prior_measurement_v0/source_decomp_stage15a_roundtrip \
  --output $EXP/manifests/source_decomp_manifest.csv

$PYTHON -m scripts.off_prior_measurement.source_decomp_roundtrip \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv

# Measurement was run with 8 rank shards on GPUs 0-7, then merged.
$PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --world-size 8 \
  --merge-shards

$PYTHON -m scripts.off_prior_measurement.source_decomp_summarize \
  --raw-metrics $EXP/measurements/raw_source_decomp_metrics.csv \
  --output-dir $EXP

$PYTHON -m scripts.off_prior_measurement.source_decomp_plot \
  --summary-dir $EXP/summaries \
  --figures-dir $EXP/figures

$PYTHON -m scripts.off_prior_measurement.source_decomp_conclusion \
  --experiment-dir $EXP
```

## Expected Outputs

```text
manifests/reference_manifest.csv
manifests/source_decomp_manifest.csv
measurements/raw_source_decomp_metrics.csv
summaries/source_group_summary.csv
summaries/source_regime_summary.csv
summaries/source_gap_summary.csv
summaries/timestep_frequency_summary.csv
figures/source_gap_bars.png
figures/artifact_fraction_by_source.png
figures/clean_timestep_curves.png
conclusion.md
```

## Result

Selected conditioning: `class`.

```text
class      base control  ordinary real  DreamBooth ref  real-domain gap  subject-specific gap
backpack       0.057695       0.073823        0.063271         0.016128              -0.010552
cat            0.059336       0.070294        0.059464         0.010957              -0.010829
dog            0.058292       0.070935        0.061365         0.012643              -0.009571
vase           0.055904       0.071771        0.068069         0.015867              -0.003702
```

Summary:

```text
subject-specific positive classes: 0 of 4
mean real-domain gap: 0.0139
mean subject-specific gap: -0.0087
mean DreamBooth artifact fraction: 0.9809
Go / Pivot decision: Pivot
```

## Interpretation

The Stage 1.4 dog positive signal did not survive better controls. With multiple DreamBooth references and four ordinary-real regimes per class, DreamBooth references are not more off-prior than ordinary real images under the current clean residual metric. Ordinary real images are consistently farther from base-generated controls than DreamBooth references are from ordinary real controls.

The artifact fraction is also extremely high, around 0.98. This means most raw residual energy is still explained by the roundtrip/projection artifact direction, not by a clean personalization-specific component.

Conclusion: do not start Stage 2 personalization fine-tuning from this metric. The current evidence supports either redesigning the measurement or pivoting the paper story toward real-image projection/domain alignment rather than claiming a stable DreamBooth subject-specific target gap.

## Next Action

The next research step should locate why the metric fails before proposing DADT training. Promising checks are prompt-matched generated controls, caption-matched real controls, and a less roundtrip-dominated residual definition. Training should wait until the measurement can isolate a stable prior-harmful component.
