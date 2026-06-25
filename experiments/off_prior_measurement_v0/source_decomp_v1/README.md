# Off-Prior Measurement V0 Source Decomposition V1

Status: implementation prepared. The real smoke run is blocked until local ordinary-real control images are provided at the paths listed in `data/manifests/ordinary_real_controls_v1.yaml`.

Purpose: separate VAE/projection artifact, ordinary real-image domain gap, and DreamBooth subject-specific target gap before any personalization fine-tuning.

## Current Blocker

The source-decomposition manifest intentionally refuses to substitute DreamBooth images for ordinary-real controls. A dry run currently fails with a clear missing-file error for:

```text
data/local_real_controls/dog/dog_real_00.jpg
data/local_real_controls/cat/cat_real_00.jpg
data/local_real_controls/backpack/backpack_real_00.jpg
data/local_real_controls/vase/vase_real_00.jpg
```

Add local research-only images at those paths, or update `data/manifests/ordinary_real_controls_v1.yaml` to point to equivalent local class images. Do not commit raw ordinary-real images unless their license and size make that appropriate.

## Verification

Lightweight test suite:

```text
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
55 passed
```

Manifest validation dry run:

```text
FileNotFoundError: Missing ordinary real control image paths: data/local_real_controls/dog/dog_real_00.jpg, data/local_real_controls/cat/cat_real_00.jpg, data/local_real_controls/backpack/backpack_real_00.jpg, data/local_real_controls/vase/vase_real_00.jpg
```

## Run Order

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/source_decomp_v1.yaml
export EXP=experiments/off_prior_measurement_v0/source_decomp_v1

$PYTHON -m scripts.off_prior_measurement.source_decomp_manifest \
  --reference-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/reference_manifest.csv \
  --control-manifest experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_v1.yaml \
  --roundtrip-root data/cache/off_prior_measurement_v0/source_decomp_roundtrip \
  --output $EXP/manifests/source_decomp_manifest.csv

$PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv

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
manifests/source_decomp_manifest.csv
measurements/raw_source_decomp_metrics.csv
summaries/source_group_summary.csv
summaries/source_gap_summary.csv
summaries/timestep_frequency_summary.csv
figures/source_gap_bars.png
figures/artifact_fraction_by_source.png
figures/clean_timestep_curves.png
conclusion.md
```

## Decision Rule

Proceed toward Stage 2 only if the projection-cleaned DreamBooth subject-specific gap is positive across most smoke classes, the effect is not explained by ordinary real-image domain gap, and DreamBooth artifact fraction is not dominant. Otherwise pivot the research story before spending GPU on personalization fine-tuning.
