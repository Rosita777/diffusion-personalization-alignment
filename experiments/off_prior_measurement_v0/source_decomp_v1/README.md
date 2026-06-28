# Off-Prior Measurement V0 Source Decomposition V1

Status: complete. The Stage 1.4 smoke run produced a Pivot decision.

Purpose: separate VAE/projection artifact, ordinary real-image domain gap, and DreamBooth subject-specific target gap before any personalization fine-tuning.

## Data

The ordinary-real controls are COCO 2017 validation images listed in `data/manifests/ordinary_real_controls_v1.yaml`:

```text
dog:      000000029393
cat:      000000443303
backpack: 000000370478
vase:     000000168458
```

Raw COCO images are cached under `data/cache/off_prior_measurement_v0/ordinary_real_controls/coco2017/` and are not committed. They were selected because the class object is visible in an ordinary real scene, not because they are personalization subjects.

## Verification

Lightweight test suite:

```text
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_source_decomp_manifest.py tests/off_prior_measurement/test_source_decomp_measure.py tests/off_prior_measurement/test_source_decomp_roundtrip.py tests/off_prior_measurement/test_source_decomp_plot.py tests/off_prior_measurement/test_source_decomp_conclusion.py tests/off_prior_measurement/test_source_decomp_summarize.py -v
15 passed
```

Smoke run size:

```text
manifest rows: 79
raw metric rows: 1975
source groups: base_generated_control, ordinary_real_control, dreambooth_reference
classes: dog, cat, backpack, vase
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

$PYTHON -m scripts.off_prior_measurement.source_decomp_roundtrip \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv

CUDA_VISIBLE_DEVICES=1 $PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --rank 0 \
  --world-size 4 \
  --output-name raw_source_decomp_metrics_rank0.csv

CUDA_VISIBLE_DEVICES=2 $PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --rank 1 \
  --world-size 4 \
  --output-name raw_source_decomp_metrics_rank1.csv

CUDA_VISIBLE_DEVICES=3 $PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --rank 2 \
  --world-size 4 \
  --output-name raw_source_decomp_metrics_rank2.csv

CUDA_VISIBLE_DEVICES=5 $PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --rank 3 \
  --world-size 4 \
  --output-name raw_source_decomp_metrics_rank3.csv

$PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --world-size 4 \
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

## Result

Selected conditioning: `class`.

```text
class      real_domain_gap  subject_specific_gap
backpack          0.017963             -0.009677
cat               0.008754             -0.012851
dog               0.010689              0.117667
vase              0.016680             -0.006093
```

Summary:

```text
subject-specific positive classes: 1 of 4
mean real-domain gap: 0.0135
mean subject-specific gap: 0.0223
mean DreamBooth artifact fraction: 0.8961
Go / Pivot decision: Pivot
```

Interpretation: the current measurement does not support a stable personalization-specific target-gap story. The positive mean subject-specific gap is driven by dog; three of four classes are negative relative to ordinary real controls. The artifact fraction is still high, so Stage 2 personalization fine-tuning should not proceed from this metric.

## Decision Rule

Proceed toward Stage 2 only if the projection-cleaned DreamBooth subject-specific gap is positive across most smoke classes, the effect is not explained by ordinary real-image domain gap, and DreamBooth artifact fraction is not dominant. Otherwise pivot the research story before spending GPU on personalization fine-tuning.
