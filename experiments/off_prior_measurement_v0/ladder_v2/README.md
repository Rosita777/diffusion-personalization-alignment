# Off-Prior Measurement V0 Ladder V2

Status: completed 8-subject smoke run. Go / No-Go result: No-Go.

Purpose: test whether reference prior compatibility controls denoising-target off-priorness before personalization fine-tuning.

Reference-regime ladder:

```text
easy_control < standard_reference < hard_reference
```

Subjects:

```text
dog, cat, backpack, vase, colorful_sneaker, shiny_sneaker, fancy_boot, dog7
```

This smoke run uses `reference_images_per_subject: 1` to keep large GitHub-hosted DreamBooth images tractable on the current network path. It uses `dog7` instead of `grey_sloth_plushie` because the latter repeatedly stalled during blob download in this environment. Paper-scale runs should increase the image count and return to all 30 subjects after the measurement/control design is revised.

Completed run summary:

- Raw measurement rows: 19,200.
- Selected conditioning: `class`.
- Hard-reference positive subjects: 3 of 8.
- Hard greater than standard: 0 of 8.
- Standard greater than easy: 5 of 8.
- Base hard-control positive subjects: 3 of 8.
- Roundtrip sanity check passed: false.
- Mean floor-adjusted residuals: easy `-0.0050`, standard `0.0117`, hard `0.0013`.
- Strongest timestep: 50.
- Strongest latent DCT band: low.

Interpretation: the current metric sees standard DreamBooth references as modestly above easy controls, but the deterministic hard-reference variants do not strengthen that signal. VAE roundtrip controls are also high enough to be a serious confound. Do not move to personalization fine-tuning from this result alone; revise the off-priorness measurement and controls first.

Reproduction run order:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2

$PYTHON -m scripts.off_prior_measurement.generate_controls --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.dreambooth_data --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.hard_references --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
$PYTHON -m scripts.off_prior_measurement.roundtrip_controls --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
```

Single-GPU measurement:

```bash
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv
```

Four-GPU measurement:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 0 --world-size 4 --output-name raw_metrics_rank0.csv
CUDA_VISIBLE_DEVICES=1 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 1 --world-size 4 --output-name raw_metrics_rank1.csv
CUDA_VISIBLE_DEVICES=2 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 2 --world-size 4 --output-name raw_metrics_rank2.csv
CUDA_VISIBLE_DEVICES=3 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 3 --world-size 4 --output-name raw_metrics_rank3.csv
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --world-size 4 --merge-shards
```

Analysis:

```bash
$PYTHON -m scripts.off_prior_measurement.summarize --raw-metrics $EXP/measurements/raw_metrics.csv --output-dir $EXP
$PYTHON -m scripts.off_prior_measurement.plot --scored-metrics $EXP/summaries/scored_metrics.csv --figures-dir $EXP/figures
$PYTHON -m scripts.off_prior_measurement.write_conclusion --experiment-dir $EXP
```

Go / No-Go:

- `hard_reference` positive for at least 6 of 8 subjects under class or class-context conditioning.
- `hard_reference > standard_reference` for at least 6 of 8 subjects.
- `standard_reference > easy_control` for at least 4 of 8 subjects.
- `base_hard_control` remains positive.
- `roundtrip_control` does not explain the hard-reference signal.
