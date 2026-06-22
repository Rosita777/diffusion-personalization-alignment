# Off-Prior Measurement V0 Smoke Test

Status: pipeline code implemented and lightweight unit-tested; full SD 1.5/DreamBooth GPU run not generated yet.

Purpose: measure whether DreamBooth reference-image denoising targets are farther from the SD 1.5 base prediction than base-generated controls.

Dataset:

- DreamBooth smoke subjects: dog, cat, backpack, colorful_sneaker, vase.
- Standard references: original DreamBooth images.
- Easy controls: SD 1.5 class-prompt generations.
- Hard controls: SD 1.5 unusual-context prompt generations.
- VAE roundtrip controls: original DreamBooth references encoded and decoded through the SD 1.5 VAE.

Primary comparison:

```text
dreambooth_reference vs. base_easy_control
```

Hard controls are not evidence for the main claim. They test whether unusual context prompts alone explain residual behavior.

Run order:

```bash
python -m scripts.off_prior_measurement.generate_controls --config configs/off_prior_measurement_v0/smoke_test.yaml
python -m scripts.off_prior_measurement.dreambooth_data --config configs/off_prior_measurement_v0/smoke_test.yaml
python -m scripts.off_prior_measurement.roundtrip_controls --config configs/off_prior_measurement_v0/smoke_test.yaml --reference-manifest experiments/off_prior_measurement_v0/smoke_test/manifests/reference_manifest.csv
python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv
python -m scripts.off_prior_measurement.summarize --raw-metrics experiments/off_prior_measurement_v0/smoke_test/measurements/raw_metrics.csv --output-dir experiments/off_prior_measurement_v0/smoke_test
python -m scripts.off_prior_measurement.plot --scored-metrics experiments/off_prior_measurement_v0/smoke_test/summaries/scored_metrics.csv --figures-dir experiments/off_prior_measurement_v0/smoke_test/figures
python -m scripts.off_prior_measurement.write_conclusion --experiment-dir experiments/off_prior_measurement_v0/smoke_test
```

Go / no-go:

Proceed to Stage 2 only if DreamBooth reference residuals are above the easy-control floor under class or class-plus-context conditioning for at least 4 of 5 subjects.
