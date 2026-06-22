# Off-Prior Measurement V0 Smoke Test

Status: complete first smoke run on 2026-06-22. Result is No-Go under the current 4-of-5 subject rule.

Purpose: measure whether DreamBooth reference-image denoising targets are farther from the SD 1.5 base prediction than base-generated controls.

Dataset:

- DreamBooth smoke subjects: dog, cat, backpack, clock, vase.
- Standard references: original DreamBooth images.
- Easy controls: SD 1.5 class-prompt generations.
- Hard controls: SD 1.5 unusual-context prompt generations.
- VAE roundtrip controls: original DreamBooth references encoded and decoded through the SD 1.5 VAE.

Local access choices:

- Model: `configs/off_prior_measurement_v0/smoke_test.yaml` points to `data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5`, populated from ModelScope with only diffusers-required files.
- Dataset: `dataset_source: github_api` downloads selected `google/dreambooth` subject folders through the GitHub Contents API. The smoke run uses `clock` instead of `colorful_sneaker` because the current network path handles small API-backed files reliably while large raw GitHub image downloads hang; larger subjects can return in the all-subject run once dataset access is stabilized.

Latest result:

- Raw metrics: `measurements/raw_metrics.csv` with 14,400 rows.
- Reference positives: 2 of 5 subjects under `class`, 2 of 5 under `class_context`, and 2 of 5 under `null`.
- Mean floor-adjusted residual for DreamBooth references is slightly negative under class-conditioned settings.
- Mean floor-adjusted residual for hard controls is positive, so the metric can detect deliberately unusual generated controls.
- Current interpretation: the runnable lightweight subject subset does not support moving to Stage 2 fine-tuning yet. Next revision should focus on stronger reference-regime construction, larger subjects once dataset access is solved, or a more targeted off-priorness metric.

Primary comparison:

```text
dreambooth_reference vs. base_easy_control
```

Hard controls are not evidence for the main claim. They test whether unusual context prompts alone explain residual behavior.

Run order:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.generate_controls --config configs/off_prior_measurement_v0/smoke_test.yaml
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.dreambooth_data --config configs/off_prior_measurement_v0/smoke_test.yaml
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.roundtrip_controls --config configs/off_prior_measurement_v0/smoke_test.yaml --reference-manifest experiments/off_prior_measurement_v0/smoke_test/manifests/reference_manifest.csv
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.summarize --raw-metrics experiments/off_prior_measurement_v0/smoke_test/measurements/raw_metrics.csv --output-dir experiments/off_prior_measurement_v0/smoke_test
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.plot --scored-metrics experiments/off_prior_measurement_v0/smoke_test/summaries/scored_metrics.csv --figures-dir experiments/off_prior_measurement_v0/smoke_test/figures
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.write_conclusion --experiment-dir experiments/off_prior_measurement_v0/smoke_test
```

Multi-GPU measurement example:

```bash
CUDA_VISIBLE_DEVICES=5 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv --rank 0 --world-size 4 --output-name raw_metrics_rank0.csv
CUDA_VISIBLE_DEVICES=1 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv --rank 1 --world-size 4 --output-name raw_metrics_rank1.csv
CUDA_VISIBLE_DEVICES=7 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv --rank 2 --world-size 4 --output-name raw_metrics_rank2.csv
CUDA_VISIBLE_DEVICES=2 /home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv --rank 3 --world-size 4 --output-name raw_metrics_rank3.csv
/home/deepseek_VG/.conda/envs/dyme/bin/python -m scripts.off_prior_measurement.measure --config configs/off_prior_measurement_v0/smoke_test.yaml --manifest experiments/off_prior_measurement_v0/smoke_test/manifests/combined_manifest.csv --world-size 4 --merge-shards
```

Go / no-go:

Proceed to Stage 2 only if DreamBooth reference residuals are above the easy-control floor under class or class-plus-context conditioning for at least 4 of 5 subjects.
