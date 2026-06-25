# Off-Prior Measurement V0 Ladder V2 Clean

Status: completed Stage 1.3 roundtrip-confound diagnostic. Go / No-Go result: No-Go.

Source experiment: `experiments/off_prior_measurement_v0/ladder_v2/`.

Purpose: subtract VAE roundtrip artifacts from the v2 floor-adjusted residuals before deciding whether Stage 2 personalization fine-tuning is scientifically justified.

Completed run summary:

- Selected conditioning: `class`.
- Clean standard-reference positive subjects: 0 of 8.
- Clean hard-reference not below standard subjects: 0 of 8.
- Mean raw standard-reference residual: `0.0117`.
- Mean raw hard-reference residual: `0.0013`.
- Mean clean standard-reference residual: `-0.0067`.
- Mean clean hard-reference residual: `-0.0171`.
- Mean standard-reference roundtrip attribution ratio: `1.1502`.
- Strongest clean standard-reference timestep: `999`.

Interpretation: the modest raw standard-reference signal observed in v2 does not survive roundtrip subtraction. Current v2 target-gap evidence is therefore dominated by VAE/reconstruction or preprocessing artifacts, not a clean reference-specific off-prior signal. Do not proceed to Stage 2 personalization fine-tuning from this metric.

Run order:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export SRC=experiments/off_prior_measurement_v0/ladder_v2
export DST=experiments/off_prior_measurement_v0/ladder_v2_clean

$PYTHON -m scripts.off_prior_measurement.clean_offprior --scored-metrics $SRC/summaries/scored_metrics.csv --output-dir $DST --source-experiment $SRC
$PYTHON -m scripts.off_prior_measurement.plot_clean_offprior --clean-scored-metrics $DST/summaries/clean_scored_metrics.csv --figures-dir $DST/figures
$PYTHON -m scripts.off_prior_measurement.write_clean_conclusion --experiment-dir $DST
```
