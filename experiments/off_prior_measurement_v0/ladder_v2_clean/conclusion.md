# Stage 1.3 Clean Off-Priorness Conclusion

Date: 2026-06-25

Source experiment:

```text
experiments/off_prior_measurement_v0/ladder_v2
```

Selected conditioning:

```text
class
```

Result summary:

- Clean standard-reference positive subjects: 0 of 8.
- Clean hard-reference not below standard subjects: 0 of 8.
- Mean raw standard-reference residual: 0.0117.
- Mean raw hard-reference residual: 0.0013.
- Mean clean standard-reference residual: -0.0067.
- Mean clean hard-reference residual: -0.0171.
- Mean standard-reference roundtrip attribution ratio: 1.1502.
- Positive clean-standard concentration: 1.0000.
- Strongest clean standard-reference timestep: 999.

Go / No-Go checks:

- Clean standard positive for at least 5 subjects: False.
- Clean standard mean is positive: False.
- Clean hard is not systematically below clean standard: False.
- Standard roundtrip attribution ratio is below 0.75: False.
- Signal is not concentrated in one subject: False.

Interpretation:

- Go / no-go decision: No-Go.
- If Go: proceed to Stage 2 correlation-with-forgetting using clean off-priorness.
- If No-Go: revise off-priorness measurement before personalization fine-tuning.

Caveat:

This diagnostic subtracts VAE roundtrip artifacts from existing v2 measurements. It still does not prove downstream personalization forgetting.
