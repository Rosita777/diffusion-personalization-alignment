# Stage 1 Smoke-Test Conclusion

Date: 2026-06-22

Config:

```text
configs/off_prior_measurement_v0/smoke_test.yaml
```

Primary comparison:

```text
dreambooth_reference vs. base_easy_control
```

Subjects:

```text
dog, cat, backpack, clock, vase
```

Result summary:

- Class-prompt conditioning: 2 of 5 subjects have positive mean floor-adjusted residual; mean value = -0.0112.
- Class-plus-context conditioning: 2 of 5 subjects have positive mean floor-adjusted residual; mean value = -0.0113.
- Null conditioning: 2 of 5 subjects have positive mean floor-adjusted residual; mean value = -0.0130.
- Strongest timestep by mean floor-adjusted residual: 50.
- Strongest latent DCT band by mean residual energy: low.

Interpretation:

- Go / no-go decision: No-Go.
- Main reason: class or class-plus-context conditioning passes the 4-of-5 subject rule = False.
- Most important caveat: this smoke test measures target residual structure only; it does not prove downstream forgetting until Stage 2.

Next step:

- If Go: create the Stage 2 correlation-with-forgetting plan.
- If No-Go: revise the off-priorness metric or conditioning design before any personalization training.
