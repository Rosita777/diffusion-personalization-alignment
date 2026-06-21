# Claude Review Of DADT Spec

Date: 2026-06-21

Model consulted:

```text
anthropic/claude-opus-4.7 through the 360 OpenAI-compatible API
```

Purpose: reviewer-style critique of the v0 research design spec.

## Main Takeaways

The critique agreed that the project has main-conference potential if the diagnostic claim is strong:

```text
pre-training reference-target off-priorness predicts post-training prior drift
```

It also warned that the method can easily look like gated distillation unless the measurement is rigorous and the method improves the subject-fidelity / prior-preservation Pareto frontier.

## Accepted Changes

- Added DCO as a priority nearest-neighbor work.
- Added SDS/VSD and ProlificDreamer-style score distillation as related-work risks.
- Added Min-SNR and P2 weighting as timestep-weighting risks.
- Strengthened Stage 1 measurement with controls:
  - base-generated prediction-error floor;
  - `c_base` ablation;
  - timestep/SNR normalization;
  - in-distribution image controls;
  - VAE roundtrip control;
  - latent-vs-image frequency sanity check.
- Reframed Claim 4 from single-point superiority to Pareto-frontier improvement.
- Added an explicit subject-fidelity preservation claim.
- Rewrote `P_prior_harmful` as a v0 inductive bias rather than a solved detector.

## Not Accepted

Claude claimed that `2605.xxxxx` is not a valid arXiv identifier. This is incorrect under the current 2026 date. MixSD and Spectral Progressive Diffusion were kept with their arXiv IDs and marked as verified on 2026-06-21.

## Current Decision

Do not start DADT training yet. The next technical plan should cover only Stage 1 measurement:

```text
reference residual measurement
base-generated error floor
c_base ablation
timestep/SNR normalization
small frequency sanity check
```

The go/no-go signal is whether reference-image residuals are meaningfully above the base-generated image residual floor and robust to conditioning choice.
