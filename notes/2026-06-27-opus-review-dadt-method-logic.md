# Opus Review Of DADT Method Logic

Date: 2026-06-27

Status: external-model critique. Treat as reviewer-style input, not authority.

## Context

We asked Claude Opus 4.6, through the 987 API relay, to critique the DADT idea:

```text
reference-induced residual r = v_ref - v_base
as the training pull away from the base diffusion model,
with the goal of removing only artifact / background / prior-harmful parts
while preserving subject identity.
```

No API key or secret is stored in this note.

## Useful Points From Opus

### Main Verdict

Opus judged the direction as plausible but still pre-paper. Its strongest warning:

```text
Do not pitch a target-correction method before proving the phenomenon it corrects.
```

This agrees with our current Stage 1.4 gate.

### Strongest Argument For r

The best defense of `r = v_ref - v_base` is not just "distribution gap".

For denoising MSE training, near the pretrained initialization:

```text
loss = ||v_theta - v_ref||^2
```

If the base model currently predicts `v_base`, then:

```text
r = v_ref - v_base
```

is the residual that drives the update. In plain language:

```text
r is the pull that the reference target applies to the model.
```

This is stronger than saying `r` is vaguely off-prior. It connects the measurement directly to the training gradient.

Important caution:

```text
r is not automatically harmful.
```

Only decomposed components that correlate with prior drift or artifact leakage should be called harmful.

### Biggest Risks

Opus highlighted five reviewer attacks:

1. The DreamBooth-specific gap may not exist after removing VAE artifact and ordinary-real image gap.
2. A simple `alpha * v_ref + (1 - alpha) * v_base` method will look like ordinary distillation.
3. Identity/artifact decomposition is underdetermined without strong validation.
4. Target alignment may reduce subject fidelity.
5. Small metric gains will not be convincing in crowded personalization literature.

These are real risks and should shape the next experiments.

### Suggested Method Ladder

Opus suggested four possible method levels:

1. Frequency-band gating.
2. Attention-guided subject/background masking.
3. Learned residual projector.
4. Timestep-dependent residual subspace decomposition.

The most useful advice is:

```text
Make the analysis itself decide the method axes.
```

That means we should not assume in advance that frequency, region, or timestep is the right decomposition. Stage 1.4 / Stage 2 should reveal where the separable signal lives.

### Minimal Experiment Package

Opus recommended that the minimal convincing package include:

```text
decomposition analysis
gap-forgetting correlation
global scalar-mixing baseline
target-alignment ablations
identity metric
class-prior retention metric
qualitative failure cases
```

The scalar-mixing baseline is especially important because reviewers may say:

```text
You are just diluting the reference target.
```

### What Not To Claim

Do not claim:

```text
all r is bad
DADT replaces DreamBooth or LoRA
the decomposition is perfect
this works for every diffusion architecture
prior preservation is wrong
simple averaging is enough
```

Claim instead:

```text
reference targets contain multiple residual sources;
some are identity-useful and some are prior-harmful;
DADT is target-level and complementary to replay / regularization.
```

## Our Judgment

The most valuable Opus contribution is the gradient framing:

```text
r is the reference-induced training pull.
```

This should replace loose wording like:

```text
r is the bad off-prior component.
```

We should keep Stage 1.4, despite Opus suggesting fewer moving parts, because our prior experiments already showed VAE roundtrip is a serious confound. Without ordinary-real and roundtrip controls, the measurement is not trustworthy.

However, Opus is right that the paper should not become an overcomplicated four-way taxonomy. The paper-facing story can be simpler:

```text
We use source decomposition to avoid confounds,
then build the method only around the empirically separable harmful axis.
```

## Updated Research Logic

The clean logic should be:

```text
1. In diffusion personalization, r = v_ref - v_base is the reference-induced training pull.
2. r contains both identity-useful and incidental components.
3. Stage 1.4 tests whether DreamBooth references contain residual structure beyond VAE artifact and ordinary-real image gap.
4. If such structure exists, test whether it predicts class-prior forgetting.
5. Only then design DADT to suppress the harmful, empirically identified component.
6. Evaluate subject fidelity and prior retention together.
```

## Practical Next Step

Do not start method coding yet.

Immediate next experiment remains:

```text
complete Stage 1.4 ordinary-real controls
run source decomposition
look for where r differs: timestep, frequency, spatial region, or subspace
```

Only after that should we choose between:

```text
frequency-aware target correction
subject/background-aware target correction
subspace-projection target correction
```
