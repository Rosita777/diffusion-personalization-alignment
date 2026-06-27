# Diffusion Literature Grounding For DADT

Date: 2026-06-27

Status: working related-work map. This note grounds DADT in diffusion papers so the project does not rely only on the MixSD analogy.

## Short Answer

The formula:

```text
r = v_ref - v_base
```

can be defended, but only with a careful meaning.

We should say:

```text
r is the training pull from the reference target away from the base model's current denoising behavior.
```

We should not say:

```text
r is automatically bad.
```

The bad part has to be proven by decomposition and intervention:

```text
source decomposition -> gap-forgetting correlation -> selective target correction
```

In plain words:

```text
v_ref tells the model what the reference image wants it to learn.
v_base tells us what the base model would currently do.
r tells us how the reference target would pull the base model away.
But only part of r may be harmful.
```

## Why Diffusion Papers Let Us Study Denoising Directions

### DDPM / Denoising Score Matching

Source: https://arxiv.org/abs/2006.11239

DDPM gives the basic language for this project: diffusion models are trained to predict denoising targets, and the training is connected to denoising score matching.

What this lets us say:

```text
It is legitimate to analyze the denoising target or score direction.
```

What it does not prove:

```text
It does not prove reference targets are harmful.
```

So DDPM supports the object we measure, not the whole DADT claim.

### Score / Direction Difference Papers

Sources:

- Delta Denoising Score: https://arxiv.org/abs/2304.07090
- DreamFusion / SDS: https://arxiv.org/abs/2209.14988
- ProlificDreamer / VSD: https://arxiv.org/abs/2305.16213

These papers make score or denoising direction differences useful in optimization or editing settings.

What this lets us say:

```text
Comparing denoising directions is diffusion-native, not invented only from an LLM analogy.
```

What it does not prove:

```text
These papers are not diffusion personalization target-correction methods.
```

So they support the tool, not the novelty claim.

## What Personalization Papers Already Do

### DreamBooth

Source: https://arxiv.org/abs/2208.12242

DreamBooth takes a few subject images, binds a unique identifier to that subject, and fine-tunes the text-to-image model. It also uses class-prior preservation so the model does not collapse the whole class into the subject.

Plain version:

```text
DreamBooth teaches the model "sks dog = this specific dog",
then makes it review normal dogs.
```

What it already solves:

```text
It knows personalization can damage the class prior.
```

What it does not ask:

```text
Should the reference-image denoising target itself be learned exactly?
```

DADT starts from this missing question.

### Textual Inversion

Source: https://arxiv.org/abs/2208.01618

Textual Inversion learns a new token embedding for the concept while keeping most of the model fixed.

Plain version:

```text
It adds a new word to the model's dictionary.
```

Why it matters:

```text
It shows one way to reduce damage is to restrict where learning happens.
```

But DADT is different:

```text
Textual Inversion restricts parameters.
DADT edits the training target.
```

### Custom Diffusion

Source: https://arxiv.org/abs/2212.04488

Custom Diffusion customizes concepts by updating a small subset of parameters, especially around text-image binding.

Plain version:

```text
It mostly changes how words attach to visual features.
```

Why it matters:

```text
It suggests subject learning is related to text-image binding, not just generic image reconstruction.
```

DADT can borrow this lesson by using text-attention or subject masks when deciding which target components to preserve.

## Dangerous Nearest Neighbors

### DreamBooth Prior Preservation

DreamBooth's prior preservation is like replay:

```text
learn this dog, but also review normal dogs.
```

Our distinction:

```text
DreamBooth adds extra class examples.
DADT changes what the reference examples teach.
```

So reviewers may ask:

```text
Is DADT just better prior preservation?
```

We need an ablation:

```text
same prior-preservation data, with and without target alignment
```

### Direct Consistency Optimization

Source: https://arxiv.org/abs/2402.12004

DCO keeps the customized model consistent with the pretrained model.

Plain version:

```text
After or during fine-tuning, do not let the personalized model wander too far from the base model.
```

Our distinction:

```text
DCO controls the model/output behavior.
DADT controls the reference target before the model learns it.
```

Necessary experiment:

```text
Compare target correction against model-level consistency regularization.
```

### Preserve and Personalize

Source: https://arxiv.org/abs/2505.19519

Preserve and Personalize explicitly frames personalization as distributional drift and uses regularization to preserve the pretrained distribution.

Plain version:

```text
It says personalization drifts, so constrain the model.
```

Our distinction:

```text
P&P is regularization-level preservation.
DADT is target-level preservation.
```

This is the most dangerous nearby work. We must not rely on wording alone. We need experiments showing that target correction helps even when regularization or prior preservation is controlled.

## Why Timestep / Region / Frequency Matter

### Prompt-to-Prompt / Cross-Attention Control

Source: https://arxiv.org/abs/2208.01626

Prompt-to-Prompt shows that cross-attention carries important text-image layout and editing information.

Why it matters for DADT:

```text
The subject and background should not be treated equally.
```

Possible use:

```text
Preserve target components inside subject-attended regions more strongly.
Align background-attended regions more strongly to the base prior.
```

### Min-SNR And P2

Sources:

- Min-SNR: https://arxiv.org/abs/2303.09556
- P2: https://arxiv.org/abs/2204.00227

These papers show that diffusion timesteps should not all be treated the same during training.

Why it matters for DADT:

```text
One global mixing coefficient is too naive.
```

But we must distinguish:

```text
Min-SNR / P2 reweight the loss.
DADT changes target components.
```

So a timestep-weighting baseline is necessary.

## What To Borrow From MixSD

Borrow:

```text
Forgetting can start from the target, not only from parameter movement.
Use base-model compatibility evidence before proposing a correction.
Evaluate both new-knowledge learning and old-capability retention.
```

Do not borrow directly:

```text
token-level mixing
simple global mixing rate
assuming two conditionals perfectly separate knowledge and style
using one NLL-like scalar as the whole proof
```

Diffusion has structure that LLM token SFT does not:

```text
timestep
frequency
spatial region
subject/background
VAE projection artifact
ordinary real-image domain gap
```

This structure should be the novelty of DADT.

## Proposed Logical Closure

### Step 1: Source Diagnosis

Question:

```text
Is the DreamBooth reference target gap larger than ordinary real-image target gap after removing VAE artifact?
```

This is Stage 1.4.

If no:

```text
Do not claim personalization-specific target gap.
Pivot to real-image-to-diffusion-prior projection alignment.
```

If yes:

```text
We can say personalization reference supervision contains a subject-specific target gap.
```

### Step 2: Forgetting Correlation

Question:

```text
Do subjects/images with larger cleaned target gap cause worse class-prior forgetting after fine-tuning?
```

This is the bridge from measurement to importance.

Needed result:

```text
larger clean gap -> worse ordinary class generation / diversity / prior retention
```

### Step 3: Selective Intervention

Question:

```text
Can we reduce the harmful part of the target gap without removing subject identity?
```

Needed result:

```text
subject similarity stays high
class prior improves
target compatibility improves
```

This is the actual method proof.

## Candidate DADT Method Ladder

### DADT-v0: Projection-Cleaned Target

Use Stage 1.4 decomposition:

```text
remove or downweight VAE/projection-like residual direction
```

This is the cleanest first version, but may be too conservative.

### DADT-v1: Timestep/Frequency-Aware Target

Use stronger alignment at timesteps/frequencies that mostly carry class/background structure, and weaker alignment where identity details appear.

Plain version:

```text
early/global structure: be careful
late/fine details: let the subject through
```

### DADT-v2: Subject/Background-Aware Target

Use a subject mask or cross-attention map:

```text
inside subject: preserve v_ref more
background: align more toward v_base
```

This is the most intuitive reviewer-facing version.

### DADT-v3: Sensitivity-Aware Target

Approximate whether a target component causes large updates in base-prior-sensitive directions.

Plain version:

```text
if learning this component shakes the base model too much, align it more
```

This is closest to MixSD's Fisher story, but it is harder and should come after simpler versions.

## Required Baselines

To make the story hard to dismiss, compare against:

```text
DreamBooth
DreamBooth + prior preservation
Textual Inversion or LoRA DreamBooth if feasible
simple global mixing: alpha * v_ref + (1 - alpha) * v_base
timestep loss weighting only
regularization / consistency surrogate for P&P or DCO
DADT
```

The global mixing baseline is important because reviewers may say:

```text
You just diluted the reference target.
```

DADT must beat that baseline or at least produce a better identity-prior tradeoff.

## What The Paper Should Not Claim

Do not claim:

```text
All reference deviation is bad.
r itself is harmful.
Base model is always right.
DADT is just MixSD for diffusion.
Simple velocity averaging is enough.
```

Do claim:

```text
Reference targets contain multiple deviation sources.
Some deviation is subject identity and should be preserved.
Some deviation is artifact/background/prior-harmful and should be aligned.
DADT is a target-construction method, complementary to replay and regularization.
```

## Reviewer-Ready One-Liner

```text
Personalization needs to move away from the base prior, but not every reference-induced denoising deviation is semantically necessary. DADT decomposes this deviation and aligns only the prior-harmful components while preserving subject identity.
```

Chinese version:

```text
个性化当然要偏离原模型，但 reference target 里的每个偏离都不一定值得学。DADT 要留下身份信息，削弱会伤害 prior 的偶然偏离。
```
