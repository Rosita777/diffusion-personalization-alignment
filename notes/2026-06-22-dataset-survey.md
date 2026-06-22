# Dataset Survey For Personalization Target Alignment

Date: 2026-06-22

Purpose: identify which datasets and evaluation sets are used by the main inspiration papers and nearest-neighbor personalization papers, then choose a practical dataset plan for DADT.

## Short Answer

Most personalization and anti-forgetting papers still use the DreamBooth / DreamBench benchmark as the central testbed:

```text
30 subjects
15 classes
4-6 reference images per subject
25 evaluation prompts per subject
```

For DADT, the most defensible first formal setup is:

```text
Stage 1 smoke test: a small DreamBooth subset with easy / standard / hard reference regimes.
Main measurement: all 30 DreamBooth subjects.
Scale-up / diversity check: CustomConcept101 or DreamBench++.
Optional style stress test: StyleDrop-style images if we later study style personalization.
```

## Core Reference Papers

### DreamBooth

Sources:

- https://arxiv.org/abs/2208.12242
- https://github.com/google/dreambooth
- https://huggingface.co/datasets/google/dreambooth

Dataset:

- 30 subjects across 15 classes.
- 9 live subjects, including dogs and cats.
- 21 object subjects.
- 4-6 images per subject.
- Images are captured by the authors or sourced from Unsplash.
- Images are taken under different conditions, environments, and viewpoints.
- The paper also defines 25 prompts: 20 recontextualization prompts and 5 property-modification prompts.

Why it matters for us:

DreamBooth is the de facto subject-personalization benchmark. It is small enough for controlled measurement, widely recognized, and already contains variation in environment and viewpoint. It should be our first benchmark.

### Preserve and Personalize

Sources:

- https://arxiv.org/abs/2505.19519
- https://arxiv.org/html/2505.19519v3
- https://openreview.net/forum?id=p4oYf6IbG5

Dataset and evaluation:

- Uses the DreamBooth benchmark as the primary personalization dataset.
- Reports 30 subjects, each with up to 6 example images.
- Uses the benchmark's 25 evaluation prompts per subject.
- Generates 4 images per prompt, producing 100 images per subject and 3,000 images across all 30 subjects.
- Evaluates with DINO, CLIP-I, and CLIP-T.
- Tests across SD 1.5, SDXL, and SD 3.0.
- Compares DreamBooth, DreamBooth-LoRA, Textual Inversion, Custom Diffusion, SVDiff, OFT, BLIP-Diffusion, and IP-Adapter.

Why it matters for us:

P&P is our dangerous nearest neighbor and uses DreamBooth as its main benchmark. If we use DreamBooth too, comparisons and reviewer positioning become much cleaner.

### MixSD

Sources:

- https://arxiv.org/abs/2605.16865
- https://arxiv.org/html/2605.16865v3

Datasets:

- KGFact: synthetic factual-recall corpus from a simulated world graph with novel entities.
- KGFunc: synthetic arithmetic function-acquisition corpus.
- SimpleQA: open-domain factual QA.
- General capability benchmarks: MMLU, GSM8K, MATH500, AIME2024, HumanEval.
- Knowledge editing stress test: MQuAKE.

Why it matters for us:

MixSD is not a diffusion personalization paper, so its datasets do not transfer directly. The useful lesson is experimental structure:

```text
controlled synthetic data for clean diagnosis
plus established benchmarks for credibility
```

For DADT, the analogous design is:

```text
controlled easy / standard / hard reference regimes
plus DreamBooth / CustomConcept101 / DreamBench++ benchmarks
```

### Spectral Progressive Diffusion

Sources:

- https://arxiv.org/abs/2605.18736
- https://arxiv.org/html/2605.18736v1

Datasets and benchmarks:

- MS-COCO validation prompts for image quality and prompt-alignment evaluation.
- GenEval and T2I-CompBench for compositional prompt alignment.
- VBench for video generation evaluation.
- Fine-tuning uses synthetic images generated from full-resolution models on 5K MS-COCO prompts.

Why it matters for us:

SPD is not a personalization benchmark paper. Its value is not its dataset choice but the frequency/timestep finding. For DADT, it supports frequency-aware analysis, while DreamBooth-style personalization datasets remain the correct main benchmark.

## Nearest-Neighbor Personalization Papers

### Direct Consistency Optimization

Sources:

- https://arxiv.org/abs/2402.12004
- https://arxiv.org/html/2402.12004v2
- https://openreview.net/forum?id=VazkRbCGxt
- https://github.com/kyungmnlee/dco

Dataset and evaluation:

- Subject personalization: DreamBooth dataset with 30 subjects and 4-6 images per subject.
- Style personalization: 10 images from the StyleDrop dataset.
- Subject + style composition: combines 30 DreamBooth subjects with 10 StyleDrop style images.
- Uses DINOv2 for image/subject similarity and SigLIP for image-text similarity.
- Evaluates Pareto curves under reward guidance scales.
- Uses comprehensive captions generated with LLaVA and manually filtered.

Why it matters for us:

DCO reinforces that DreamBooth is the central benchmark for robust personalization. It also tells us that `class-plus-context` captions are important; this supports our planned `c_base` ablation.

### Custom Diffusion / CustomConcept101

Sources:

- https://arxiv.org/abs/2212.04488
- https://github.com/adobe-research/custom-diffusion
- https://github.com/adobe-research/custom-diffusion/blob/main/customconcept101/README.md

Dataset:

- CustomConcept101 contains 101 concepts.
- Each concept has 3-15 images.
- Includes prompt files for single-concept and multi-concept evaluation.
- Used for evaluating customization and multi-concept composition.

Why it matters for us:

CustomConcept101 is larger and more diverse than DreamBooth, so it is useful after the first DreamBooth measurement works. It is not ideal as the first smoke test because its size and multi-concept prompt setup add complexity before we know whether off-priorness is measurable.

### DreamBench++

Sources:

- https://arxiv.org/html/2406.16855v2
- https://dreambenchplus.github.io

Dataset:

- 150 high-quality images.
- 1,350 prompts.
- Categories include objects, living subjects, and styles.
- Images are sourced from Unsplash, Rawpixel, Google Image Search, and authorized contributions.
- It explicitly aims to improve diversity over DreamBench and CustomConcept101.
- Uses GPT-4o-style multimodal evaluation aligned with human preference.

Why it matters for us:

DreamBench++ is attractive for a later formal benchmark because it includes difficulty and diversity. It is probably too heavy for Stage 1, but very useful for a paper-scale evaluation if our measurement signal holds.

## Recommended Dataset Plan For DADT

### Stage 1 Smoke Test

Use 3-5 classes from DreamBooth-style subjects:

- dog or cat: live subject;
- backpack or sneaker: everyday object;
- plush toy or toy: object with texture/shape identity;
- vase or clock: structured object if available;
- one visually unusual subject if available.

For each class/subject, construct three reference regimes:

```text
Easy / in-prior:
base-generated or highly typical class images.

Standard:
original DreamBooth reference images.

Hard / off-prior:
reference images or curated variants with unusual background, lighting, pose, crop, style, or strong subject-background correlation.
```

The hard regime should be declared as a controlled hard-personalization regime, not hidden cherry-picking.

### Main Measurement Experiment

Use all 30 DreamBooth subjects:

- compute target residuals against base-generated error floors;
- report `c_base` ablations: null, class prompt, class-plus-context prompt;
- stratify subjects by measured reference prior compatibility;
- test whether off-priorness differs across subject types.

### Formal Paper Expansion

If Stage 1 works:

- add CustomConcept101 for more concepts and broader composition;
- add DreamBench++ for diversity and human-aligned evaluation;
- keep StyleDrop only if we extend the method to style personalization.

## Why This Supports The "Small Trick"

Selecting harder reference images is scientifically acceptable if framed as a controlled variable:

```text
reference prior compatibility
```

The paper should not secretly cherry-pick hard examples. It should explicitly test:

```text
Easy < Standard < Hard
```

for target off-priorness and later prior drift. This turns the trick into a clean causal-style experimental design:

```text
as reference images move off-prior,
target off-priorness should increase,
and forgetting risk should increase.
```

## Current Decision

Use DreamBooth as the first official benchmark and create a controlled easy / standard / hard split around it. Do not start with CustomConcept101 or DreamBench++ until the DreamBooth measurement pipeline works.
