# Near-Neighbor Reading Notes

Date: 2026-06-21

Purpose: read the nearest-neighbor papers flagged by the DADT spec review and clarify how they affect positioning, baselines, and Stage 1 measurement.

## Direct Consistency Optimization

Sources:

- https://arxiv.org/abs/2402.12004
- https://openreview.net/forum?id=VazkRbCGxt
- https://github.com/kyungmnlee/dco

Core idea:

Direct Consistency Optimization (DCO) is a robust customization method for text-to-image diffusion personalization. It fine-tunes the model for reference consistency while penalizing deviation from the pretrained model, so the customized model keeps prompt fidelity and composes better with other concepts or separately tuned models.

Why it is dangerous for us:

```text
DCO already says low-shot personalization forgets pretrained knowledge,
and it directly controls deviation between fine-tuned and pretrained models.
```

So DCO is closer than ordinary DreamBooth prior preservation. It is not just replay; it is a model-objective-level anchor to the pretrained model.

How DADT must differ:

```text
DCO: constrain the personalized model / objective so updates stay close to the pretrained model.
DADT: measure and correct the reference-image denoising target before the model learns it.
```

This distinction is only convincing if experiments show that target correction changes the subject-fidelity / prior-preservation frontier beyond DCO or a faithful DCO-style surrogate.

Details relevant to our setup:

- The official implementation uses SDXL LoRA through diffusers and PEFT.
- The repo emphasizes comprehensive captions and base prompts without the learned token.
- It discourages rare token identifiers because they can inherit unfavorable semantics.

Implication for Stage 1:

The `c_base` ablation is not optional. DCO's use of base prompts suggests that `class-plus-context prompt` may be a better `v_base` condition than a bare class prompt for some subjects.

## SDS / DreamFusion

Sources:

- https://arxiv.org/abs/2209.14988
- https://ar5iv.labs.arxiv.org/html/2209.14988

Core idea:

DreamFusion introduced Score Distillation Sampling (SDS), using a frozen pretrained 2D text-to-image diffusion model as a prior for optimizing a differentiable 3D representation such as a NeRF. The diffusion model is not personalized; its score provides optimization guidance for another representation.

Why it matters:

SDS makes it clear that using a pretrained diffusion score as a reference or optimization signal is not new.

How DADT must differ:

```text
SDS: pretrained score guides sample / representation optimization.
DADT: pretrained score field is used to diagnose and correct reference-image personalization targets during training.
```

We should never claim that the use of `v_base` is novel by itself. The novelty must be:

- reference-target off-priorness as a personalization forgetting diagnosis;
- base-error-floor-controlled measurement;
- selective target correction for prior-harmful components.

## VSD / ProlificDreamer

Sources:

- https://arxiv.org/abs/2305.16213
- https://ar5iv.labs.arxiv.org/html/2305.16213

Core idea:

ProlificDreamer proposes Variational Score Distillation (VSD). It treats the optimized 3D scene as a distribution rather than a single point, explains SDS as a special case, and uses a learned score model, implemented with LoRA, to improve diversity and quality in text-to-3D generation.

Why it matters:

VSD is a strong reminder that comparing or subtracting score-like quantities has a long history. It also shows that naive score distillation can cause over-smoothing, over-saturation, and diversity loss.

Implication for DADT:

If DADT globally pulls `v_ref` toward `v_base`, reviewers can interpret it as a distillation-like smoothing method. The paper must show that:

- DADT is selective, not global;
- it preserves subject identity;
- it improves a Pareto frontier instead of merely regularizing harder.

VSD is not a direct personalization baseline. When DADT training begins, decide explicitly whether a VSD-style training-time score guidance surrogate is fair and feasible as a comparison.

## Min-SNR Weighting

Sources:

- https://arxiv.org/abs/2303.09556
- https://arxiv.org/html/2303.09556v3

Core idea:

Min-SNR treats diffusion training across timesteps as a multi-task learning problem. It observes conflicting optimization directions across timesteps and uses a clamped SNR-based loss weight to balance these tasks and speed convergence.

Why it matters:

DADT has timestep-aware components. Reviewers can ask whether gains come from timestep loss reweighting rather than target correction.

How DADT must differ:

```text
Min-SNR: reweight the loss across timesteps.
DADT: modify target components based on measured off-priorness.
```

Required baseline once training starts:

- DreamBooth or LoRA with Min-SNR-style timestep weighting;
- DADT with no loss reweighting;
- DADT plus weighting only as a separate ablation, not as the main comparison.

Implication for Stage 1:

Raw residuals across timesteps are not comparable unless normalized by timestep scale or SNR. Otherwise the heatmap may only reflect scheduler scale.

## P2 Weighting

Sources:

- https://arxiv.org/abs/2204.00227
- https://ar5iv.labs.arxiv.org/html/2204.00227

Core idea:

P2 weighting studies what diffusion models learn at different noise levels and prioritizes timesteps where denoising teaches perceptually rich visual content. It argues that slightly corrupted images can be restored from local detail, while more corrupted regimes require higher-level visual context.

Why it matters:

P2 supports the broad idea that timestep roles are not uniform. However, it is still a loss-weighting method, not target construction.

How it affects DADT:

P2 is useful as conceptual support for timestep-aware design, but we should not overclaim that it proves our exact frequency hypothesis. DADT must test its own off-priorness structure.

Required caution:

If DADT uses timestep/frequency gates, include P2-style weighting as a baseline or at least as an ablation family once training starts.

## Positioning After Reading

The strongest current positioning is:

```text
Existing methods preserve pretrained behavior by replaying data,
regularizing model updates, using pretrained scores as guidance,
or weighting timesteps differently.

DADT studies whether the reference-image denoising target itself is
already off-prior, then corrects target components before training.
```

What is now more important than before:

- Stage 1 must include a base-generated error floor.
- `c_base` should be measured as an ablation: null, class, and class-plus-context.
- Residuals must be normalized by timestep scale or SNR.
- DCO must be treated as a priority baseline, not a footnote.
- DADT training should not start until the measurement signal is confirmed.

## Stage 1 Go / No-Go

Proceed only if:

```text
reference-image residuals > base-generated residual floor
```

under at least the class prompt and one other `c_base` variant, and after timestep/SNR normalization.

If this does not hold, the current off-prior framing is too weak and should be revised before method work.
