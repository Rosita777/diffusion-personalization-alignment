# MixSD Reread For DADT

Date: 2026-06-26

Status: first reread note. Use this to guide the DADT story, not as proof that the diffusion hypothesis already holds.

Primary source:

- https://arxiv.org/abs/2605.16865
- https://arxiv.org/html/2605.16865v3

## One-Sentence Takeaway

MixSD is powerful because it says forgetting can start from the supervision target itself, then fixes the target before training. This is exactly the kind of story DADT wants, but diffusion needs its own target-gap evidence.

## Beginner Version

普通 SFT 像是让模型背一批标准答案。问题是，这些标准答案虽然事实正确，但可能不像模型原本会自然说出来的话。模型硬背这些答案时，可能不只是学了新知识，也把原来的说话习惯和通用能力弄坏了。

MixSD 的想法是：不要直接拿外部标准答案硬训。让 base model 自己当老师，在两个状态下各说一遍：

- `expert conditional`: 给它看新知识，所以它知道该说什么事实。
- `naive conditional`: 不给它看新知识，所以它保持原来的语言分布和表达习惯。

然后 MixSD 把这两种 token 混起来，得到一个更像 base model 自己会说、但仍然含有新知识的训练 target。

## What MixSD Is Really Claiming

MixSD 的关键 claim 不是“参数更新太多会忘”，而是：

```text
SFT target distribution gap can cause forgetting.
```

也就是说，遗忘可能来自老师给的答案本身太偏。训练还没开始之前，target 已经和 base model 的 native distribution 不匹配。

这和我们的 DADT 对应关系是：

```text
LLM SFT target 是否离 base LM distribution 太远
~= diffusion personalization denoising target 是否离 base score / velocity field 太远
```

但我们现在的 Stage 1.3 结果提醒我们：diffusion 里这个 gap 不能粗暴测，因为 VAE roundtrip 和普通真实图片 domain gap 会混进去。

## Method Mechanism

MixSD 不是额外加 replay，也不是只加 regularization。它改的是监督信号。

对每个生成位置，它用同一个 base/reference model 得到两个候选 token 分布：

- expert side: 条件里包含新知识，负责保事实。
- naive side: 条件里不包含新知识，负责保 base distribution。

然后用一个 mixing rate 选择 token 来源。mixing rate 越大，naive/base-prior token 越多；mixing rate 为 0 时就是纯 expert supervision。

训练本身还是普通 NLL，只是训练 target 已经被重新构造过。

对我们很重要的一点：

```text
MixSD 不是简单平均两个 logits。
它是在构造一条更自然的 target sequence。
```

所以 DADT 也不应该只是：

```text
v_mixed = alpha * v_ref + (1 - alpha) * v_base
```

更像 MixSD 的 diffusion 版本应该是 trajectory-level / timestep-aware / frequency-aware / region-aware target construction。

## Evidence Pattern

MixSD 的证据分两层。

### NLL Evidence

NLL 可以通俗理解成：

```text
base model 读这个 target 费不费劲。
```

如果一个 SFT target 在 base model 下 NLL 很高，说明这个答案虽然正确，但对 base model 来说“不顺口”。MixSD 发现普通 SFT target 有很多 high-NLL token，而 MixSD 构造出来的 target 大幅减少 high-NLL token，同时仍保留大部分事实相关 token。

DADT 需要对应的 diffusion 证据：

```text
reference denoising target 对 base score field 来说是否“不顺”？
```

但我们不能只看一个 raw residual。Stage 1.4 现在要拆清楚：

```text
VAE/projection artifact
ordinary real-image domain gap
DreamBooth subject-specific mismatch
```

只有第三项成立，personalization-specific DADT story 才最强。

### Fisher Evidence

Fisher 可以通俗理解成：

```text
模型哪些地方最敏感，最不能乱碰。
```

MixSD 发现，forgetting 和普通参数移动距离不一定强相关；更关键的是更新有没有打到 Fisher-sensitive directions。MixSD 的目标构造减少了这种敏感方向上的移动。

这启发我们后面可以加一个更强的 diffusion 诊断：

```text
target alignment 是否减少 UNet 在 base-prior-sensitive directions 上的梯度？
```

这不是 Stage 1.4 必做项，但可以作为后续 reviewer-strengthening evidence。

## Direct Mapping To DADT

| MixSD | DADT candidate |
| --- | --- |
| SFT target sequence | personalization denoising target / trajectory |
| high-NLL token | high off-prior residual component |
| factual token to preserve | subject-identity component to preserve |
| naive conditional | class-prior / base-score trajectory |
| expert conditional | subject/reference-aware trajectory |
| mixing rate | timestep/frequency/region-dependent alignment strength |
| Fisher-sensitive LM directions | base-prior-sensitive UNet directions |

The key phrase for the paper:

```text
not all target deviation is bad; the useful subject-specific part should be preserved, while prior-harmful target components should be aligned.
```

## How This Changes Our Current Plan

Stage 1.4 is the right next gate because MixSD-style evidence must first prove the target gap is real and specific.

The question should be:

```text
After removing VAE/projection artifact and ordinary real-image domain gap,
are DreamBooth reference targets still more off-prior than ordinary real images?
```

If yes, DADT can say:

```text
personalization supervision has a subject-specific target distribution gap.
```

If no, the project should pivot toward:

```text
real-image-to-diffusion-prior projection alignment
```

which is still publishable but less personalization-specific.

## What Not To Copy

Do not sell DADT as a direct MixSD transplant. Reviewers may reject that as ordinary distillation.

Avoid:

```text
simple global velocity averaging
single scalar off-prior score
only weight regularization
only replay/prior preservation
```

Prefer:

```text
source-decomposed target-gap evidence
timestep-aware alignment
frequency-aware alignment
subject/background or region-aware alignment
identity-preservation checks
prior-retention checks
```

## Paper Story Template

MixSD's story:

```text
SFT forgetting exists
-> target distribution gap explains part of it
-> NLL and Fisher support the mechanism
-> mixed self-distilled targets improve memorization-retention tradeoff
```

DADT's desired story:

```text
diffusion personalization prior drift exists
-> personalization denoising targets contain source-specific target gaps
-> source decomposition and trajectory diagnostics support the mechanism
-> distribution-aligned denoising targets improve identity-retention / prior-retention tradeoff
```

## Immediate Action

Use MixSD as a writing and evidence template, not as enough evidence by itself. The immediate experimental need remains Stage 1.4 ordinary-real controls:

```text
dog_real_00.jpg
cat_real_00.jpg
backpack_real_00.jpg
vase_real_00.jpg
```

After Stage 1.4, revisit this note and decide whether the paper story should stay personalization-specific or pivot to real-image projection alignment.
