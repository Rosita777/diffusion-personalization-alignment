# Off-Prior Measurement V0 Source Decomposition Stage 1.5B Prompt-Matched Controls

Status: complete. This prompt-matching diagnosis did not rescue the metric.

Purpose: test whether Stage 1.5A failed mainly because the ordinary-real controls were measured with coarse class prompts such as `a photo of a dog`. Stage 1.5B replaces those class prompts with image-specific prompt-matched descriptions and compares ordinary real images against base SD 1.5 images generated from the same descriptions.

This is a control-only diagnosis. It does not contain DreamBooth reference rows and should not be read as a personalization Go / Pivot experiment.

## Data

The run uses the same 16 COCO ordinary-real controls as Stage 1.5A, with prompt-matched descriptions listed in:

```text
data/manifests/ordinary_real_controls_stage15b_prompt_matched.yaml
```

For each prompt-matched ordinary-real image, SD 1.5 generated two base controls:

```text
generated controls: 32 images
ordinary real controls: 16 images
conditioning: prompt_matched
```

Generated images are cached locally under:

```text
data/cache/off_prior_measurement_v0/generated_controls_stage15b_prompt_matched/
```

Raw generated images and raw COCO images are not committed.

## Run Size

```text
manifest rows: 48
raw metric rows: 1200
timesteps: 50, 200, 500, 800, 999
noise seeds: 0, 1, 2, 3, 4
source groups: base_generated_control, ordinary_real_control
```

## Run Order

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched.yaml
export EXP=experiments/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched
export GEN=data/cache/off_prior_measurement_v0/generated_controls_stage15b_prompt_matched

CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.off_prior_measurement.prompt_matched_controls generate \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_stage15b_prompt_matched.yaml \
  --generated-root $GEN \
  --model-id data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5 \
  --device cuda \
  --dtype float16 \
  --resolution 512 \
  --seeds-per-prompt 2 \
  --num-inference-steps 30 \
  --guidance-scale 7.5

$PYTHON -m scripts.off_prior_measurement.prompt_matched_controls manifest \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_stage15b_prompt_matched.yaml \
  --generated-root $GEN \
  --seeds-per-prompt 2 \
  --output $EXP/manifests/prompt_matched_control_manifest.csv

$PYTHON -m scripts.off_prior_measurement.source_decomp_manifest \
  --reference-manifest $EXP/manifests/empty_reference_manifest.csv \
  --control-manifest $EXP/manifests/prompt_matched_control_manifest.csv \
  --ordinary-real-manifest data/manifests/ordinary_real_controls_stage15b_prompt_matched.yaml \
  --roundtrip-root data/cache/off_prior_measurement_v0/source_decomp_stage15b_prompt_matched_roundtrip \
  --output $EXP/manifests/source_decomp_manifest.csv

CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.off_prior_measurement.source_decomp_roundtrip \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv

# Measurement was run with 8 rank shards on GPUs 0-7, then merged.
$PYTHON -m scripts.off_prior_measurement.source_decomp_measure \
  --config $CONFIG \
  --manifest $EXP/manifests/source_decomp_manifest.csv \
  --world-size 8 \
  --merge-shards

$PYTHON -m scripts.off_prior_measurement.source_decomp_summarize \
  --raw-metrics $EXP/measurements/raw_source_decomp_metrics.csv \
  --output-dir $EXP

$PYTHON -m scripts.off_prior_measurement.source_decomp_plot \
  --summary-dir $EXP/summaries \
  --figures-dir $EXP/figures

$PYTHON -m scripts.off_prior_measurement.source_decomp_conclusion \
  --experiment-dir $EXP
```

## Result

Prompt-matched real-domain gap:

```text
class      prompt-matched real-domain gap
backpack                        0.015855
cat                             0.008615
dog                             0.012796
vase                            0.014554
mean                            0.012955
```

Compared with Stage 1.5A class-only conditioning:

```text
class      class-only gap  prompt-matched gap  delta
backpack        0.016128            0.015855  -0.000273
cat             0.010957            0.008615  -0.002343
dog             0.012643            0.012796   0.000153
vase            0.015867            0.014554  -0.001313
mean            0.013899            0.012955  -0.000944
```

The average real-domain gap drops by only about 6.8%.

## Interpretation

Prompt mismatch is not the main reason Stage 1.5A failed. More specific prompts slightly reduce the ordinary-real gap for cat, vase, and backpack, but the gap remains positive for all four classes and remains close to the class-only result.

The current measurement is still dominated by the base-generated versus ordinary-real image gap and roundtrip/projection artifact. This means the next diagnosis should target the metric itself, not just the text prompt: for example, alternate residual definitions, VAE/preprocessing controls, or measuring target alignment in a representation less dominated by image projection artifacts.

## Decision

Do not start Stage 2 personalization fine-tuning from this metric. Stage 1.5B narrows the failure cause: coarse class prompts contribute a little, but they do not explain the failure.
