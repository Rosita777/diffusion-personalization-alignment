# Stage 1 V2 Prior-Compatibility Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Stage 1 from a standard DreamBooth smoke test into a prior-compatibility ladder experiment that measures whether harder reference images induce denoising targets farther from the pretrained SD 1.5 field.

**Architecture:** Keep the existing `scripts/off_prior_measurement/` pipeline and add one focused hard-reference construction module. The upgraded manifest carries `reference_regime`, `hardness_axis`, `variant_id`, and `source_standard_image` through measurement, summary, plotting, and conclusion generation so the ladder can be analyzed without changing the diffusion backend.

**Tech Stack:** Python 3.10 in `/home/deepseek_VG/.conda/envs/dyme`, PyTorch, diffusers, Pillow, NumPy, pandas, SciPy, PyYAML, matplotlib, pytest, local ModelScope SD 1.5 cache, GitHub DreamBooth dataset access.

**Implementation Status:** Plan written on 2026-06-22 after the v0 smoke test returned No-Go. Code execution has not started for v2.

---

## Scope

This plan implements Stage 1 v2 measurement only.

In scope:

- 8-subject prior-compatibility smoke run:
  `dog`, `cat`, `backpack`, `vase`, `colorful_sneaker`, `shiny_sneaker`, `fancy_boot`, `grey_sloth_plushie`.
- Existing controls:
  `base_easy_control`, `base_hard_control`, `vae_roundtrip_control`.
- Existing standard DreamBooth references.
- New deterministic hard-reference variants created from standard references.
- Ladder summaries:
  `easy_control`, `standard_reference`, `hard_reference`, `hard_control`, `roundtrip_control`.
- Go / No-Go logic from `docs/superpowers/specs/2026-06-22-prior-compatibility-ladder-design.md`.
- Multi-GPU measurement commands using rank sharding.
- Documentation updates for the v2 experiment directory.

Out of scope:

- Personalization fine-tuning.
- DADT target correction.
- SAM masks or true foreground/background segmentation.
- All-30-subject paper-scale run.
- SDXL or non-SD1.5 models.

## File Structure

Create:

```text
configs/off_prior_measurement_v0/ladder_v2.yaml
data/manifests/dreambooth_ladder_subjects.yaml
scripts/off_prior_measurement/hard_references.py
tests/off_prior_measurement/test_hard_references.py
experiments/off_prior_measurement_v0/ladder_v2/README.md
```

Modify:

```text
README.md
scripts/off_prior_measurement/__init__.py
scripts/off_prior_measurement/config.py
scripts/off_prior_measurement/dreambooth_data.py
scripts/off_prior_measurement/generate_controls.py
scripts/off_prior_measurement/roundtrip_controls.py
scripts/off_prior_measurement/summarize.py
scripts/off_prior_measurement/plot.py
scripts/off_prior_measurement/write_conclusion.py
tests/off_prior_measurement/test_config.py
tests/off_prior_measurement/test_dreambooth_data.py
tests/off_prior_measurement/test_generate_controls.py
tests/off_prior_measurement/test_roundtrip_controls.py
tests/off_prior_measurement/test_summarize.py
tests/off_prior_measurement/test_plot.py
tests/off_prior_measurement/test_write_conclusion.py
```

Generated but not committed:

```text
data/cache/off_prior_measurement_v0/hard_references/
experiments/off_prior_measurement_v0/ladder_v2/**/*.log
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics_rank*.csv
outputs/off_prior_measurement_v0/ladder_v2/
```

Curated and commit-worthy after the run:

```text
experiments/off_prior_measurement_v0/ladder_v2/config_resolved.yaml
experiments/off_prior_measurement_v0/ladder_v2/manifests/reference_manifest.csv
experiments/off_prior_measurement_v0/ladder_v2/manifests/hard_reference_manifest.csv
experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics.csv
experiments/off_prior_measurement_v0/ladder_v2/summaries/base_floor.csv
experiments/off_prior_measurement_v0/ladder_v2/summaries/scored_metrics.csv
experiments/off_prior_measurement_v0/ladder_v2/summaries/regime_summary.csv
experiments/off_prior_measurement_v0/ladder_v2/summaries/subject_summary.csv
experiments/off_prior_measurement_v0/ladder_v2/summaries/ladder_summary.csv
experiments/off_prior_measurement_v0/ladder_v2/figures/control_distribution.png
experiments/off_prior_measurement_v0/ladder_v2/figures/timestep_curves.png
experiments/off_prior_measurement_v0/ladder_v2/figures/frequency_heatmap.png
experiments/off_prior_measurement_v0/ladder_v2/figures/ladder_timestep_heatmap.png
experiments/off_prior_measurement_v0/ladder_v2/figures/hardness_frequency_heatmap.png
experiments/off_prior_measurement_v0/ladder_v2/conclusion.md
```

## Data Contract

The v2 combined manifest must include these columns:

```text
subject_id
image_id
image_path
source_group
reference_regime
hardness_axis
source_standard_image
variant_id
transform_parameters
class_name
class_prompt
class_context_prompt
conditioning_key
conditioning_prompt
```

Allowed `source_group` values:

```text
base_easy_control
base_hard_control
dreambooth_reference
dreambooth_hard_reference
vae_roundtrip_control
```

Allowed `reference_regime` values:

```text
easy_control
hard_control
standard_reference
hard_reference
roundtrip_control
```

Allowed `hardness_axis` values:

```text
none
crop
color_light
high_frequency
clutter_background
subject_background_binding
```

## Run Order

The v2 run order after implementation is:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2

$PYTHON -m scripts.off_prior_measurement.generate_controls --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.dreambooth_data --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.hard_references --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
$PYTHON -m scripts.off_prior_measurement.roundtrip_controls --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv
$PYTHON -m scripts.off_prior_measurement.summarize --raw-metrics $EXP/measurements/raw_metrics.csv --output-dir $EXP
$PYTHON -m scripts.off_prior_measurement.plot --scored-metrics $EXP/summaries/scored_metrics.csv --figures-dir $EXP/figures
$PYTHON -m scripts.off_prior_measurement.write_conclusion --experiment-dir $EXP
```

The `dreambooth_data` command writes a combined manifest whose hard-reference image paths are deterministic. The `hard_references` command must run before measurement so those paths exist.

---

### Task 1: Extend Config For Ladder V2

**Files:**

- Modify: `scripts/off_prior_measurement/config.py`
- Modify: `tests/off_prior_measurement/test_config.py`
- Create: `data/manifests/dreambooth_ladder_subjects.yaml`
- Create: `configs/off_prior_measurement_v0/ladder_v2.yaml`

- [ ] **Step 1: Write the failing config test**

Append this test to `tests/off_prior_measurement/test_config.py`:

```python
def test_load_config_parses_ladder_v2_fields(tmp_path):
    subject_path = tmp_path / "subjects.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: colorful_sneaker
    hf_subset: colorful_sneaker
    class_name: sneaker
    class_prompt: a photo of a sneaker
    class_context_prompt: a photo of a sneaker on the floor
    hard_control_prompt: a photo of a colorful sneaker under neon light on reflective metal
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_name: ladder_v2
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github
subject_manifest: {subject_path}
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/ladder_v2
debug_output_dir: outputs/off_prior_measurement_v0/ladder_v2
timesteps: [50, 200]
noise_seeds: [0, 1]
conditionings: ["null", "class", "class_context"]
control_images_per_subject: 2
batch_size: 1
save_debug_tensors: false
hard_reference_variants:
  - crop_large_subject
  - low_light_color_shift
hard_reference_limit_per_subject: 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment_name == "ladder_v2"
    assert config.hard_reference_variants == ["crop_large_subject", "low_light_color_shift"]
    assert config.hard_reference_limit_per_subject == 3
```

- [ ] **Step 2: Run the new test and confirm it fails**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_config.py::test_load_config_parses_ladder_v2_fields -v
```

Expected:

```text
FAILED
AttributeError: 'ExperimentConfig' object has no attribute 'hard_reference_variants'
```

- [ ] **Step 3: Extend the config dataclass**

In `scripts/off_prior_measurement/config.py`, add these fields to `ExperimentConfig` after `dataset_source`:

```python
    hard_reference_variants: list[str] | None = None
    hard_reference_limit_per_subject: int | None = None
```

In `load_config`, before returning `ExperimentConfig`, add:

```python
    hard_reference_variants = raw.get(
        "hard_reference_variants",
        [
            "crop_large_subject",
            "crop_small_subject",
            "low_light_color_shift",
            "high_saturation_color_shift",
            "background_clutter_overlay",
            "edge_reflection_texture",
        ],
    )
    hard_reference_limit = raw.get("hard_reference_limit_per_subject")
```

Then pass these fields into the returned dataclass:

```python
        hard_reference_variants=[str(item) for item in hard_reference_variants],
        hard_reference_limit_per_subject=None
        if hard_reference_limit is None
        else int(hard_reference_limit),
```

- [ ] **Step 4: Run config tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_config.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Create the v2 subject manifest**

Create `data/manifests/dreambooth_ladder_subjects.yaml` with:

```yaml
subjects:
  - subject_id: dog
    hf_subset: dog
    class_name: dog
    class_prompt: a photo of a dog
    class_context_prompt: a photo of a dog in a natural scene
    hard_control_prompt: a photo of a dog under dramatic colored lighting in a cluttered room
  - subject_id: cat
    hf_subset: cat
    class_name: cat
    class_prompt: a photo of a cat
    class_context_prompt: a photo of a cat indoors
    hard_control_prompt: a photo of a cat under colored nightclub lighting beside reflective metal
  - subject_id: backpack
    hf_subset: backpack
    class_name: backpack
    class_prompt: a photo of a backpack
    class_context_prompt: a photo of a backpack on a table
    hard_control_prompt: a photo of a backpack half cropped in a crowded market at night
  - subject_id: vase
    hf_subset: vase
    class_name: vase
    class_prompt: a photo of a vase
    class_context_prompt: a photo of a vase in a room
    hard_control_prompt: a photo of a vase in snow with the Eiffel Tower in the background
  - subject_id: colorful_sneaker
    hf_subset: colorful_sneaker
    class_name: sneaker
    class_prompt: a photo of a sneaker
    class_context_prompt: a photo of a sneaker on the floor
    hard_control_prompt: a photo of a colorful sneaker under neon light on reflective metal
  - subject_id: shiny_sneaker
    hf_subset: shiny_sneaker
    class_name: sneaker
    class_prompt: a photo of a sneaker
    class_context_prompt: a photo of a sneaker on the floor
    hard_control_prompt: a photo of a shiny sneaker with strong reflections in a crowded shop window
  - subject_id: fancy_boot
    hf_subset: fancy_boot
    class_name: boot
    class_prompt: a photo of a boot
    class_context_prompt: a photo of a boot on the floor
    hard_control_prompt: a photo of a fancy boot half cropped under red stage lighting
  - subject_id: grey_sloth_plushie
    hf_subset: grey_sloth_plushie
    class_name: plushie
    class_prompt: a photo of a plushie
    class_context_prompt: a photo of a plushie on a table
    hard_control_prompt: a photo of a grey sloth plushie in a cluttered toy store under mixed lighting
```

- [ ] **Step 6: Create the v2 config**

Create `configs/off_prior_measurement_v0/ladder_v2.yaml` with:

```yaml
experiment_name: ladder_v2
model_id: data/cache/modelscope/AI-ModelScope/stable-diffusion-v1-5
prediction_type: epsilon
device: cuda
dtype: float16
resolution: 512
dataset_repo: google/dreambooth
dataset_source: github
subject_manifest: data/manifests/dreambooth_ladder_subjects.yaml
cache_dir: data/cache/off_prior_measurement_v0
output_dir: experiments/off_prior_measurement_v0/ladder_v2
debug_output_dir: outputs/off_prior_measurement_v0/ladder_v2
timesteps: [50, 200, 500, 800, 999]
noise_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
conditionings: ["null", "class", "class_context"]
control_images_per_subject: 4
batch_size: 1
save_debug_tensors: false
hard_reference_variants:
  - crop_large_subject
  - crop_small_subject
  - low_light_color_shift
  - high_saturation_color_shift
  - background_clutter_overlay
  - edge_reflection_texture
hard_reference_limit_per_subject: 4
```

- [ ] **Step 7: Validate config loading**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
from scripts.off_prior_measurement.config import load_config
config = load_config("configs/off_prior_measurement_v0/ladder_v2.yaml")
print(config.experiment_name)
print(len(config.subjects))
print(config.hard_reference_variants)
print(config.hard_reference_limit_per_subject)
PY
```

Expected:

```text
ladder_v2
8
['crop_large_subject', 'crop_small_subject', 'low_light_color_shift', 'high_saturation_color_shift', 'background_clutter_overlay', 'edge_reflection_texture']
4
```

- [ ] **Step 8: Commit**

Run:

```bash
git add scripts/off_prior_measurement/config.py tests/off_prior_measurement/test_config.py data/manifests/dreambooth_ladder_subjects.yaml configs/off_prior_measurement_v0/ladder_v2.yaml
git commit -m "feat: add ladder v2 config"
```

---

### Task 2: Add Deterministic Hard Reference Generation

**Files:**

- Create: `scripts/off_prior_measurement/hard_references.py`
- Create: `tests/off_prior_measurement/test_hard_references.py`
- Modify: `scripts/off_prior_measurement/__init__.py`

- [ ] **Step 1: Write hard-reference unit tests**

Create `tests/off_prior_measurement/test_hard_references.py` with:

```python
import json

import pandas as pd
from PIL import Image

from scripts.off_prior_measurement.hard_references import (
    VARIANT_TO_AXIS,
    apply_variant,
    build_hard_reference_manifest,
    generate_hard_references_from_manifest,
)


def test_apply_variant_changes_pixels_deterministically(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (120, 120, 120)).save(image_path)

    first = apply_variant(Image.open(image_path), "low_light_color_shift")
    second = apply_variant(Image.open(image_path), "low_light_color_shift")

    assert first.size == (32, 32)
    assert list(first.getdata()) == list(second.getdata())
    assert first.getpixel((0, 0)) != (120, 120, 120)


def test_build_hard_reference_manifest_adds_variant_fields(tmp_path):
    source_path = tmp_path / "dataset" / "dog" / "00.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(source_path)
    reference_manifest = tmp_path / "reference_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": str(source_path),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "source_standard_image": "",
                "variant_id": "",
                "transform_parameters": "{}",
                "class_name": "dog",
                "class_prompt": "a photo of a dog",
                "class_context_prompt": "a photo of a dog in a natural scene",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_manifest, index=False)

    manifest = build_hard_reference_manifest(
        reference_manifest_path=reference_manifest,
        hard_root=tmp_path / "hard",
        variants=["crop_large_subject", "edge_reflection_texture"],
    )

    assert len(manifest) == 2
    assert set(manifest["source_group"]) == {"dreambooth_hard_reference"}
    assert set(manifest["reference_regime"]) == {"hard_reference"}
    assert set(manifest["hardness_axis"]) == {VARIANT_TO_AXIS["crop_large_subject"], VARIANT_TO_AXIS["edge_reflection_texture"]}
    assert set(manifest["source_standard_image"]) == {str(source_path)}
    assert all(json.loads(value) for value in manifest["transform_parameters"])


def test_generate_hard_references_from_manifest_writes_images(tmp_path):
    source_path = tmp_path / "dataset" / "dog" / "00.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), (100, 100, 100)).save(source_path)
    reference_manifest = tmp_path / "reference_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": str(source_path),
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_manifest, index=False)

    hard_root = generate_hard_references_from_manifest(
        reference_manifest_path=reference_manifest,
        hard_root=tmp_path / "hard",
        variants=["high_saturation_color_shift"],
    )

    outputs = sorted(hard_root.glob("dog/*.png"))
    assert len(outputs) == 1
    assert outputs[0].name == "00__high_saturation_color_shift.png"
```

- [ ] **Step 2: Run the new tests and confirm import failure**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_hard_references.py -v
```

Expected:

```text
FAILED
ModuleNotFoundError: No module named 'scripts.off_prior_measurement.hard_references'
```

- [ ] **Step 3: Create hard reference module**

Create `scripts/off_prior_measurement/hard_references.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings

VARIANT_TO_AXIS = {
    "crop_large_subject": "crop",
    "crop_small_subject": "crop",
    "low_light_color_shift": "color_light",
    "high_saturation_color_shift": "color_light",
    "background_clutter_overlay": "clutter_background",
    "edge_reflection_texture": "high_frequency",
}


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * fraction))
    crop_h = max(1, int(height * fraction))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BICUBIC)


def _small_subject(image: Image.Image) -> Image.Image:
    width, height = image.size
    canvas = ImageOps.mirror(image).resize((width, height), Image.Resampling.BICUBIC)
    canvas = ImageEnhance.Brightness(canvas).enhance(0.55)
    canvas = ImageEnhance.Contrast(canvas).enhance(0.75)
    subject = image.resize((int(width * 0.58), int(height * 0.58)), Image.Resampling.BICUBIC)
    paste_x = int(width * 0.30)
    paste_y = int(height * 0.28)
    canvas.paste(subject, (paste_x, paste_y))
    return canvas


def _clutter_overlay(image: Image.Image) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    width, height = output.size
    colors = [(255, 60, 60, 76), (40, 160, 255, 76), (255, 220, 40, 76), (40, 220, 140, 76)]
    boxes = [
        (0.02, 0.04, 0.26, 0.18),
        (0.70, 0.08, 0.96, 0.24),
        (0.05, 0.72, 0.30, 0.92),
        (0.72, 0.68, 0.96, 0.94),
    ]
    for color, box in zip(colors, boxes):
        draw.rectangle(tuple(int(value * (width if idx % 2 == 0 else height)) for idx, value in enumerate(box)), fill=color)
    return output


def _edge_reflection(image: Image.Image) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    width, height = output.size
    for offset in range(-height, width, 14):
        draw.line((offset, height, offset + height, 0), fill=(255, 255, 255, 70), width=3)
    return ImageEnhance.Contrast(output).enhance(1.25)


def apply_variant(image: Image.Image, variant_id: str) -> Image.Image:
    image = image.convert("RGB")
    if variant_id == "crop_large_subject":
        return _center_crop(image, 0.68)
    if variant_id == "crop_small_subject":
        return _small_subject(image)
    if variant_id == "low_light_color_shift":
        dark = ImageEnhance.Brightness(image).enhance(0.45)
        return ImageEnhance.Color(dark).enhance(0.70)
    if variant_id == "high_saturation_color_shift":
        saturated = ImageEnhance.Color(image).enhance(1.85)
        return ImageEnhance.Contrast(saturated).enhance(1.20)
    if variant_id == "background_clutter_overlay":
        return _clutter_overlay(image)
    if variant_id == "edge_reflection_texture":
        return _edge_reflection(image)
    raise ValueError(f"Unsupported hard reference variant: {variant_id}")


def _variant_parameters(variant_id: str) -> str:
    parameters = {
        "variant_id": variant_id,
        "hardness_axis": VARIANT_TO_AXIS[variant_id],
        "deterministic": True,
    }
    return json.dumps(parameters, sort_keys=True)


def _hard_image_path(hard_root: Path, subject_id: str, image_id: str, variant_id: str) -> Path:
    return hard_root / subject_id / f"{image_id}__{variant_id}.png"


def build_hard_reference_manifest(
    reference_manifest_path: str | Path,
    hard_root: str | Path,
    variants: list[str],
) -> pd.DataFrame:
    reference = read_csv_preserve_strings(reference_manifest_path)
    hard_root = Path(hard_root)
    rows: list[dict[str, object]] = []
    for row in reference.to_dict("records"):
        for variant_id in variants:
            if variant_id not in VARIANT_TO_AXIS:
                raise ValueError(f"Unsupported hard reference variant: {variant_id}")
            new_row = dict(row)
            new_row["image_path"] = str(_hard_image_path(hard_root, row["subject_id"], row["image_id"], variant_id))
            new_row["source_group"] = "dreambooth_hard_reference"
            new_row["reference_regime"] = "hard_reference"
            new_row["hardness_axis"] = VARIANT_TO_AXIS[variant_id]
            new_row["source_standard_image"] = str(row["image_path"])
            new_row["variant_id"] = variant_id
            new_row["transform_parameters"] = _variant_parameters(variant_id)
            rows.append(new_row)
    return pd.DataFrame(rows)


def generate_hard_references_from_manifest(
    reference_manifest_path: str | Path,
    hard_root: str | Path,
    variants: list[str],
) -> Path:
    reference = read_csv_preserve_strings(reference_manifest_path)
    hard_root = Path(hard_root)
    unique_images = reference[["subject_id", "image_id", "image_path"]].drop_duplicates()
    for row in unique_images.to_dict("records"):
        image = Image.open(row["image_path"]).convert("RGB")
        for variant_id in variants:
            output_path = _hard_image_path(hard_root, row["subject_id"], row["image_id"], variant_id)
            if output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            apply_variant(image, variant_id).save(output_path)
    return hard_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-manifest", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    hard_root = config.cache_dir / "hard_references"
    root = generate_hard_references_from_manifest(
        reference_manifest_path=args.reference_manifest,
        hard_root=hard_root,
        variants=config.hard_reference_variants or [],
    )
    print(root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add module export**

In `scripts/off_prior_measurement/__init__.py`, add `"hard_references"` to `__all__`:

```python
    "hard_references",
```

- [ ] **Step 5: Run hard-reference tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_hard_references.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/off_prior_measurement/hard_references.py scripts/off_prior_measurement/__init__.py tests/off_prior_measurement/test_hard_references.py
git commit -m "feat: add hard reference variants"
```

---

### Task 3: Upgrade Manifests To Carry Ladder Fields

**Files:**

- Modify: `scripts/off_prior_measurement/dreambooth_data.py`
- Modify: `scripts/off_prior_measurement/generate_controls.py`
- Modify: `scripts/off_prior_measurement/roundtrip_controls.py`
- Modify: `tests/off_prior_measurement/test_dreambooth_data.py`
- Modify: `tests/off_prior_measurement/test_generate_controls.py`
- Modify: `tests/off_prior_measurement/test_roundtrip_controls.py`

- [ ] **Step 1: Update the reference manifest test**

In `tests/off_prior_measurement/test_dreambooth_data.py`, update `test_build_reference_manifest_from_local_images` so the expected columns are:

```python
    assert list(manifest.columns) == [
        "subject_id",
        "image_id",
        "image_path",
        "source_group",
        "reference_regime",
        "hardness_axis",
        "source_standard_image",
        "variant_id",
        "transform_parameters",
        "class_name",
        "class_prompt",
        "class_context_prompt",
        "conditioning_key",
        "conditioning_prompt",
    ]
    assert set(manifest["reference_regime"]) == {"standard_reference"}
    assert set(manifest["hardness_axis"]) == {"none"}
    assert set(manifest["variant_id"]) == {""}
```

- [ ] **Step 2: Update the combined manifest test**

In `tests/off_prior_measurement/test_dreambooth_data.py`, update `test_write_combined_manifest_includes_references_controls_and_roundtrip` by creating one expected hard image before calling `write_combined_manifest`:

```python
    hard_root = tmp_path / "cache" / "hard_references" / "dog"
    hard_root.mkdir(parents=True)
    (hard_root / "00__crop_large_subject.png").write_bytes(b"hard-reference")
```

Add these config fields to the test config:

```yaml
hard_reference_variants:
  - crop_large_subject
hard_reference_limit_per_subject: 1
```

Then assert:

```python
    assert set(manifest["source_group"]) == {
        "dreambooth_reference",
        "dreambooth_hard_reference",
        "base_easy_control",
        "base_hard_control",
        "vae_roundtrip_control",
    }
    assert set(manifest["reference_regime"]) == {
        "standard_reference",
        "hard_reference",
        "easy_control",
        "hard_control",
        "roundtrip_control",
    }
    assert "hard_reference_manifest.csv" in {path.name for path in (tmp_path / "experiment" / "manifests").iterdir()}
```

- [ ] **Step 3: Update control manifest test**

In `tests/off_prior_measurement/test_generate_controls.py`, add:

```python
    assert set(manifest["hardness_axis"]) == {"none", "clutter_background"}
    assert set(manifest["variant_id"]) == {""}
```

The expected mapping is:

```text
base_easy_control -> hardness_axis none
base_hard_control -> hardness_axis clutter_background
```

- [ ] **Step 4: Update roundtrip manifest test**

In `tests/off_prior_measurement/test_roundtrip_controls.py`, add a test:

```python
def test_build_roundtrip_manifest_preserves_ladder_metadata(tmp_path):
    reference_manifest = tmp_path / "reference_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": str(tmp_path / "dog" / "00.png"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "source_standard_image": "",
                "variant_id": "",
                "transform_parameters": "{}",
                "class_name": "dog",
                "class_prompt": "a photo of a dog",
                "class_context_prompt": "a photo of a dog in a natural scene",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_manifest, index=False)

    manifest = build_roundtrip_manifest(reference_manifest, tmp_path / "roundtrip")

    assert set(manifest["source_group"]) == {"vae_roundtrip_control"}
    assert set(manifest["reference_regime"]) == {"roundtrip_control"}
    assert set(manifest["hardness_axis"]) == {"none"}
```

- [ ] **Step 5: Run manifest tests and confirm failures**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/off_prior_measurement/test_dreambooth_data.py \
  tests/off_prior_measurement/test_generate_controls.py \
  tests/off_prior_measurement/test_roundtrip_controls.py -v
```

Expected:

```text
FAILED
```

Failures should mention missing ladder columns or unexpected `reference_regime`.

- [ ] **Step 6: Add ladder fields to standard references**

In `scripts/off_prior_measurement/dreambooth_data.py`, add the ladder fields inside `build_reference_manifest`:

```python
                        "reference_regime": "standard_reference",
                        "hardness_axis": "none",
                        "source_standard_image": "",
                        "variant_id": "",
                        "transform_parameters": "{}",
```

- [ ] **Step 7: Add ladder fields to base controls**

In `scripts/off_prior_measurement/generate_controls.py`, change the regime loop to:

```python
        for regime_dir, source_group, reference_regime, hardness_axis in [
            ("easy", "base_easy_control", "easy_control", "none"),
            ("hard", "base_hard_control", "hard_control", "clutter_background"),
        ]:
```

Add these fields to every control row:

```python
                            "hardness_axis": hardness_axis,
                            "source_standard_image": "",
                            "variant_id": "",
                            "transform_parameters": "{}",
```

- [ ] **Step 8: Add hard manifest to combined manifest**

In `scripts/off_prior_measurement/dreambooth_data.py`, import the hard manifest builder inside `write_combined_manifest`:

```python
    from scripts.off_prior_measurement.hard_references import build_hard_reference_manifest
```

After writing `reference_manifest.csv`, add:

```python
    hard_reference_root = config.cache_dir / "hard_references"
    hard_reference = build_hard_reference_manifest(
        reference_manifest_path=reference_manifest_path,
        hard_root=hard_reference_root,
        variants=(config.hard_reference_variants or []),
    )
    if config.hard_reference_limit_per_subject is not None:
        hard_reference = (
            hard_reference.sort_values(["subject_id", "image_id", "variant_id", "conditioning_key"])
            .groupby(["subject_id", "variant_id", "conditioning_key"], as_index=False)
            .head(config.hard_reference_limit_per_subject)
            .reset_index(drop=True)
        )
    hard_reference_manifest_path = manifest_dir / "hard_reference_manifest.csv"
    hard_reference.to_csv(hard_reference_manifest_path, index=False)
```

Change the final concat to:

```python
    combined = pd.concat([reference, hard_reference, controls, roundtrip], ignore_index=True)
```

- [ ] **Step 9: Preserve metadata in roundtrip controls**

In `scripts/off_prior_measurement/roundtrip_controls.py`, keep existing columns and overwrite only:

```python
        new_row["source_group"] = "vae_roundtrip_control"
        new_row["reference_regime"] = "roundtrip_control"
        new_row["hardness_axis"] = "none"
        new_row["source_standard_image"] = row["image_path"]
        new_row["variant_id"] = ""
        new_row["transform_parameters"] = "{}"
```

- [ ] **Step 10: Run manifest tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/off_prior_measurement/test_dreambooth_data.py \
  tests/off_prior_measurement/test_generate_controls.py \
  tests/off_prior_measurement/test_roundtrip_controls.py -v
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 11: Commit**

Run:

```bash
git add scripts/off_prior_measurement/dreambooth_data.py scripts/off_prior_measurement/generate_controls.py scripts/off_prior_measurement/roundtrip_controls.py tests/off_prior_measurement/test_dreambooth_data.py tests/off_prior_measurement/test_generate_controls.py tests/off_prior_measurement/test_roundtrip_controls.py
git commit -m "feat: carry ladder metadata in manifests"
```

---

### Task 4: Add Ladder Summaries

**Files:**

- Modify: `scripts/off_prior_measurement/summarize.py`
- Modify: `tests/off_prior_measurement/test_summarize.py`

- [ ] **Step 1: Write ladder-summary test**

Append this test to `tests/off_prior_measurement/test_summarize.py`:

```python
def test_summarize_metrics_writes_ladder_summary(tmp_path):
    raw_path = tmp_path / "raw_metrics.csv"
    output_dir = tmp_path / "experiment"
    rows = []
    for regime, group, value in [
        ("easy_control", "base_easy_control", 0.20),
        ("standard_reference", "dreambooth_reference", 0.35),
        ("hard_reference", "dreambooth_hard_reference", 0.60),
        ("hard_control", "base_hard_control", 0.50),
        ("roundtrip_control", "vae_roundtrip_control", 0.30),
    ]:
        rows.append(
            {
                "subject_id": "dog",
                "source_group": group,
                "reference_regime": regime,
                "hardness_axis": "none",
                "conditioning_key": "class",
                "timestep": 50,
                "normalized_l2": value,
                "cosine_distance": 0.1,
                "dct_delta_low": value,
                "dct_delta_mid": value,
                "dct_delta_high": value,
            }
        )
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    paths = summarize_metrics(raw_path, output_dir)

    ladder = pd.read_csv(paths["ladder_summary"])
    row = ladder.iloc[0]
    assert round(row["standard_minus_easy"], 6) == 0.15
    assert round(row["hard_minus_standard"], 6) == 0.25
    assert bool(row["ladder_monotonic"]) is True
```

- [ ] **Step 2: Run the new test and confirm it fails**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_summarize.py::test_summarize_metrics_writes_ladder_summary -v
```

Expected:

```text
FAILED
KeyError: 'ladder_summary'
```

- [ ] **Step 3: Group summaries by hardness axis**

In `scripts/off_prior_measurement/summarize.py`, change `regime_summary` grouping to:

```python
        scored.groupby(
            ["source_group", "reference_regime", "hardness_axis", "conditioning_key", "timestep"],
            as_index=False,
        )
```

If `hardness_axis` is absent in an old CSV, normalize it before grouping:

```python
    if "hardness_axis" not in raw.columns:
        raw["hardness_axis"] = "none"
```

- [ ] **Step 4: Add ladder summary computation**

In `summarize_metrics`, after `subject_summary` is written, add:

```python
    ladder_base = (
        scored.groupby(["subject_id", "conditioning_key", "reference_regime"], as_index=False)
        .agg(mean_floor_adjusted_l2=("floor_adjusted_l2", "mean"))
    )
    ladder_wide = ladder_base.pivot_table(
        index=["subject_id", "conditioning_key"],
        columns="reference_regime",
        values="mean_floor_adjusted_l2",
        aggfunc="mean",
    ).reset_index()
    for column in ["easy_control", "standard_reference", "hard_reference", "hard_control", "roundtrip_control"]:
        if column not in ladder_wide.columns:
            ladder_wide[column] = float("nan")
    ladder_wide["standard_minus_easy"] = ladder_wide["standard_reference"] - ladder_wide["easy_control"]
    ladder_wide["hard_minus_standard"] = ladder_wide["hard_reference"] - ladder_wide["standard_reference"]
    ladder_wide["ladder_monotonic"] = (
        (ladder_wide["hard_reference"] > ladder_wide["standard_reference"])
        & (ladder_wide["standard_reference"] > ladder_wide["easy_control"])
    )
    ladder_summary_path = summaries_dir / "ladder_summary.csv"
    ladder_wide.to_csv(ladder_summary_path, index=False)
```

Add `"ladder_summary": ladder_summary_path` to the returned dictionary.

- [ ] **Step 5: Run summarize tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_summarize.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/off_prior_measurement/summarize.py tests/off_prior_measurement/test_summarize.py
git commit -m "feat: summarize prior compatibility ladder"
```

---

### Task 5: Add Ladder Figures

**Files:**

- Modify: `scripts/off_prior_measurement/plot.py`
- Modify: `tests/off_prior_measurement/test_plot.py`

- [ ] **Step 1: Extend plot test**

In `tests/off_prior_measurement/test_plot.py`, add a hard-reference row to the test DataFrame:

```python
            {
                "subject_id": "dog",
                "source_group": "dreambooth_hard_reference",
                "reference_regime": "hard_reference",
                "hardness_axis": "crop",
                "conditioning_key": "class",
                "timestep": 50,
                "floor_adjusted_l2": 0.6,
                "normalized_l2": 0.8,
                "dct_delta_low": 3.0,
                "dct_delta_mid": 2.0,
                "dct_delta_high": 1.0,
            },
```

Then assert:

```python
    assert paths["ladder_timestep_heatmap"].exists()
    assert paths["hardness_frequency_heatmap"].exists()
```

- [ ] **Step 2: Run plot test and confirm failure**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_plot.py -v
```

Expected:

```text
FAILED
KeyError: 'ladder_timestep_heatmap'
```

- [ ] **Step 3: Add reference-regime heatmap**

In `scripts/off_prior_measurement/plot.py`, after `timestep_curves`, add:

```python
    ladder_heatmap = figures_dir / "ladder_timestep_heatmap.png"
    ladder_data = scored.pivot_table(
        index="reference_regime",
        columns="timestep",
        values="floor_adjusted_l2",
        aggfunc="mean",
    ).fillna(0.0)
    plt.figure(figsize=(8, 4))
    plt.imshow(ladder_data.to_numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="mean floor_adjusted_l2")
    plt.xticks(np.arange(len(ladder_data.columns)), ladder_data.columns)
    plt.yticks(np.arange(len(ladder_data.index)), ladder_data.index)
    plt.xlabel("timestep")
    plt.ylabel("reference_regime")
    plt.tight_layout()
    plt.savefig(ladder_heatmap, dpi=200)
    plt.close()
```

- [ ] **Step 4: Add hardness-axis frequency heatmap**

In `scripts/off_prior_measurement/plot.py`, add:

```python
    hardness_frequency_heatmap = figures_dir / "hardness_frequency_heatmap.png"
    if "hardness_axis" not in scored.columns:
        scored["hardness_axis"] = "none"
    hardness_data = (
        scored.groupby("hardness_axis")[["dct_delta_low", "dct_delta_mid", "dct_delta_high"]]
        .mean()
        .rename(columns={"dct_delta_low": "low", "dct_delta_mid": "mid", "dct_delta_high": "high"})
    )
    plt.figure(figsize=(7, 4))
    plt.imshow(hardness_data.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="mean DCT residual energy")
    plt.xticks(np.arange(len(hardness_data.columns)), hardness_data.columns)
    plt.yticks(np.arange(len(hardness_data.index)), hardness_data.index)
    plt.xlabel("frequency band")
    plt.ylabel("hardness_axis")
    plt.tight_layout()
    plt.savefig(hardness_frequency_heatmap, dpi=200)
    plt.close()
```

Add both figures to the return dictionary:

```python
        "ladder_timestep_heatmap": ladder_heatmap,
        "hardness_frequency_heatmap": hardness_frequency_heatmap,
```

- [ ] **Step 5: Run plot tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_plot.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/off_prior_measurement/plot.py tests/off_prior_measurement/test_plot.py
git commit -m "feat: plot prior compatibility ladder"
```

---

### Task 6: Replace V0 Conclusion Logic With Ladder Go / No-Go

**Files:**

- Modify: `scripts/off_prior_measurement/write_conclusion.py`
- Modify: `tests/off_prior_measurement/test_write_conclusion.py`

- [ ] **Step 1: Write v2 conclusion test**

Replace the main test in `tests/off_prior_measurement/test_write_conclusion.py` with:

```python
import pandas as pd

from scripts.off_prior_measurement.write_conclusion import write_conclusion


def test_write_conclusion_uses_ladder_go_no_go(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    subjects = [f"subject_{idx}" for idx in range(8)]
    ladder_rows = []
    subject_rows = []
    for idx, subject in enumerate(subjects):
        standard = 0.2 if idx < 4 else -0.05
        hard = standard + 0.3 if idx < 6 else standard - 0.1
        ladder_rows.append(
            {
                "subject_id": subject,
                "conditioning_key": "class",
                "easy_control": 0.0,
                "standard_reference": standard,
                "hard_reference": hard,
                "hard_control": 0.25,
                "roundtrip_control": 0.1,
                "standard_minus_easy": standard,
                "hard_minus_standard": hard - standard,
                "ladder_monotonic": hard > standard > 0.0,
            }
        )
        subject_rows.extend(
            [
                {
                    "subject_id": subject,
                    "source_group": "dreambooth_hard_reference",
                    "conditioning_key": "class",
                    "mean_floor_adjusted_l2": hard,
                    "median_floor_adjusted_l2": hard,
                    "mean_cosine_distance": 0.2,
                    "n": 1,
                },
                {
                    "subject_id": subject,
                    "source_group": "base_hard_control",
                    "conditioning_key": "class",
                    "mean_floor_adjusted_l2": 0.25,
                    "median_floor_adjusted_l2": 0.25,
                    "mean_cosine_distance": 0.2,
                    "n": 1,
                },
            ]
        )
    pd.DataFrame(ladder_rows).to_csv(summaries / "ladder_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(summaries / "subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_hard_reference",
                "reference_regime": "hard_reference",
                "hardness_axis": "crop",
                "conditioning_key": "class",
                "timestep": 800,
                "mean_floor_adjusted_l2": 0.7,
            }
        ]
    ).to_csv(summaries / "regime_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_hard_reference",
                "reference_regime": "hard_reference",
                "hardness_axis": "crop",
                "dct_delta_low": 3.0,
                "dct_delta_mid": 2.0,
                "dct_delta_high": 1.0,
            }
        ]
    ).to_csv(summaries / "scored_metrics.csv", index=False)

    conclusion_path = write_conclusion(experiment_dir)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "Stage 1 V2 Prior-Compatibility Ladder Conclusion" in text
    assert "Hard-reference positive subjects: 6 of 8" in text
    assert "Hard greater than standard: 6 of 8" in text
    assert "Standard greater than easy: 4 of 8" in text
    assert "Go / no-go decision: Go" in text
```

- [ ] **Step 2: Run conclusion test and confirm failure**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_write_conclusion.py -v
```

Expected:

```text
FAILED
```

The failure should show the old v0 conclusion title or missing ladder counts.

- [ ] **Step 3: Implement v2 conclusion reader**

In `scripts/off_prior_measurement/write_conclusion.py`, replace `SMOKE_SUBJECTS` and fixed 5-subject logic with helpers:

```python
def _best_conditioning_ladder(ladder: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    candidates = []
    for key in ["class", "class_context"]:
        rows = ladder[ladder["conditioning_key"] == key]
        if rows.empty:
            continue
        hard_positive = int((rows["hard_reference"] > 0).sum())
        hard_gt_standard = int((rows["hard_reference"] > rows["standard_reference"]).sum())
        standard_gt_easy = int((rows["standard_reference"] > rows["easy_control"]).sum())
        candidates.append((hard_positive + hard_gt_standard + standard_gt_easy, key, rows))
    if not candidates:
        return "none", ladder.iloc[0:0]
    _, key, rows = max(candidates, key=lambda item: item[0])
    return key, rows
```

Use this Go / No-Go logic:

```python
    hard_positive = int((ladder_rows["hard_reference"] > 0).sum())
    hard_gt_standard = int((ladder_rows["hard_reference"] > ladder_rows["standard_reference"]).sum())
    standard_gt_easy = int((ladder_rows["standard_reference"] > ladder_rows["easy_control"]).sum())
    subject_count = int(ladder_rows["subject_id"].nunique())
    base_hard_positive = int(
        (
            subject_summary[
                (subject_summary["source_group"] == "base_hard_control")
                & (subject_summary["conditioning_key"] == best_conditioning)
            ]["mean_floor_adjusted_l2"]
            > 0
        ).sum()
    )
    roundtrip_ok = bool((ladder_rows["roundtrip_control"] <= ladder_rows["standard_reference"]).sum() >= subject_count // 2)
    go = (
        hard_positive >= 6
        and hard_gt_standard >= 6
        and standard_gt_easy >= 4
        and base_hard_positive >= max(1, subject_count // 2)
        and roundtrip_ok
    )
```

Write a Markdown conclusion containing the exact lines checked by the test.

- [ ] **Step 4: Run conclusion tests**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement/test_write_conclusion.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/off_prior_measurement/write_conclusion.py tests/off_prior_measurement/test_write_conclusion.py
git commit -m "feat: write ladder go no-go conclusion"
```

---

### Task 7: Add V2 Experiment README And Full Test Run

**Files:**

- Create: `experiments/off_prior_measurement_v0/ladder_v2/README.md`
- Modify: `README.md`

- [ ] **Step 1: Create v2 experiment README**

Create `experiments/off_prior_measurement_v0/ladder_v2/README.md` with:

````markdown
# Off-Prior Measurement V0 Ladder V2

Status: implementation prepared; full run not completed.

Purpose: test whether reference prior compatibility controls denoising-target off-priorness before personalization fine-tuning.

Reference-regime ladder:

```text
easy_control < standard_reference < hard_reference
```

Subjects:

```text
dog, cat, backpack, vase, colorful_sneaker, shiny_sneaker, fancy_boot, grey_sloth_plushie
```

Run order:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2

$PYTHON -m scripts.off_prior_measurement.generate_controls --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.dreambooth_data --config $CONFIG
$PYTHON -m scripts.off_prior_measurement.hard_references --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
$PYTHON -m scripts.off_prior_measurement.roundtrip_controls --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv
```

Single-GPU measurement:

```bash
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv
```

Four-GPU measurement:

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 0 --world-size 4 --output-name raw_metrics_rank0.csv
CUDA_VISIBLE_DEVICES=1 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 1 --world-size 4 --output-name raw_metrics_rank1.csv
CUDA_VISIBLE_DEVICES=2 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 2 --world-size 4 --output-name raw_metrics_rank2.csv
CUDA_VISIBLE_DEVICES=3 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 3 --world-size 4 --output-name raw_metrics_rank3.csv
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --world-size 4 --merge-shards
```

Analysis:

```bash
$PYTHON -m scripts.off_prior_measurement.summarize --raw-metrics $EXP/measurements/raw_metrics.csv --output-dir $EXP
$PYTHON -m scripts.off_prior_measurement.plot --scored-metrics $EXP/summaries/scored_metrics.csv --figures-dir $EXP/figures
$PYTHON -m scripts.off_prior_measurement.write_conclusion --experiment-dir $EXP
```

Go / No-Go:

- `hard_reference` positive for at least 6 of 8 subjects under class or class-context conditioning.
- `hard_reference > standard_reference` for at least 6 of 8 subjects.
- `standard_reference > easy_control` for at least 4 of 8 subjects.
- `base_hard_control` remains positive.
- `roundtrip_control` does not explain the hard-reference signal.
````

- [ ] **Step 2: Update top-level README**

In `README.md`, add this plan path to Current design inputs:

```text
docs/superpowers/plans/2026-06-22-stage-1-v2-prior-compatibility-ladder.md
```

Change the immediate next step to:

```text
Immediate next step: execute the Stage 1 v2 prior-compatibility ladder implementation plan, then run the 8-subject ladder experiment.
```

- [ ] **Step 3: Run the full lightweight test suite**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 4: Commit**

Run:

```bash
git add experiments/off_prior_measurement_v0/ladder_v2/README.md README.md
git commit -m "docs: document ladder v2 experiment"
```

---

### Task 8: Run V2 Data Preparation Smoke

**Files:**

- Generated: `experiments/off_prior_measurement_v0/ladder_v2/manifests/*.csv`
- Generated: `data/cache/off_prior_measurement_v0/hard_references/`
- Modify: `experiments/off_prior_measurement_v0/ladder_v2/README.md`

- [ ] **Step 1: Confirm DreamBooth subject directories exist upstream**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
import json
import urllib.request

url = "https://api.github.com/repos/google/dreambooth/contents/dataset?ref=main"
request = urllib.request.Request(url, headers={"User-Agent": "dadt-ladder-v2"})
with urllib.request.urlopen(request, timeout=30) as response:
    names = {item["name"] for item in json.loads(response.read().decode("utf-8")) if item.get("type") == "dir"}
wanted = {"dog", "cat", "backpack", "vase", "colorful_sneaker", "shiny_sneaker", "fancy_boot", "grey_sloth_plushie"}
missing = sorted(wanted - names)
print("missing", missing)
PY
```

Expected:

```text
missing []
```

- [ ] **Step 2: Generate base controls**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
$PYTHON -m scripts.off_prior_measurement.generate_controls --config $CONFIG 2>&1 | tee experiments/off_prior_measurement_v0/ladder_v2/generate_controls.log
```

Expected:

```text
data/cache/off_prior_measurement_v0/generated_controls
```

- [ ] **Step 3: Build manifests**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
$PYTHON -m scripts.off_prior_measurement.dreambooth_data --config $CONFIG 2>&1 | tee experiments/off_prior_measurement_v0/ladder_v2/build_manifest.log
```

Expected:

```text
experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv
```

- [ ] **Step 4: Generate hard references**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2
$PYTHON -m scripts.off_prior_measurement.hard_references --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv 2>&1 | tee $EXP/hard_references.log
```

Expected:

```text
data/cache/off_prior_measurement_v0/hard_references
```

- [ ] **Step 5: Generate VAE roundtrip controls**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2
$PYTHON -m scripts.off_prior_measurement.roundtrip_controls --config $CONFIG --reference-manifest $EXP/manifests/reference_manifest.csv 2>&1 | tee $EXP/roundtrip_controls.log
```

Expected:

```text
data/cache/off_prior_measurement_v0/vae_roundtrip_controls
```

- [ ] **Step 6: Validate combined manifest counts**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
import pandas as pd
path = "experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv"
manifest = pd.read_csv(path, keep_default_na=False)
print(manifest.groupby(["source_group", "reference_regime"]).size())
print("subjects", manifest["subject_id"].nunique())
print("missing_images", sum(not __import__("pathlib").Path(p).exists() for p in manifest["image_path"].drop_duplicates()))
PY
```

Expected:

```text
subjects 8
missing_images 0
```

The exact row counts depend on how many DreamBooth images each subject has, but all five `source_group` values must appear.

- [ ] **Step 7: Update v2 README status**

In `experiments/off_prior_measurement_v0/ladder_v2/README.md`, change:

```text
Status: implementation prepared; full run not completed.
```

to:

```text
Status: data preparation completed; measurement not completed.
```

- [ ] **Step 8: Commit**

Run:

```bash
git add experiments/off_prior_measurement_v0/ladder_v2/README.md experiments/off_prior_measurement_v0/ladder_v2/manifests/reference_manifest.csv experiments/off_prior_measurement_v0/ladder_v2/manifests/hard_reference_manifest.csv experiments/off_prior_measurement_v0/ladder_v2/manifests/combined_manifest.csv
git commit -m "data: prepare ladder v2 manifests"
```

---

### Task 9: Run Multi-GPU Measurement And Analysis

**Files:**

- Generated: `experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics.csv`
- Generated: `experiments/off_prior_measurement_v0/ladder_v2/summaries/*.csv`
- Generated: `experiments/off_prior_measurement_v0/ladder_v2/figures/*.png`
- Generated: `experiments/off_prior_measurement_v0/ladder_v2/conclusion.md`
- Modify: `experiments/off_prior_measurement_v0/ladder_v2/README.md`
- Modify: `README.md`

- [ ] **Step 1: Check visible GPUs**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

Expected:

```text
At least four GPUs with enough free memory for SD 1.5 inference.
```

- [ ] **Step 2: Launch four measurement shards**

Run the four commands in separate shell sessions:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2

CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 0 --world-size 4 --output-name raw_metrics_rank0.csv 2>&1 | tee $EXP/measure_rank0.log
CUDA_VISIBLE_DEVICES=1 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 1 --world-size 4 --output-name raw_metrics_rank1.csv 2>&1 | tee $EXP/measure_rank1.log
CUDA_VISIBLE_DEVICES=2 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 2 --world-size 4 --output-name raw_metrics_rank2.csv 2>&1 | tee $EXP/measure_rank2.log
CUDA_VISIBLE_DEVICES=3 $PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --rank 3 --world-size 4 --output-name raw_metrics_rank3.csv 2>&1 | tee $EXP/measure_rank3.log
```

Expected:

```text
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics_rank0.csv
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics_rank1.csv
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics_rank2.csv
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics_rank3.csv
```

- [ ] **Step 3: Merge shards**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export CONFIG=configs/off_prior_measurement_v0/ladder_v2.yaml
export EXP=experiments/off_prior_measurement_v0/ladder_v2
$PYTHON -m scripts.off_prior_measurement.measure --config $CONFIG --manifest $EXP/manifests/combined_manifest.csv --world-size 4 --merge-shards 2>&1 | tee $EXP/merge_measurements.log
```

Expected:

```text
experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics.csv
```

- [ ] **Step 4: Summarize, plot, and write conclusion**

Run:

```bash
export PYTHON=/home/deepseek_VG/.conda/envs/dyme/bin/python
export EXP=experiments/off_prior_measurement_v0/ladder_v2
$PYTHON -m scripts.off_prior_measurement.summarize --raw-metrics $EXP/measurements/raw_metrics.csv --output-dir $EXP 2>&1 | tee $EXP/summarize.log
$PYTHON -m scripts.off_prior_measurement.plot --scored-metrics $EXP/summaries/scored_metrics.csv --figures-dir $EXP/figures 2>&1 | tee $EXP/plot.log
$PYTHON -m scripts.off_prior_measurement.write_conclusion --experiment-dir $EXP 2>&1 | tee $EXP/write_conclusion.log
```

Expected:

```text
experiments/off_prior_measurement_v0/ladder_v2/summaries/ladder_summary.csv
experiments/off_prior_measurement_v0/ladder_v2/figures/ladder_timestep_heatmap.png
experiments/off_prior_measurement_v0/ladder_v2/figures/hardness_frequency_heatmap.png
experiments/off_prior_measurement_v0/ladder_v2/conclusion.md
```

- [ ] **Step 5: Inspect ladder result**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python - <<'PY'
import pandas as pd
ladder = pd.read_csv("experiments/off_prior_measurement_v0/ladder_v2/summaries/ladder_summary.csv")
for key in ["class", "class_context", "null"]:
    rows = ladder[ladder["conditioning_key"] == key]
    if rows.empty:
        continue
    print(key)
    print("hard_positive", int((rows["hard_reference"] > 0).sum()), "of", rows["subject_id"].nunique())
    print("hard_gt_standard", int((rows["hard_reference"] > rows["standard_reference"]).sum()), "of", rows["subject_id"].nunique())
    print("standard_gt_easy", int((rows["standard_reference"] > rows["easy_control"]).sum()), "of", rows["subject_id"].nunique())
PY
```

Expected:

```text
class
class_context
null
```

At least one of `class` or `class_context` must satisfy the v2 Go / No-Go thresholds to move toward Stage 2.

- [ ] **Step 6: Update documentation with result**

In `experiments/off_prior_measurement_v0/ladder_v2/README.md`, update `Status` to one of:

```text
Status: complete v2 ladder run on 2026-06-22. Result is Go under the 8-subject ladder rule.
```

or:

```text
Status: complete v2 ladder run on 2026-06-22. Result is No-Go under the 8-subject ladder rule.
```

In top-level `README.md`, update `Current Status` and `Immediate next step` according to the conclusion:

```text
If Go: plan Stage 2 correlation-with-forgetting and DADT target correction.
If No-Go: revise the off-priorness metric before any personalization fine-tuning.
```

- [ ] **Step 7: Run tests after analysis**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/off_prior_measurement -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 8: Commit curated v2 results**

Run:

```bash
git add README.md experiments/off_prior_measurement_v0/ladder_v2/README.md experiments/off_prior_measurement_v0/ladder_v2/conclusion.md experiments/off_prior_measurement_v0/ladder_v2/config_resolved.yaml experiments/off_prior_measurement_v0/ladder_v2/measurements/raw_metrics.csv experiments/off_prior_measurement_v0/ladder_v2/summaries/*.csv experiments/off_prior_measurement_v0/ladder_v2/figures/*.png
git commit -m "exp: run ladder v2 measurement"
```

---

### Task 10: GitHub API Backup

**Files:**

- No source changes required unless `notes/2026-06-21-github-backup-setup.md` becomes inaccurate.

- [ ] **Step 1: Confirm no secrets are tracked**

Run:

```bash
git check-ignore -v token.txt
git ls-files | grep -E 'token.txt|\\.env$|\\.key$|\\.pem$' || true
```

Expected:

```text
.gitignore:2:token.txt	token.txt
```

The second command must print no tracked secret path.

- [ ] **Step 2: Back up to GitHub**

Use the existing GitHub Git Data API backup method from `notes/2026-06-21-github-backup-setup.md`. The backup script must print:

```text
contains_token False
local_commit <local sha>
remote_commit <remote sha>
```

- [ ] **Step 3: Record backup in the final response**

Report the local commit, remote backup commit, and `contains_token False` to the user. Do not print token text.

## Self-Review Checklist

- Spec coverage: Tasks 1-6 implement the v2 ladder data contract, hard references, summaries, figures, and Go / No-Go rule; Tasks 7-10 cover experiment docs, data preparation, multi-GPU measurement, results, and backup.
- No hidden cherry-picking: hard references are deterministic and every variant is recorded with `variant_id`, `hardness_axis`, `source_standard_image`, and `transform_parameters`.
- Reproducibility: every generated artifact has a command, expected path, and commit step.
- Project hygiene: README and experiment README are updated before and after the run.
