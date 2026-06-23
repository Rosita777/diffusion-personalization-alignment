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
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )


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
    colors = [
        (255, 60, 60, 76),
        (40, 160, 255, 76),
        (255, 220, 40, 76),
        (40, 220, 140, 76),
    ]
    boxes = [
        (0.02, 0.04, 0.26, 0.18),
        (0.70, 0.08, 0.96, 0.24),
        (0.05, 0.72, 0.30, 0.92),
        (0.72, 0.68, 0.96, 0.94),
    ]
    for color, box in zip(colors, boxes):
        left = int(box[0] * width)
        top = int(box[1] * height)
        right = int(box[2] * width)
        bottom = int(box[3] * height)
        draw.rectangle((left, top, right, bottom), fill=color)
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
            new_row["image_path"] = str(
                _hard_image_path(hard_root, row["subject_id"], row["image_id"], variant_id)
            )
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
