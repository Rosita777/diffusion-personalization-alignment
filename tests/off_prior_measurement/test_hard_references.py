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
    assert set(manifest["hardness_axis"]) == {
        VARIANT_TO_AXIS["crop_large_subject"],
        VARIANT_TO_AXIS["edge_reflection_texture"],
    }
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
