import pandas as pd
import pytest
import yaml

from scripts.off_prior_measurement.source_decomp_manifest import (
    build_source_decomp_manifest,
    load_ordinary_real_controls,
)


def test_load_ordinary_real_controls_preserves_string_ids(tmp_path):
    manifest = tmp_path / "ordinary.yaml"
    image = tmp_path / "dog_real_00.jpg"
    image.write_bytes(b"fake")
    manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "00",
                        "image_path": str(image),
                        "source_dataset": "local",
                        "source_license_note": "local placeholder",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    controls = load_ordinary_real_controls(manifest)

    assert controls.iloc[0]["image_id"] == "00"
    assert controls.iloc[0]["source_group"] == "ordinary_real_control"


def test_load_ordinary_real_controls_preserves_diagnosis_regime_fields(tmp_path):
    manifest = tmp_path / "ordinary.yaml"
    image = tmp_path / "dog_real_00.jpg"
    image.write_bytes(b"fake")
    manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "dog_matched_00",
                        "image_path": str(image),
                        "reference_regime": "ordinary_matched_context",
                        "hardness_axis": "matched_context",
                        "conditioning_prompt": "a photo of a dog outdoors",
                        "source_dataset": "coco2017_val",
                        "source_license_note": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    controls = load_ordinary_real_controls(manifest)

    assert controls.iloc[0]["reference_regime"] == "ordinary_matched_context"
    assert controls.iloc[0]["hardness_axis"] == "matched_context"
    assert controls.iloc[0]["conditioning_prompt"] == "a photo of a dog outdoors"


def test_build_source_decomp_manifest_rejects_missing_ordinary_real_controls(tmp_path):
    reference_path = tmp_path / "reference.csv"
    controls_path = tmp_path / "controls.csv"
    ordinary_path = tmp_path / "ordinary.yaml"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "db_00",
                "image_path": str(tmp_path / "db_00.jpg"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "base_00",
                "image_path": str(tmp_path / "base_00.png"),
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(controls_path, index=False)
    ordinary_path.write_text("ordinary_real_controls: []", encoding="utf-8")

    with pytest.raises(ValueError, match="ordinary real controls"):
        build_source_decomp_manifest(reference_path, controls_path, ordinary_path, tmp_path / "roundtrip")


def test_build_source_decomp_manifest_rejects_missing_ordinary_real_image_paths(tmp_path):
    reference_path = tmp_path / "reference.csv"
    controls_path = tmp_path / "controls.csv"
    ordinary_path = tmp_path / "ordinary.yaml"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "db_00",
                "image_path": str(tmp_path / "db_00.jpg"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "base_00",
                "image_path": str(tmp_path / "base_00.png"),
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(controls_path, index=False)
    ordinary_path.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "dog_real_00",
                        "image_path": str(tmp_path / "missing_dog.jpg"),
                        "source_dataset": "local",
                        "source_license_note": "local placeholder",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing_dog.jpg"):
        build_source_decomp_manifest(reference_path, controls_path, ordinary_path, tmp_path / "roundtrip")


def test_build_source_decomp_manifest_keeps_only_classes_with_ordinary_controls(tmp_path):
    reference_path = tmp_path / "reference.csv"
    controls_path = tmp_path / "controls.csv"
    ordinary_path = tmp_path / "ordinary.yaml"
    dog_real = tmp_path / "dog_real_00.jpg"
    dog_real.write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "db_dog_00",
                "image_path": str(tmp_path / "db_dog_00.jpg"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            },
            {
                "subject_id": "cat",
                "class_name": "cat",
                "image_id": "db_cat_00",
                "image_path": str(tmp_path / "db_cat_00.jpg"),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a cat",
            },
        ]
    ).to_csv(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "base_dog_00",
                "image_path": str(tmp_path / "base_dog_00.png"),
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            },
            {
                "subject_id": "cat",
                "class_name": "cat",
                "image_id": "base_cat_00",
                "image_path": str(tmp_path / "base_cat_00.png"),
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a cat",
            },
        ]
    ).to_csv(controls_path, index=False)
    ordinary_path.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "dog_real_00",
                        "image_path": str(dog_real),
                        "source_dataset": "local",
                        "source_license_note": "local placeholder",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_source_decomp_manifest(reference_path, controls_path, ordinary_path, tmp_path / "roundtrip")

    assert sorted(manifest["class_name"].unique()) == ["dog"]
    assert sorted(manifest["source_group"].unique()) == [
        "base_generated_control",
        "dreambooth_reference",
        "ordinary_real_control",
    ]
