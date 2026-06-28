import pytest
import yaml
from PIL import Image

from scripts.off_prior_measurement.prompt_matched_controls import (
    build_prompt_matched_control_manifest,
    prompt_matched_generation_jobs,
)


def test_build_prompt_matched_control_manifest_preserves_prompt_fields(tmp_path):
    real_image = tmp_path / "real.jpg"
    real_image.write_bytes(b"fake")
    ordinary_manifest = tmp_path / "ordinary.yaml"
    ordinary_manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "dog_matched_00",
                        "image_path": str(real_image),
                        "reference_regime": "ordinary_matched_closeup",
                        "hardness_axis": "matched_closeup",
                        "conditioning_key": "prompt_matched",
                        "conditioning_prompt": "a close-up photo of a dog near a bowl",
                        "source_dataset": "coco2017_val",
                        "source_license_note": "test image",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generated_root = tmp_path / "generated"
    generated_dir = generated_root / "dog" / "dog_matched_00"
    generated_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(generated_dir / "seed_0000.png")
    Image.new("RGB", (8, 8), "black").save(generated_dir / "seed_0001.png")

    manifest = build_prompt_matched_control_manifest(
        ordinary_manifest_path=ordinary_manifest,
        generated_root=generated_root,
        seeds_per_prompt=2,
    )

    assert len(manifest) == 2
    first = manifest.iloc[0]
    assert first["subject_id"] == "dog"
    assert first["class_name"] == "dog"
    assert first["image_id"] == "dog_matched_00_seed_0000"
    assert first["source_group"] == "base_generated_control"
    assert first["reference_regime"] == "prompt_matched_generated"
    assert first["hardness_axis"] == "matched_closeup"
    assert first["conditioning_key"] == "prompt_matched"
    assert first["conditioning_prompt"] == "a close-up photo of a dog near a bowl"
    assert first["source_dataset"] == "base_sd15_prompt_matched"


def test_build_prompt_matched_control_manifest_rejects_missing_generated_images(tmp_path):
    real_image = tmp_path / "real.jpg"
    real_image.write_bytes(b"fake")
    ordinary_manifest = tmp_path / "ordinary.yaml"
    ordinary_manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "dog",
                        "image_id": "dog_matched_00",
                        "image_path": str(real_image),
                        "conditioning_key": "prompt_matched",
                        "conditioning_prompt": "a close-up photo of a dog",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        build_prompt_matched_control_manifest(
            ordinary_manifest_path=ordinary_manifest,
            generated_root=tmp_path / "generated",
            seeds_per_prompt=1,
        )


def test_prompt_matched_generation_jobs_use_stable_output_paths(tmp_path):
    real_image = tmp_path / "real.jpg"
    real_image.write_bytes(b"fake")
    ordinary_manifest = tmp_path / "ordinary.yaml"
    ordinary_manifest.write_text(
        yaml.safe_dump(
            {
                "ordinary_real_controls": [
                    {
                        "class_name": "cat",
                        "image_id": "cat_window_00",
                        "image_path": str(real_image),
                        "conditioning_key": "prompt_matched",
                        "conditioning_prompt": "a photo of a cat sitting near a window",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = list(
        prompt_matched_generation_jobs(
            ordinary_manifest_path=ordinary_manifest,
            generated_root=tmp_path / "generated",
            seeds_per_prompt=2,
        )
    )

    assert [job.seed for job in jobs] == [0, 1]
    assert [job.prompt for job in jobs] == [
        "a photo of a cat sitting near a window",
        "a photo of a cat sitting near a window",
    ]
    assert jobs[0].output_path == tmp_path / "generated" / "cat" / "cat_window_00" / "seed_0000.png"
    assert jobs[1].output_path == tmp_path / "generated" / "cat" / "cat_window_00" / "seed_0001.png"
