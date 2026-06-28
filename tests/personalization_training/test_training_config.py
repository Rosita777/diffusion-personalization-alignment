from pathlib import Path

import pytest
import yaml

from scripts.personalization_training.config import load_training_config


def _base_config(tmp_path: Path, image_path: Path) -> dict:
    return {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "output_dir": str(tmp_path / "out"),
        "resolution": 512,
        "subjects": [
            {
                "subject_id": "vase",
                "class_name": "vase",
                "instance_prompt": "a photo of sks vase",
                "class_prompt": "a photo of a vase",
                "image_paths": [str(image_path)],
            }
        ],
        "training": {
            "condition": "dadt_lf_late",
            "max_train_steps": 1,
            "learning_rate": 1e-4,
            "lora_rank": 4,
            "seed": 0,
            "train_batch_size": 1,
        },
        "alignment": {
            "alpha": 0.5,
            "late_timestep_threshold": 800,
            "low_radius": 2,
            "mid_radius": 4,
        },
        "evaluation": {
            "num_images_per_prompt": 1,
            "prompts": ["a photo of sks vase"],
            "class_prompts": ["a photo of a vase"],
        },
    }


def test_load_training_config_rejects_invalid_alpha(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"fake")
    path = tmp_path / "config.yaml"
    config = _base_config(tmp_path, image)
    config["alignment"]["alpha"] = 1.5
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="alpha"):
        load_training_config(path)


def test_load_training_config_rejects_missing_subject_image(tmp_path):
    path = tmp_path / "config.yaml"
    config = _base_config(tmp_path, tmp_path / "missing.png")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_training_config(path)


def test_load_training_config_reads_valid_config(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"fake")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_base_config(tmp_path, image)), encoding="utf-8")

    config = load_training_config(path)

    assert config.model_id == "runwayml/stable-diffusion-v1-5"
    assert config.subjects[0].subject_id == "vase"
    assert config.subjects[0].image_paths == [image]
    assert config.training.condition == "dadt_lf_late"
    assert config.alignment.alpha == 0.5
