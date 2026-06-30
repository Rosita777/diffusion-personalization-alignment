from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.personalization_training.config import load_training_config
from scripts.personalization_training.train_lora_dreambooth import (
    build_training_examples,
    collate_reference_batch,
    config_with_condition,
    config_with_max_train_steps,
    filter_config_subjects,
    target_for_condition,
    training_output_dir,
)


def _write_config(tmp_path: Path) -> Path:
    image = tmp_path / "source.png"
    image.write_bytes(b"fake")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "output_dir": str(tmp_path / "out"),
                "resolution": 512,
                "subjects": [
                    {
                        "subject_id": "vase",
                        "class_name": "vase",
                        "instance_prompt": "a photo of sks vase",
                        "class_prompt": "a photo of a vase",
                        "image_paths": [str(image)],
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
        ),
        encoding="utf-8",
    )
    return path


def test_train_entrypoint_help_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.personalization_training.train_lora_dreambooth", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--dry-run" in result.stdout


def test_train_entrypoint_dry_run_prints_planned_subject(tmp_path):
    config_path = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.personalization_training.train_lora_dreambooth",
            "--config",
            str(config_path),
            "--condition",
            "vanilla",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "condition: vanilla" in result.stdout
    assert "subject: vase" in result.stdout


def test_train_entrypoint_dry_run_subject_id_uses_subject_output_dir(tmp_path):
    config_path = _write_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.personalization_training.train_lora_dreambooth",
            "--config",
            str(config_path),
            "--subject-id",
            "vase",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"output_dir: {tmp_path / 'out' / 'dadt_lf_late' / 'vase'}" in result.stdout


def test_eval_and_report_entrypoints_expose_help():
    for module in [
        "scripts.personalization_training.generate_eval_grid",
        "scripts.personalization_training.write_stage2a_report",
    ]:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout


def test_evaluation_prompts_for_subject_filters_instance_and_class_prompts(tmp_path):
    from scripts.personalization_training.generate_eval_grid import evaluation_prompts_for_subject

    config_path = _write_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dog_image = tmp_path / "dog.png"
    dog_image.write_bytes(b"fake")
    raw["subjects"].append(
        {
            "subject_id": "dog",
            "class_name": "dog",
            "instance_prompt": "a photo of sks dog",
            "class_prompt": "a photo of a dog",
            "image_paths": [str(dog_image)],
        }
    )
    raw["evaluation"]["prompts"] = [
        "a photo of sks vase on a wooden table",
        "a photo of sks dog in a park",
    ]
    raw["evaluation"]["class_prompts"] = [
        "a photo of a vase",
        "a photo of a dog",
    ]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_training_config(config_path)

    prompts = evaluation_prompts_for_subject(config, subject_id="vase")

    assert [prompt.kind for prompt in prompts] == ["subject", "class"]
    assert [prompt.text for prompt in prompts] == [
        "a photo of sks vase on a wooden table",
        "a photo of a vase",
    ]


def test_safe_prompt_slug_is_stable_and_filesystem_friendly():
    from scripts.personalization_training.generate_eval_grid import safe_prompt_slug

    slug = safe_prompt_slug("A photo of SKS vase, on a wooden table!!!", max_length=32)

    assert slug == "a-photo-of-sks-vase-on-a-wooden"


def test_load_eval_lora_weights_uses_unet_adapter_without_prefix(tmp_path):
    from scripts.personalization_training.generate_eval_grid import load_eval_lora_weights

    class FakeUnet:
        def __init__(self) -> None:
            self.calls = []

        def load_lora_adapter(self, weights_dir, **kwargs):
            self.calls.append((weights_dir, kwargs))

    class FakePipe:
        def __init__(self) -> None:
            self.unet = FakeUnet()

    pipe = FakePipe()
    weights_dir = tmp_path / "weights"

    load_eval_lora_weights(pipe, weights_dir)

    assert pipe.unet.calls == [
        (
            weights_dir,
            {
                "adapter_name": "default",
                "prefix": None,
                "weight_name": "pytorch_lora_weights.safetensors",
            },
        )
    ]


def test_load_eval_lora_weights_disables_awq_dispatch_before_loading(tmp_path, monkeypatch):
    import scripts.personalization_training.generate_eval_grid as eval_grid

    calls = []

    class FakeUnet:
        def load_lora_adapter(self, *_args, **_kwargs):
            calls.append("load")

    class FakePipe:
        def __init__(self) -> None:
            self.unet = FakeUnet()

    def fake_disable_awq():
        calls.append("disable_awq")

    monkeypatch.setattr(eval_grid, "_disable_awq_lora_dispatch", fake_disable_awq)

    eval_grid.load_eval_lora_weights(FakePipe(), tmp_path / "weights")

    assert calls == ["disable_awq", "load"]


def test_build_training_examples_pairs_subject_images_with_prompts(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    examples = build_training_examples(config)

    assert len(examples) == 1
    assert examples[0].subject_id == "vase"
    assert examples[0].prompt == "a photo of sks vase"
    assert examples[0].class_prompt == "a photo of a vase"


def test_config_with_condition_overrides_without_mutating_original(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    overridden = config_with_condition(config, "vanilla")

    assert config.training.condition == "dadt_lf_late"
    assert overridden.training.condition == "vanilla"


def test_config_with_max_train_steps_overrides_without_mutating_original(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    overridden = config_with_max_train_steps(config, 7)

    assert config.training.max_train_steps == 1
    assert overridden.training.max_train_steps == 7


def test_training_output_dir_is_condition_specific(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    assert training_output_dir(config_with_condition(config, "vanilla")) == tmp_path / "out" / "vanilla"
    assert training_output_dir(config) == tmp_path / "out" / "dadt_lf_late"
    assert training_output_dir(config, subject_id="vase") == tmp_path / "out" / "dadt_lf_late" / "vase"


def test_training_output_dir_uses_run_name_when_provided(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)

    assert training_output_dir(config, subject_id="vase", run_name="alpha075") == (
        tmp_path / "out" / "alpha075" / "vase"
    )


def test_filter_config_subjects_keeps_requested_subject_without_mutating_original(tmp_path):
    config_path = _write_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dog_image = tmp_path / "dog.png"
    dog_image.write_bytes(b"fake")
    raw["subjects"].append(
        {
            "subject_id": "dog",
            "class_name": "dog",
            "instance_prompt": "a photo of sks dog",
            "class_prompt": "a photo of a dog",
            "image_paths": [str(dog_image)],
        }
    )
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_training_config(config_path)

    filtered = filter_config_subjects(config, subject_id="dog")

    assert [subject.subject_id for subject in config.subjects] == ["vase", "dog"]
    assert [subject.subject_id for subject in filtered.subjects] == ["dog"]


def test_target_for_condition_applies_alignment_only_for_dadt(tmp_path):
    import torch

    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)

    vanilla_target = target_for_condition(ref, base, timestep=900, config=config_with_condition(config, "vanilla"))
    dadt_target = target_for_condition(ref, base, timestep=900, config=config)

    assert torch.allclose(vanilla_target, ref)
    assert not torch.allclose(dadt_target, ref)


def test_target_for_condition_applies_cfg_residual_gate(tmp_path):
    import torch

    config_path = _write_config(tmp_path)
    config = config_with_condition(load_training_config(config_path), "dadt_cfg_residual_gate")
    ref = torch.tensor([[[[10.0, -10.0]]]])
    class_prediction = torch.zeros_like(ref)
    null_prediction = torch.full_like(ref, -1.0)

    target = target_for_condition(
        ref,
        class_prediction,
        timestep=900,
        config=config,
        null_prediction=null_prediction,
    )

    assert torch.allclose(target, torch.tensor([[[[5.0, -10.0]]]]))


def test_target_for_condition_requires_null_prediction_for_cfg_gate(tmp_path):
    import torch

    config_path = _write_config(tmp_path)
    config = config_with_condition(load_training_config(config_path), "dadt_cfg_residual_gate")
    ref = torch.randn(1, 2, 4, 4)
    class_prediction = torch.randn(1, 2, 4, 4)

    with pytest.raises(ValueError, match="null_prediction"):
        target_for_condition(ref, class_prediction, timestep=900, config=config)


def test_reference_image_dataset_returns_normalized_pixels_and_prompts(tmp_path):
    import torch
    from PIL import Image

    from scripts.personalization_training.train_lora_dreambooth import ReferenceImageDataset

    image = tmp_path / "ref.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image)
    config_path = _write_config(tmp_path)
    config = load_training_config(config_path)
    config.subjects[0].image_paths[0].unlink()
    image.rename(config.subjects[0].image_paths[0])

    dataset = ReferenceImageDataset(build_training_examples(config), resolution=8)
    item = dataset[0]

    assert item["pixel_values"].shape == (3, 8, 8)
    assert item["pixel_values"].dtype == torch.float32
    assert item["pixel_values"].min().item() >= -1.0
    assert item["pixel_values"].max().item() <= 1.0
    assert item["prompt"] == "a photo of sks vase"
    assert item["class_prompt"] == "a photo of a vase"


def test_collate_reference_batch_stacks_pixels_and_keeps_prompt_lists():
    import torch

    batch = [
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "prompt": "a photo of sks vase",
            "class_prompt": "a photo of a vase",
            "subject_id": "vase",
            "image_path": "vase.png",
        },
        {
            "pixel_values": torch.ones(3, 8, 8),
            "prompt": "a photo of sks dog",
            "class_prompt": "a photo of a dog",
            "subject_id": "dog",
            "image_path": "dog.png",
        },
    ]

    collated = collate_reference_batch(batch)

    assert collated["pixel_values"].shape == (2, 3, 8, 8)
    assert collated["prompts"] == ["a photo of sks vase", "a photo of sks dog"]
    assert collated["class_prompts"] == ["a photo of a vase", "a photo of a dog"]


def test_cast_trainable_parameters_to_float32_preserves_frozen_dtype():
    import torch

    from scripts.personalization_training.train_lora_dreambooth import _cast_trainable_parameters_to_float32

    frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=True)
    module = torch.nn.ParameterList([frozen, trainable])

    _cast_trainable_parameters_to_float32(module)

    assert frozen.dtype == torch.float16
    assert trainable.dtype == torch.float32


def test_ensure_finite_loss_rejects_nan():
    import pytest
    import torch

    from scripts.personalization_training.train_lora_dreambooth import _ensure_finite_loss

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        _ensure_finite_loss(torch.tensor(float("nan")), step=3)
