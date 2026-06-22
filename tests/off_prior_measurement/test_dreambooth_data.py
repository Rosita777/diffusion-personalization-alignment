import pandas as pd

from scripts.off_prior_measurement.config import SubjectSpec
from scripts.off_prior_measurement.dreambooth_data import (
    build_reference_manifest,
    conditioning_prompt,
    write_combined_manifest,
)


def test_conditioning_prompt_variants():
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )

    assert conditioning_prompt(subject, "null") == ""
    assert conditioning_prompt(subject, "class") == "a photo of a dog"
    assert conditioning_prompt(subject, "class_context") == "a photo of a dog in a natural scene"


def test_build_reference_manifest_from_local_images(tmp_path):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    image_dir = tmp_path / "dataset" / "dog"
    image_dir.mkdir(parents=True)
    (image_dir / "00.png").write_bytes(b"fake")
    (image_dir / "01.jpg").write_bytes(b"fake")

    manifest = build_reference_manifest(
        dataset_root=tmp_path / "dataset",
        subjects=[subject],
        conditionings=["null", "class", "class_context"],
    )

    assert list(manifest.columns) == [
        "subject_id",
        "image_id",
        "image_path",
        "source_group",
        "reference_regime",
        "class_name",
        "class_prompt",
        "class_context_prompt",
        "conditioning_key",
        "conditioning_prompt",
    ]
    assert len(manifest) == 6
    assert set(manifest["source_group"]) == {"dreambooth_reference"}
    assert set(manifest["reference_regime"]) == {"standard"}
    assert set(manifest["conditioning_key"]) == {"null", "class", "class_context"}
    assert isinstance(manifest, pd.DataFrame)


def test_write_combined_manifest_includes_references_controls_and_roundtrip(tmp_path, monkeypatch):
    subject_path = tmp_path / "subjects.yaml"
    subject_path.write_text(
        """
subjects:
  - subject_id: dog
    hf_subset: dog
    class_name: dog
    class_prompt: a photo of a dog
    class_context_prompt: a photo of a dog in a natural scene
    hard_control_prompt: a photo of a dog under dramatic stage lighting
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_name: smoke_test
model_id: runwayml/stable-diffusion-v1-5
prediction_type: epsilon
device: cpu
dtype: float32
resolution: 512
dataset_repo: google/dreambooth
subject_manifest: {subject_path}
cache_dir: {tmp_path / "cache"}
output_dir: {tmp_path / "experiment"}
debug_output_dir: {tmp_path / "outputs"}
timesteps: [50]
noise_seeds: [0]
conditionings: ["class"]
control_images_per_subject: 1
batch_size: 1
save_debug_tensors: false
""".strip(),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "dog").mkdir(parents=True)
    (dataset_root / "dog" / "00.png").write_bytes(b"reference")
    generated_root = tmp_path / "cache" / "generated_controls" / "dog"
    (generated_root / "easy").mkdir(parents=True)
    (generated_root / "hard").mkdir(parents=True)
    (generated_root / "easy" / "seed_0000.png").write_bytes(b"easy")
    (generated_root / "hard" / "seed_0000.png").write_bytes(b"hard")

    monkeypatch.setattr(
        "scripts.off_prior_measurement.dreambooth_data.download_dreambooth_subjects",
        lambda config: dataset_root,
    )

    manifest_path = write_combined_manifest(config_path)

    manifest = pd.read_csv(manifest_path)
    assert set(manifest["source_group"]) == {
        "dreambooth_reference",
        "base_easy_control",
        "base_hard_control",
        "vae_roundtrip_control",
    }
    assert (tmp_path / "experiment" / "manifests" / "reference_manifest.csv").exists()
