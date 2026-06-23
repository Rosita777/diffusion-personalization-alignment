import pandas as pd

from scripts.off_prior_measurement.roundtrip_controls import build_roundtrip_manifest, generate_roundtrip_controls


def test_build_roundtrip_manifest_rewrites_reference_rows(tmp_path):
    original = tmp_path / "dog_00.png"
    original.write_bytes(b"fake")
    reference_manifest = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "dog_00",
                "image_path": str(original),
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

    roundtrip_root = tmp_path / "roundtrip"
    (roundtrip_root / "dog").mkdir(parents=True)
    (roundtrip_root / "dog" / "dog_00.png").write_bytes(b"roundtrip")

    manifest = build_roundtrip_manifest(reference_manifest, roundtrip_root)

    assert len(manifest) == 1
    assert manifest.loc[0, "source_group"] == "vae_roundtrip_control"
    assert manifest.loc[0, "reference_regime"] == "roundtrip_control"
    assert manifest.loc[0, "hardness_axis"] == "none"
    assert manifest.loc[0, "source_standard_image"] == str(original)
    assert manifest.loc[0, "variant_id"] == ""
    assert manifest.loc[0, "image_path"].endswith("roundtrip/dog/dog_00.png")


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


def test_generate_roundtrip_controls_preserves_zero_padded_image_ids(tmp_path, monkeypatch):
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
experiment_name: ladder_v2
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
    image_path = tmp_path / "dataset" / "dog" / "00.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake")
    reference_manifest = tmp_path / "reference_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": str(image_path),
            }
        ]
    ).to_csv(reference_manifest, index=False)

    class FakeVae:
        def to(self, device):
            return self

        def eval(self):
            return None

    class FakeAutoencoder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeVae()

    def fake_roundtrip_image(vae, image_path, output_path, resolution, device, dtype):
        del vae, image_path, resolution, device, dtype
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"roundtrip")

    monkeypatch.setattr("diffusers.AutoencoderKL", FakeAutoencoder)
    monkeypatch.setattr("scripts.off_prior_measurement.roundtrip_controls.vae_roundtrip_image", fake_roundtrip_image)

    generate_roundtrip_controls(config_path, reference_manifest)

    assert (tmp_path / "cache" / "vae_roundtrip_controls" / "dog" / "00.png").exists()
    assert not (tmp_path / "cache" / "vae_roundtrip_controls" / "dog" / "0.png").exists()
