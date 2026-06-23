import io
import base64
import http.client
import zipfile

import pandas as pd

from scripts.off_prior_measurement.config import ExperimentConfig, SubjectSpec
from scripts.off_prior_measurement.dreambooth_data import (
    build_reference_manifest,
    conditioning_prompt,
    download_dreambooth_subjects,
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
    assert len(manifest) == 6
    assert set(manifest["source_group"]) == {"dreambooth_reference"}
    assert set(manifest["reference_regime"]) == {"standard_reference"}
    assert set(manifest["hardness_axis"]) == {"none"}
    assert set(manifest["variant_id"]) == {""}
    assert set(manifest["conditioning_key"]) == {"null", "class", "class_context"}
    assert isinstance(manifest, pd.DataFrame)


def test_build_reference_manifest_can_limit_images_per_subject(tmp_path):
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
    for name in ["00.png", "01.png", "02.png"]:
        (image_dir / name).write_bytes(b"fake")

    manifest = build_reference_manifest(
        dataset_root=tmp_path / "dataset",
        subjects=[subject],
        conditionings=["class"],
        image_limit=2,
    )

    assert manifest["image_id"].tolist() == ["00", "01"]


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
hard_reference_variants:
  - crop_large_subject
hard_reference_limit_per_subject: 1
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
    hard_root = tmp_path / "cache" / "hard_references" / "dog"
    hard_root.mkdir(parents=True)
    (hard_root / "00__crop_large_subject.png").write_bytes(b"hard-reference")

    monkeypatch.setattr(
        "scripts.off_prior_measurement.dreambooth_data.download_dreambooth_subjects",
        lambda config: dataset_root,
    )

    manifest_path = write_combined_manifest(config_path)

    manifest = pd.read_csv(manifest_path)
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
    assert (tmp_path / "experiment" / "manifests" / "reference_manifest.csv").exists()
    assert (tmp_path / "experiment" / "manifests" / "hard_reference_manifest.csv").exists()


def test_download_dreambooth_subjects_falls_back_to_github(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    config = ExperimentConfig(
        experiment_name="smoke_test",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
    )

    def fail_snapshot_download(*args, **kwargs):
        raise OSError("huggingface offline")

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    api_user_agents = []

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        if "api.github.com" in url:
            if hasattr(request, "header_items"):
                api_user_agents.append(dict(request.header_items()).get("User-agent"))
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"download_url": "https://raw.githubusercontent.com/google/dreambooth/main/dataset/dog/00.jpg"}]'
            )
        return FakeResponse(b"reference-image")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot_download)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dataset_root = download_dreambooth_subjects(config)

    image_path = dataset_root / "dog" / "00.jpg"
    assert dataset_root == tmp_path / "cache" / "dreambooth_dataset" / "dataset"
    assert api_user_agents == ["diffusion-personalization-target-alignment"]
    assert image_path.read_bytes() == b"reference-image"


def test_download_dreambooth_subjects_can_use_github_directly(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    config = ExperimentConfig(
        experiment_name="smoke_test",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github",
    )
    called_huggingface = False

    def fail_if_called(*args, **kwargs):
        nonlocal called_huggingface
        called_huggingface = True
        raise AssertionError("Hugging Face should not be called in github dataset_source mode")

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    api_user_agents = []

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        if "api.github.com" in url:
            if hasattr(request, "header_items"):
                api_user_agents.append(dict(request.header_items()).get("User-agent"))
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"download_url": "https://raw.githubusercontent.com/google/dreambooth/main/dataset/dog/00.jpg"}]'
            )
        return FakeResponse(b"github-reference")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_if_called)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dataset_root = download_dreambooth_subjects(config)

    assert not called_huggingface
    assert api_user_agents == ["diffusion-personalization-target-alignment"]
    assert (dataset_root / "dog" / "00.jpg").read_bytes() == b"github-reference"


def test_download_dreambooth_subjects_prefers_github_zip(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    config = ExperimentConfig(
        experiment_name="smoke_test",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github",
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dreambooth-main/dataset/dog/00.jpg", b"zip-reference")
        handle.writestr("dreambooth-main/dataset/cat/00.jpg", b"unused")

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1) -> bytes:
            if size == -1:
                start = self.offset
                self.offset = len(self.payload)
                return self.payload[start:]
            start = self.offset
            end = min(len(self.payload), start + size)
            self.offset = end
            return self.payload[start:end]

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        if "codeload.github.com" in url:
            return FakeResponse(archive.getvalue())
        raise AssertionError(f"Unexpected non-zip request: {url}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dataset_root = download_dreambooth_subjects(config)

    assert (dataset_root / "dog" / "00.jpg").read_bytes() == b"zip-reference"
    assert not (dataset_root / "cat").exists()


def test_download_dreambooth_subjects_can_use_github_api_contents(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="clock",
        hf_subset="clock",
        class_name="clock",
        class_prompt="a photo of a clock",
        class_context_prompt="a photo of a clock on a wall",
        hard_control_prompt="a photo of a clock in a dark forest",
    )
    config = ExperimentConfig(
        experiment_name="smoke_test",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github_api",
    )
    encoded = base64.b64encode(b"api-reference").decode("ascii")

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        headers = dict(request.header_items())
        assert headers.get("Authorization") == "Bearer test-token"
        if url.endswith("/dataset/clock?ref=main"):
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"url": "https://api.github.com/repos/google/dreambooth/contents/dataset/clock/00.jpg?ref=main"}]'
            )
        if url.endswith("/dataset/clock/00.jpg?ref=main"):
            return FakeResponse(
                (
                    '{"name": "00.jpg", "encoding": "base64", "content": "'
                    + encoded
                    + '"}'
                ).encode("utf-8")
            )
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    dataset_root = download_dreambooth_subjects(config)

    assert (dataset_root / "clock" / "00.jpg").read_bytes() == b"api-reference"


def test_download_dreambooth_subjects_uses_git_blob_for_large_github_api_files(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="colorful_sneaker",
        hf_subset="colorful_sneaker",
        class_name="sneaker",
        class_prompt="a photo of a sneaker",
        class_context_prompt="a photo of a sneaker on the floor",
        hard_control_prompt="a photo of a colorful sneaker under neon light",
    )
    config = ExperimentConfig(
        experiment_name="ladder_v2",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github_api",
    )
    encoded = base64.b64encode(b"large-api-reference").decode("ascii")

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    requested_urls = []

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        requested_urls.append(url)
        if url.endswith("/dataset/colorful_sneaker?ref=main"):
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"url": "https://api.github.com/repos/google/dreambooth/contents/dataset/colorful_sneaker/00.jpg?ref=main"}]'
            )
        if url.endswith("/dataset/colorful_sneaker/00.jpg?ref=main"):
            return FakeResponse(
                b'{"name": "00.jpg", "encoding": "none", "content": "", '
                b'"git_url": "https://api.github.com/repos/google/dreambooth/git/blobs/blob-sha"}'
            )
        if url.endswith("/git/blobs/blob-sha"):
            return FakeResponse(
                (
                    '{"sha": "blob-sha", "encoding": "base64", "content": "'
                    + encoded
                    + '"}'
                ).encode("utf-8")
            )
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    dataset_root = download_dreambooth_subjects(config)

    assert requested_urls[-1].endswith("/git/blobs/blob-sha")
    assert (dataset_root / "colorful_sneaker" / "00.jpg").read_bytes() == b"large-api-reference"


def test_download_dreambooth_subjects_respects_reference_image_limit_for_github_api(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="dog",
        hf_subset="dog",
        class_name="dog",
        class_prompt="a photo of a dog",
        class_context_prompt="a photo of a dog in a natural scene",
        hard_control_prompt="a photo of a dog under dramatic stage lighting",
    )
    config = ExperimentConfig(
        experiment_name="ladder_v2",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github_api",
        reference_images_per_subject=1,
    )

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return self.payload

    requested_urls = []

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        requested_urls.append(url)
        if url.endswith("/dataset/dog?ref=main"):
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"url": "https://api.github.com/repos/google/dreambooth/contents/dataset/dog/00.jpg?ref=main"},'
                b'{"name": "01.jpg", "type": "file", '
                b'"url": "https://api.github.com/repos/google/dreambooth/contents/dataset/dog/01.jpg?ref=main"}]'
            )
        if url.endswith("/dataset/dog/00.jpg?ref=main"):
            encoded = base64.b64encode(b"first-reference").decode("ascii")
            return FakeResponse(
                (
                    '{"name": "00.jpg", "encoding": "base64", "content": "'
                    + encoded
                    + '"}'
                ).encode("utf-8")
            )
        if url.endswith("/dataset/dog/01.jpg?ref=main"):
            raise AssertionError("reference_images_per_subject should skip 01.jpg")
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dataset_root = download_dreambooth_subjects(config)

    assert len([url for url in requested_urls if url.endswith(".jpg?ref=main")]) == 1
    assert (dataset_root / "dog" / "00.jpg").read_bytes() == b"first-reference"
    assert not (dataset_root / "dog" / "01.jpg").exists()


def test_download_dreambooth_subjects_retries_incomplete_git_blob_reads(tmp_path, monkeypatch):
    subject = SubjectSpec(
        subject_id="shiny_sneaker",
        hf_subset="shiny_sneaker",
        class_name="sneaker",
        class_prompt="a photo of a sneaker",
        class_context_prompt="a photo of a sneaker on the floor",
        hard_control_prompt="a photo of a shiny sneaker with reflections",
    )
    config = ExperimentConfig(
        experiment_name="ladder_v2",
        model_id="runwayml/stable-diffusion-v1-5",
        prediction_type="epsilon",
        device="cpu",
        dtype="float32",
        resolution=512,
        dataset_repo="google/dreambooth",
        subject_manifest=tmp_path / "subjects.yaml",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "experiment",
        debug_output_dir=tmp_path / "outputs",
        timesteps=[50],
        noise_seeds=[0],
        conditionings=["class"],
        control_images_per_subject=1,
        batch_size=1,
        save_debug_tensors=False,
        subjects=[subject],
        dataset_source="github_api",
    )
    encoded = base64.b64encode(b"retry-api-reference").decode("ascii")

    class FakeResponse:
        def __init__(self, payload: bytes, fail_once: bool = False):
            self.payload = payload
            self.fail_once = fail_once

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            if self.fail_once:
                raise http.client.IncompleteRead(b"partial", 7)
            return self.payload

    blob_attempts = {"count": 0}

    def fake_urlopen(request, timeout):
        url = getattr(request, "full_url", request)
        if url.endswith("/dataset/shiny_sneaker?ref=main"):
            return FakeResponse(
                b'[{"name": "00.jpg", "type": "file", '
                b'"url": "https://api.github.com/repos/google/dreambooth/contents/dataset/shiny_sneaker/00.jpg?ref=main"}]'
            )
        if url.endswith("/dataset/shiny_sneaker/00.jpg?ref=main"):
            return FakeResponse(
                b'{"name": "00.jpg", "encoding": "none", "content": "", '
                b'"git_url": "https://api.github.com/repos/google/dreambooth/git/blobs/retry-sha"}'
            )
        if url.endswith("/git/blobs/retry-sha"):
            blob_attempts["count"] += 1
            if blob_attempts["count"] == 1:
                return FakeResponse(b"", fail_once=True)
            return FakeResponse(
                (
                    '{"sha": "retry-sha", "encoding": "base64", "content": "'
                    + encoded
                    + '"}'
                ).encode("utf-8")
            )
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dataset_root = download_dreambooth_subjects(config)

    assert blob_attempts["count"] == 2
    assert (dataset_root / "shiny_sneaker" / "00.jpg").read_bytes() == b"retry-api-reference"
