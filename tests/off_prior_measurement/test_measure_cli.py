import numpy as np
import pandas as pd

from scripts.off_prior_measurement.measure import merge_measurement_shards, run_measurement, shard_manifest


class FakeBatch:
    def __init__(self):
        self.v_ref = np.ones((1, 4, 4, 4), dtype=np.float32)
        self.v_base = np.zeros((1, 4, 4, 4), dtype=np.float32)
        self.snr = 1.5


class FakeBackend:
    def measure(self, image_path, prompt, timestep, seed):
        assert image_path.endswith("image.png")
        assert isinstance(prompt, str)
        assert timestep == 50
        assert seed == 0
        return FakeBatch()


class MultiImageFakeBackend:
    def measure(self, image_path, prompt, timestep, seed):
        del prompt, timestep, seed
        batch = FakeBatch()
        batch.image_path_seen = image_path
        return batch


def test_shard_manifest_keeps_rank_slice_order():
    manifest = pd.DataFrame({"image_id": [f"image_{idx}" for idx in range(7)]})

    shard = shard_manifest(manifest, rank=1, world_size=3)

    assert shard["image_id"].tolist() == ["image_1", "image_4"]


def test_run_measurement_writes_raw_metrics(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "image",
                "image_path": str(image_path),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "class_name": "dog",
                "class_prompt": "a photo of a dog",
                "class_context_prompt": "a photo of a dog in a natural scene",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        ]
    ).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "experiment"

    raw_metrics_path = run_measurement(
        manifest_path=manifest_path,
        output_dir=output_dir,
        timesteps=[50],
        noise_seeds=[0],
        backend=FakeBackend(),
    )

    metrics = pd.read_csv(raw_metrics_path)
    assert len(metrics) == 1
    assert metrics.loc[0, "subject_id"] == "dog"
    assert metrics.loc[0, "snr"] == 1.5
    assert metrics.loc[0, "normalized_l2"] > 0.9
    assert {"dct_delta_low", "dct_delta_mid", "dct_delta_high"}.issubset(metrics.columns)


def test_run_measurement_preserves_null_conditioning_key(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "image",
                "image_path": str(image_path),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "class_name": "dog",
                "class_prompt": "a photo of a dog",
                "class_context_prompt": "a photo of a dog in a natural scene",
                "conditioning_key": "null",
                "conditioning_prompt": "",
            }
        ]
    ).to_csv(manifest_path, index=False)

    raw_metrics_path = run_measurement(
        manifest_path=manifest_path,
        output_dir=tmp_path / "experiment",
        timesteps=[50],
        noise_seeds=[0],
        backend=FakeBackend(),
    )

    metrics = pd.read_csv(raw_metrics_path, keep_default_na=False)
    assert metrics.loc[0, "conditioning_key"] == "null"
    assert metrics.loc[0, "conditioning_prompt"] == ""


def test_run_measurement_can_write_rank_shard(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    rows = []
    for idx in range(5):
        image_path = tmp_path / f"image_{idx}.png"
        image_path.write_bytes(b"fake")
        rows.append(
            {
                "subject_id": "dog",
                "image_id": f"image_{idx}",
                "image_path": str(image_path),
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "class_name": "dog",
                "class_prompt": "a photo of a dog",
                "class_context_prompt": "a photo of a dog in a natural scene",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    raw_metrics_path = run_measurement(
        manifest_path=manifest_path,
        output_dir=tmp_path / "experiment",
        timesteps=[50],
        noise_seeds=[0],
        backend=MultiImageFakeBackend(),
        rank=1,
        world_size=2,
        output_name="raw_metrics_rank1.csv",
    )

    metrics = pd.read_csv(raw_metrics_path)
    assert raw_metrics_path.name == "raw_metrics_rank1.csv"
    assert metrics["image_id"].tolist() == ["image_1", "image_3"]


def test_merge_measurement_shards_concatenates_rank_outputs(tmp_path):
    measurements_dir = tmp_path / "experiment" / "measurements"
    measurements_dir.mkdir(parents=True)
    pd.DataFrame([{"image_id": "image_0"}]).to_csv(measurements_dir / "raw_metrics_rank0.csv", index=False)
    pd.DataFrame([{"image_id": "image_1"}]).to_csv(measurements_dir / "raw_metrics_rank1.csv", index=False)

    merged_path = merge_measurement_shards(tmp_path / "experiment", world_size=2)

    merged = pd.read_csv(merged_path)
    assert merged_path.name == "raw_metrics.csv"
    assert merged["image_id"].tolist() == ["image_0", "image_1"]


def test_merge_measurement_shards_restores_empty_null_conditioning_key(tmp_path):
    measurements_dir = tmp_path / "experiment" / "measurements"
    measurements_dir.mkdir(parents=True)
    pd.DataFrame([{"image_id": "image_0", "conditioning_key": ""}]).to_csv(
        measurements_dir / "raw_metrics_rank0.csv", index=False
    )
    pd.DataFrame([{"image_id": "image_1", "conditioning_key": "class"}]).to_csv(
        measurements_dir / "raw_metrics_rank1.csv", index=False
    )

    merged_path = merge_measurement_shards(tmp_path / "experiment", world_size=2)

    merged = pd.read_csv(merged_path, keep_default_na=False)
    assert merged["conditioning_key"].tolist() == ["null", "class"]
