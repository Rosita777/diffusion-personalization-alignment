from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.off_prior_measurement.source_decomp_measure import (
    merge_source_decomp_shards,
    run_source_decomp_measurement,
)


@dataclass
class FakeBatch:
    v_ref: np.ndarray
    v_base: np.ndarray
    snr: float = 1.0


class FakeBackend:
    def measure(self, image_path, prompt, timestep, seed):
        if "roundtrip" in str(image_path):
            return FakeBatch(
                v_ref=np.array([[[[2.0, 0.0], [0.0, 0.0]]]], dtype=np.float32),
                v_base=np.zeros((1, 1, 2, 2), dtype=np.float32),
            )
        return FakeBatch(
            v_ref=np.array([[[[2.0, 2.0], [0.0, 0.0]]]], dtype=np.float32),
            v_base=np.zeros((1, 1, 2, 2), dtype=np.float32),
        )


def test_run_source_decomp_measurement_writes_projection_metrics(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": "00",
                "image_path": "dog.jpg",
                "roundtrip_image_path": "roundtrip/dog.png",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
                "source_dataset": "dreambooth",
                "source_license_note": "test",
            }
        ]
    ).to_csv(manifest, index=False)

    path = run_source_decomp_measurement(
        manifest_path=manifest,
        output_dir=tmp_path / "experiment",
        timesteps=[50],
        noise_seeds=[0],
        backend=FakeBackend(),
    )

    rows = pd.read_csv(path)
    assert round(rows.iloc[0]["artifact_fraction"], 6) > 0.0
    assert round(rows.iloc[0]["clean_fraction"], 6) > 0.0
    assert rows.iloc[0]["source_group"] == "dreambooth_reference"
    assert "dct_clean_low" in rows.columns
    assert "dct_artifact_low" in rows.columns


def test_run_source_decomp_measurement_can_write_rank_shard(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = []
    for idx in range(5):
        rows.append(
            {
                "subject_id": "dog",
                "class_name": "dog",
                "image_id": f"image_{idx}",
                "image_path": f"dog_{idx}.jpg",
                "roundtrip_image_path": f"roundtrip/dog_{idx}.png",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "conditioning_prompt": "a photo of a dog",
                "source_dataset": "dreambooth",
                "source_license_note": "test",
            }
        )
    pd.DataFrame(rows).to_csv(manifest, index=False)

    path = run_source_decomp_measurement(
        manifest_path=manifest,
        output_dir=tmp_path / "experiment",
        timesteps=[50],
        noise_seeds=[0],
        backend=FakeBackend(),
        rank=1,
        world_size=2,
        output_name="raw_source_decomp_metrics_rank1.csv",
    )

    metrics = pd.read_csv(path)
    assert path.name == "raw_source_decomp_metrics_rank1.csv"
    assert metrics["image_id"].tolist() == ["image_1", "image_3"]


def test_merge_source_decomp_shards_concatenates_rank_outputs(tmp_path):
    measurements_dir = tmp_path / "experiment" / "measurements"
    measurements_dir.mkdir(parents=True)
    pd.DataFrame([{"image_id": "image_0"}]).to_csv(
        measurements_dir / "raw_source_decomp_metrics_rank0.csv",
        index=False,
    )
    pd.DataFrame([{"image_id": "image_1"}]).to_csv(
        measurements_dir / "raw_source_decomp_metrics_rank1.csv",
        index=False,
    )

    merged_path = merge_source_decomp_shards(tmp_path / "experiment", world_size=2)

    merged = pd.read_csv(merged_path)
    assert merged_path.name == "raw_source_decomp_metrics.csv"
    assert merged["image_id"].tolist() == ["image_0", "image_1"]
