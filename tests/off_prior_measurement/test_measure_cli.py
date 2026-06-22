import numpy as np
import pandas as pd

from scripts.off_prior_measurement.measure import run_measurement


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
