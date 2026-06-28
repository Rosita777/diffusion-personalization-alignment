from pathlib import Path

import pandas as pd
import pytest

from scripts.off_prior_measurement.source_decomp_roundtrip import (
    generate_source_decomp_roundtrips_from_manifest,
)


def test_generate_source_decomp_roundtrips_writes_unique_missing_outputs(tmp_path):
    image = tmp_path / "dog.jpg"
    image.write_bytes(b"fake image")
    output = tmp_path / "roundtrip" / "dog.png"
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"image_path": str(image), "roundtrip_image_path": str(output)},
            {"image_path": str(image), "roundtrip_image_path": str(output)},
        ]
    ).to_csv(manifest, index=False)
    calls = []

    def fake_roundtrip(vae, image_path, output_path, resolution, device, dtype):
        calls.append((image_path, output_path, resolution, device, dtype))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"roundtrip")

    written = generate_source_decomp_roundtrips_from_manifest(
        manifest_path=manifest,
        vae=object(),
        resolution=512,
        device="cpu",
        dtype="float32",
        roundtrip_fn=fake_roundtrip,
    )

    assert written == [output]
    assert output.exists()
    assert len(calls) == 1


def test_generate_source_decomp_roundtrips_rejects_missing_inputs(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(tmp_path / "missing.jpg"),
                "roundtrip_image_path": str(tmp_path / "roundtrip.png"),
            }
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        generate_source_decomp_roundtrips_from_manifest(
            manifest_path=manifest,
            vae=object(),
            resolution=512,
            device="cpu",
            dtype="float32",
            roundtrip_fn=lambda *args: None,
        )
