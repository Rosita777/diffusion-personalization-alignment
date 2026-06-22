import pandas as pd

from scripts.off_prior_measurement.roundtrip_controls import build_roundtrip_manifest


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
                "reference_regime": "standard",
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
    assert manifest.loc[0, "image_path"].endswith("roundtrip/dog/dog_00.png")
