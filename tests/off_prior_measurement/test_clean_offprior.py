import pandas as pd

from scripts.off_prior_measurement.clean_offprior import compute_clean_offprior


def test_compute_clean_offprior_pairs_roundtrip_and_preserves_image_ids(tmp_path):
    scored_path = tmp_path / "scored_metrics.csv"
    output_dir = tmp_path / "ladder_v2_clean"
    rows = [
        {
            "subject_id": "dog",
            "image_id": "00",
            "source_group": "dreambooth_reference",
            "reference_regime": "standard_reference",
            "hardness_axis": "none",
            "source_standard_image": "",
            "variant_id": "",
            "conditioning_key": "class",
            "timestep": 50,
            "noise_seed": 0,
            "floor_adjusted_l2": 0.30,
            "dct_delta_low": 3.0,
            "dct_delta_mid": 2.0,
            "dct_delta_high": 1.0,
        },
        {
            "subject_id": "dog",
            "image_id": "00",
            "source_group": "vae_roundtrip_control",
            "reference_regime": "roundtrip_control",
            "hardness_axis": "none",
            "source_standard_image": "00",
            "variant_id": "",
            "conditioning_key": "class",
            "timestep": 50,
            "noise_seed": 0,
            "floor_adjusted_l2": 0.10,
            "dct_delta_low": 1.0,
            "dct_delta_mid": 1.0,
            "dct_delta_high": 1.0,
        },
        {
            "subject_id": "dog",
            "image_id": "00__crop_large_subject",
            "source_group": "dreambooth_hard_reference",
            "reference_regime": "hard_reference",
            "hardness_axis": "crop",
            "source_standard_image": "00",
            "variant_id": "crop_large_subject",
            "conditioning_key": "class",
            "timestep": 50,
            "noise_seed": 0,
            "floor_adjusted_l2": 0.45,
            "dct_delta_low": 4.0,
            "dct_delta_mid": 2.0,
            "dct_delta_high": 1.0,
        },
    ]
    pd.DataFrame(rows).to_csv(scored_path, index=False)

    paths = compute_clean_offprior(
        scored_path,
        output_dir,
        source_experiment="experiments/off_prior_measurement_v0/ladder_v2",
    )

    clean = pd.read_csv(paths["clean_scored_metrics"], keep_default_na=False)
    standard = clean[clean["source_group"] == "dreambooth_reference"].iloc[0]
    hard = clean[clean["source_group"] == "dreambooth_hard_reference"].iloc[0]
    assert standard["image_id"] == "00"
    assert round(standard["roundtrip_baseline_l2"], 6) == 0.10
    assert round(standard["clean_pair_l2"], 6) == 0.20
    assert round(hard["clean_pair_l2"], 6) == 0.35
    assert paths["clean_ladder_summary"].exists()
    assert paths["config_resolved"].exists()


def test_compute_clean_offprior_uses_subject_roundtrip_fallback(tmp_path):
    scored_path = tmp_path / "scored_metrics.csv"
    output_dir = tmp_path / "ladder_v2_clean"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "missing_exact_pair",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "source_standard_image": "",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.30,
            },
            {
                "subject_id": "dog",
                "image_id": "00",
                "source_group": "vae_roundtrip_control",
                "reference_regime": "roundtrip_control",
                "hardness_axis": "none",
                "source_standard_image": "00",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.12,
            },
            {
                "subject_id": "dog",
                "image_id": "01",
                "source_group": "vae_roundtrip_control",
                "reference_regime": "roundtrip_control",
                "hardness_axis": "none",
                "source_standard_image": "01",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.08,
            },
        ]
    ).to_csv(scored_path, index=False)

    paths = compute_clean_offprior(
        scored_path,
        output_dir,
        source_experiment="experiments/off_prior_measurement_v0/ladder_v2",
    )

    clean = pd.read_csv(paths["clean_scored_metrics"], keep_default_na=False)
    standard = clean[clean["source_group"] == "dreambooth_reference"].iloc[0]
    assert round(standard["roundtrip_baseline_l2"], 6) == 0.10
    assert round(standard["clean_subject_l2"], 6) == 0.20


def test_compute_clean_offprior_pairs_standard_reference_by_image_path(tmp_path):
    scored_path = tmp_path / "scored_metrics.csv"
    output_dir = tmp_path / "ladder_v2_clean"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": "dataset/dog/00.jpg",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "source_standard_image": "",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.30,
            },
            {
                "subject_id": "dog",
                "image_id": "00",
                "image_path": "roundtrip/dog/00.png",
                "source_group": "vae_roundtrip_control",
                "reference_regime": "roundtrip_control",
                "hardness_axis": "none",
                "source_standard_image": "dataset/dog/00.jpg",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.12,
            },
            {
                "subject_id": "dog",
                "image_id": "01",
                "image_path": "roundtrip/dog/01.png",
                "source_group": "vae_roundtrip_control",
                "reference_regime": "roundtrip_control",
                "hardness_axis": "none",
                "source_standard_image": "dataset/dog/01.jpg",
                "variant_id": "",
                "conditioning_key": "class",
                "timestep": 50,
                "noise_seed": 0,
                "floor_adjusted_l2": 0.02,
            },
        ]
    ).to_csv(scored_path, index=False)

    paths = compute_clean_offprior(
        scored_path,
        output_dir,
        source_experiment="experiments/off_prior_measurement_v0/ladder_v2",
    )

    clean = pd.read_csv(paths["clean_scored_metrics"], keep_default_na=False)
    standard = clean[clean["source_group"] == "dreambooth_reference"].iloc[0]
    assert round(standard["roundtrip_baseline_l2"], 6) == 0.12
    assert round(standard["clean_pair_l2"], 6) == 0.18
