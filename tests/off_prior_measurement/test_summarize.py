import pandas as pd

from scripts.off_prior_measurement.summarize import summarize_metrics


def test_summarize_metrics_computes_floor_adjusted_scores(tmp_path):
    raw_path = tmp_path / "raw_metrics.csv"
    output_dir = tmp_path / "experiment"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "conditioning_key": "class",
                "timestep": 50,
                "normalized_l2": 0.2,
                "cosine_distance": 0.1,
                "dct_delta_low": 1.0,
                "dct_delta_mid": 1.0,
                "dct_delta_high": 1.0,
            },
            {
                "subject_id": "dog",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "conditioning_key": "class",
                "timestep": 50,
                "normalized_l2": 0.5,
                "cosine_distance": 0.2,
                "dct_delta_low": 2.0,
                "dct_delta_mid": 2.0,
                "dct_delta_high": 2.0,
            },
        ]
    ).to_csv(raw_path, index=False)

    paths = summarize_metrics(raw_path, output_dir)

    regime_summary = pd.read_csv(paths["regime_summary"])
    reference_row = regime_summary[regime_summary["source_group"] == "dreambooth_reference"].iloc[0]
    assert round(reference_row["mean_floor_adjusted_l2"], 6) == 0.3


def test_summarize_metrics_restores_empty_null_conditioning_key(tmp_path):
    raw_path = tmp_path / "raw_metrics.csv"
    output_dir = tmp_path / "experiment"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "conditioning_key": "",
                "timestep": 50,
                "normalized_l2": 0.2,
                "cosine_distance": 0.1,
            },
            {
                "subject_id": "dog",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "conditioning_key": "",
                "timestep": 50,
                "normalized_l2": 0.5,
                "cosine_distance": 0.2,
            },
        ]
    ).to_csv(raw_path, index=False)

    paths = summarize_metrics(raw_path, output_dir)

    regime_summary = pd.read_csv(paths["regime_summary"], keep_default_na=False)
    assert set(regime_summary["conditioning_key"]) == {"null"}


def test_summarize_metrics_writes_ladder_summary(tmp_path):
    raw_path = tmp_path / "raw_metrics.csv"
    output_dir = tmp_path / "experiment"
    rows = []
    for regime, group, value in [
        ("easy_control", "base_easy_control", 0.20),
        ("standard_reference", "dreambooth_reference", 0.35),
        ("hard_reference", "dreambooth_hard_reference", 0.60),
        ("hard_control", "base_hard_control", 0.50),
        ("roundtrip_control", "vae_roundtrip_control", 0.30),
    ]:
        rows.append(
            {
                "subject_id": "dog",
                "source_group": group,
                "reference_regime": regime,
                "hardness_axis": "none",
                "conditioning_key": "class",
                "timestep": 50,
                "normalized_l2": value,
                "cosine_distance": 0.1,
                "dct_delta_low": value,
                "dct_delta_mid": value,
                "dct_delta_high": value,
            }
        )
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    paths = summarize_metrics(raw_path, output_dir)

    ladder = pd.read_csv(paths["ladder_summary"])
    row = ladder.iloc[0]
    assert round(row["standard_minus_easy"], 6) == 0.15
    assert round(row["hard_minus_standard"], 6) == 0.25
    assert bool(row["ladder_monotonic"]) is True
