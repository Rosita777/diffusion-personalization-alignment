import pandas as pd

from scripts.off_prior_measurement.plot_clean_offprior import create_clean_figures


def test_create_clean_figures_writes_png_files(tmp_path):
    clean_path = tmp_path / "clean_scored_metrics.csv"
    rows = []
    for subject, standard, hard, ratio in [
        ("dog", 0.20, 0.35, 0.4),
        ("cat", 0.10, 0.12, 0.5),
    ]:
        rows.extend(
            [
                {
                    "subject_id": subject,
                    "source_group": "base_easy_control",
                    "reference_regime": "easy_control",
                    "hardness_axis": "none",
                    "conditioning_key": "class",
                    "timestep": 50,
                    "floor_adjusted_l2": 0.0,
                    "clean_pair_l2": -0.05,
                    "roundtrip_ratio": 0.0,
                    "dct_delta_low": 1.0,
                    "dct_delta_mid": 0.5,
                    "dct_delta_high": 0.25,
                },
                {
                    "subject_id": subject,
                    "source_group": "dreambooth_reference",
                    "reference_regime": "standard_reference",
                    "hardness_axis": "none",
                    "conditioning_key": "class",
                    "timestep": 50,
                    "floor_adjusted_l2": standard + 0.10,
                    "clean_pair_l2": standard,
                    "roundtrip_ratio": ratio,
                    "dct_delta_low": 2.0,
                    "dct_delta_mid": 1.0,
                    "dct_delta_high": 0.5,
                },
                {
                    "subject_id": subject,
                    "source_group": "dreambooth_hard_reference",
                    "reference_regime": "hard_reference",
                    "hardness_axis": "crop",
                    "conditioning_key": "class",
                    "timestep": 50,
                    "floor_adjusted_l2": hard + 0.10,
                    "clean_pair_l2": hard,
                    "roundtrip_ratio": ratio,
                    "dct_delta_low": 3.0,
                    "dct_delta_mid": 2.0,
                    "dct_delta_high": 1.0,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(clean_path, index=False)

    paths = create_clean_figures(clean_path, tmp_path / "figures")

    assert paths["raw_vs_clean_ladder"].exists()
    assert paths["roundtrip_attribution_by_subject"].exists()
    assert paths["clean_timestep_curves"].exists()
    assert paths["clean_frequency_heatmap"].exists()
