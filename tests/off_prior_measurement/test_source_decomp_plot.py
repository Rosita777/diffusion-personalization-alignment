import pandas as pd

from scripts.off_prior_measurement.source_decomp_plot import create_source_decomp_figures


def test_create_source_decomp_figures_writes_png_files(tmp_path):
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "base_generated_control": 0.10,
                "ordinary_real_control": 0.20,
                "dreambooth_reference": 0.35,
                "natural_hard_reference": 0.45,
                "real_domain_gap": 0.10,
                "subject_specific_gap": 0.15,
                "natural_hard_gap": 0.10,
            }
        ]
    ).to_csv(summary_dir / "source_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": source,
                "mean_clean_norm": clean,
                "mean_raw_norm": clean + 0.05,
                "mean_artifact_fraction": 0.2,
                "mean_artifact_cosine": 0.3,
                "n": 1,
            }
            for source, clean in [
                ("base_generated_control", 0.1),
                ("ordinary_real_control", 0.2),
                ("dreambooth_reference", 0.35),
            ]
        ]
    ).to_csv(summary_dir / "source_group_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "timestep": 50,
                "mean_clean_norm": 0.35,
                "mean_raw_norm": 0.40,
                "mean_artifact_fraction": 0.2,
                "mean_dct_clean_low": 0.3,
                "mean_dct_clean_mid": 0.2,
                "mean_dct_clean_high": 0.1,
                "n": 1,
            }
        ]
    ).to_csv(summary_dir / "timestep_frequency_summary.csv", index=False)

    paths = create_source_decomp_figures(summary_dir, tmp_path / "figures")

    assert paths["source_gap_bars"].exists()
    assert paths["artifact_fraction_by_source"].exists()
    assert paths["clean_timestep_curves"].exists()


def test_create_source_decomp_figures_handles_string_numeric_columns(tmp_path):
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "base_generated_control": "0.10",
                "ordinary_real_control": "0.20",
                "dreambooth_reference": "0.35",
                "natural_hard_reference": "",
                "real_domain_gap": "0.10",
                "subject_specific_gap": "0.15",
                "natural_hard_gap": "",
            }
        ]
    ).to_csv(summary_dir / "source_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": source,
                "mean_clean_norm": str(clean),
                "mean_raw_norm": str(clean + 0.05),
                "mean_artifact_fraction": "0.2",
                "mean_artifact_cosine": "0.3",
                "n": "1",
            }
            for source, clean in [
                ("base_generated_control", 0.1),
                ("ordinary_real_control", 0.2),
                ("dreambooth_reference", 0.35),
            ]
        ]
    ).to_csv(summary_dir / "source_group_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "timestep": "50",
                "mean_clean_norm": "0.35",
                "mean_raw_norm": "0.40",
                "mean_artifact_fraction": "0.2",
                "mean_dct_clean_low": "0.3",
                "mean_dct_clean_mid": "0.2",
                "mean_dct_clean_high": "0.1",
                "n": "1",
            }
        ]
    ).to_csv(summary_dir / "timestep_frequency_summary.csv", index=False)

    paths = create_source_decomp_figures(summary_dir, tmp_path / "figures")

    assert paths["source_gap_bars"].exists()
