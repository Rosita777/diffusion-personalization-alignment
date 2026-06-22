import pandas as pd

from scripts.off_prior_measurement.plot import create_figures


def test_create_figures_writes_png_files(tmp_path):
    scored_path = tmp_path / "scored_metrics.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "source_group": "base_easy_control",
                "reference_regime": "easy_control",
                "conditioning_key": "class",
                "timestep": 50,
                "floor_adjusted_l2": 0.0,
                "normalized_l2": 0.2,
                "dct_delta_low": 1.0,
                "dct_delta_mid": 0.5,
                "dct_delta_high": 0.25,
            },
            {
                "subject_id": "dog",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "conditioning_key": "class",
                "timestep": 50,
                "floor_adjusted_l2": 0.3,
                "normalized_l2": 0.5,
                "dct_delta_low": 2.0,
                "dct_delta_mid": 1.0,
                "dct_delta_high": 0.5,
            },
        ]
    ).to_csv(scored_path, index=False)

    paths = create_figures(scored_path, tmp_path / "figures")

    assert paths["control_distribution"].exists()
    assert paths["timestep_curves"].exists()
    assert paths["frequency_heatmap"].exists()
