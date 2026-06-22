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
