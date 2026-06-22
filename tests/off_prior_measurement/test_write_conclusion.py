import pandas as pd

from scripts.off_prior_measurement.write_conclusion import write_conclusion


def test_write_conclusion_uses_summary_csvs(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "mean_floor_adjusted_l2": 0.3,
                "median_floor_adjusted_l2": 0.3,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
            {
                "subject_id": "cat",
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "mean_floor_adjusted_l2": 0.4,
                "median_floor_adjusted_l2": 0.4,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
            {
                "subject_id": "backpack",
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "mean_floor_adjusted_l2": 0.5,
                "median_floor_adjusted_l2": 0.5,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
            {
                "subject_id": "colorful_sneaker",
                "source_group": "dreambooth_reference",
                "conditioning_key": "class",
                "mean_floor_adjusted_l2": 0.2,
                "median_floor_adjusted_l2": 0.2,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
            {
                "subject_id": "vase",
                "source_group": "dreambooth_reference",
                "conditioning_key": "class_context",
                "mean_floor_adjusted_l2": 0.6,
                "median_floor_adjusted_l2": 0.6,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
            {
                "subject_id": "dog",
                "source_group": "dreambooth_reference",
                "conditioning_key": "null",
                "mean_floor_adjusted_l2": -0.1,
                "median_floor_adjusted_l2": -0.1,
                "mean_cosine_distance": 0.2,
                "n": 1,
            },
        ]
    ).to_csv(summaries / "subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "reference_regime": "standard",
                "conditioning_key": "class",
                "timestep": 800,
                "mean_floor_adjusted_l2": 0.7,
            }
        ]
    ).to_csv(summaries / "regime_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "dct_delta_low": 3.0,
                "dct_delta_mid": 2.0,
                "dct_delta_high": 1.0,
            }
        ]
    ).to_csv(summaries / "scored_metrics.csv", index=False)

    conclusion_path = write_conclusion(experiment_dir)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "Go / no-go decision: Go" in text
    assert "Strongest timestep by mean floor-adjusted residual: 800" in text
    assert "Strongest latent DCT band by mean residual energy: low" in text
