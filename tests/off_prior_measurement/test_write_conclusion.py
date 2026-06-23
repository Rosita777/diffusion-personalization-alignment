import pandas as pd

from scripts.off_prior_measurement.write_conclusion import write_conclusion


def test_write_conclusion_uses_ladder_go_no_go(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    subjects = [f"subject_{idx}" for idx in range(8)]
    ladder_rows = []
    subject_rows = []
    for idx, subject in enumerate(subjects):
        standard = 0.2 if idx < 4 else -0.05
        hard = standard + 0.3 if idx < 6 else standard - 0.1
        ladder_rows.append(
            {
                "subject_id": subject,
                "conditioning_key": "class",
                "easy_control": 0.0,
                "standard_reference": standard,
                "hard_reference": hard,
                "hard_control": 0.25,
                "roundtrip_control": 0.1,
                "standard_minus_easy": standard,
                "hard_minus_standard": hard - standard,
                "ladder_monotonic": hard > standard > 0.0,
            }
        )
        subject_rows.extend(
            [
                {
                    "subject_id": subject,
                    "source_group": "dreambooth_hard_reference",
                    "conditioning_key": "class",
                    "mean_floor_adjusted_l2": hard,
                    "median_floor_adjusted_l2": hard,
                    "mean_cosine_distance": 0.2,
                    "n": 1,
                },
                {
                    "subject_id": subject,
                    "source_group": "base_hard_control",
                    "conditioning_key": "class",
                    "mean_floor_adjusted_l2": 0.25,
                    "median_floor_adjusted_l2": 0.25,
                    "mean_cosine_distance": 0.2,
                    "n": 1,
                },
            ]
        )
    pd.DataFrame(ladder_rows).to_csv(summaries / "ladder_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(summaries / "subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_hard_reference",
                "reference_regime": "hard_reference",
                "hardness_axis": "crop",
                "conditioning_key": "class",
                "timestep": 800,
                "mean_floor_adjusted_l2": 0.7,
            }
        ]
    ).to_csv(summaries / "regime_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_hard_reference",
                "reference_regime": "hard_reference",
                "hardness_axis": "crop",
                "dct_delta_low": 3.0,
                "dct_delta_mid": 2.0,
                "dct_delta_high": 1.0,
            }
        ]
    ).to_csv(summaries / "scored_metrics.csv", index=False)

    conclusion_path = write_conclusion(experiment_dir)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "Stage 1 V2 Prior-Compatibility Ladder Conclusion" in text
    assert "Hard-reference positive subjects: 6 of 8" in text
    assert "Hard greater than standard: 6 of 8" in text
    assert "Standard greater than easy: 4 of 8" in text
    assert "Go / no-go decision: Go" in text
