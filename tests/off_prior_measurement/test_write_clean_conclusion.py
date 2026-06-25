import pandas as pd

from scripts.off_prior_measurement.write_clean_conclusion import write_clean_conclusion


def _write_clean_experiment(experiment_dir, standard_values, hard_offsets, ratios):
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    rows = []
    for idx, standard in enumerate(standard_values):
        rows.append(
            {
                "subject_id": f"subject_{idx}",
                "conditioning_key": "class",
                "raw_easy_control": 0.0,
                "raw_standard_reference": standard + 0.10,
                "raw_hard_reference": standard + hard_offsets[idx] + 0.10,
                "raw_roundtrip_control": 0.10,
                "clean_standard_reference": standard,
                "clean_hard_reference": standard + hard_offsets[idx],
                "ratio_standard_reference": ratios[idx],
                "raw_standard_minus_easy": standard + 0.10,
                "raw_hard_minus_standard": hard_offsets[idx],
                "clean_hard_minus_standard": hard_offsets[idx],
            }
        )
    pd.DataFrame(rows).to_csv(summaries / "clean_ladder_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "timestep": 50,
                "mean_clean_pair_l2": 0.2,
                "mean_raw_l2": 0.3,
                "mean_roundtrip_ratio": 0.4,
                "n": 1,
            }
        ]
    ).to_csv(summaries / "clean_regime_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "subject_0",
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "conditioning_key": "class",
                "mean_clean_pair_l2": 0.2,
                "mean_raw_l2": 0.3,
                "mean_roundtrip_ratio": 0.4,
                "n": 1,
            }
        ]
    ).to_csv(summaries / "clean_subject_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_group": "dreambooth_reference",
                "reference_regime": "standard_reference",
                "hardness_axis": "none",
                "conditioning_key": "class",
                "timestep": 50,
                "clean_pair_l2": 0.2,
                "roundtrip_ratio": 0.4,
            }
        ]
    ).to_csv(summaries / "clean_scored_metrics.csv", index=False)


def test_write_clean_conclusion_reports_go(tmp_path):
    experiment_dir = tmp_path / "clean"
    _write_clean_experiment(
        experiment_dir,
        standard_values=[0.12, 0.10, 0.08, 0.06, 0.05, -0.01, -0.02, -0.03],
        hard_offsets=[0.03, 0.02, 0.01, 0.02, 0.00, 0.01, -0.01, -0.02],
        ratios=[0.40, 0.45, 0.50, 0.35, 0.30, 0.40, 0.45, 0.50],
    )

    conclusion_path = write_clean_conclusion(experiment_dir)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "Stage 1.3 Clean Off-Priorness Conclusion" in text
    assert "Clean standard-reference positive subjects: 5 of 8" in text
    assert "Go / no-go decision: Go" in text


def test_write_clean_conclusion_reports_no_go_when_roundtrip_dominates(tmp_path):
    experiment_dir = tmp_path / "clean"
    _write_clean_experiment(
        experiment_dir,
        standard_values=[0.12, 0.10, 0.08, 0.06, 0.05, -0.01, -0.02, -0.03],
        hard_offsets=[0.03, 0.02, 0.01, 0.02, 0.00, 0.01, -0.01, -0.02],
        ratios=[0.90, 0.85, 0.80, 0.95, 0.88, 0.92, 0.86, 0.91],
    )

    conclusion_path = write_clean_conclusion(experiment_dir)

    text = conclusion_path.read_text(encoding="utf-8")
    assert "Mean standard-reference roundtrip attribution ratio: 0.8838" in text
    assert "Go / no-go decision: No-Go" in text
