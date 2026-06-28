import pandas as pd

from scripts.off_prior_measurement.source_decomp_conclusion import write_source_decomp_conclusion


def _write_summaries(experiment_dir, gaps, artifact_fraction):
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "class_name": f"class_{idx}",
                "conditioning_key": "class",
                "base_generated_control": 0.10,
                "ordinary_real_control": 0.20,
                "dreambooth_reference": 0.20 + gap,
                "natural_hard_reference": 0.25 + gap,
                "real_domain_gap": 0.10,
                "subject_specific_gap": gap,
                "natural_hard_gap": 0.05,
            }
            for idx, gap in enumerate(gaps)
        ]
    ).to_csv(summaries / "source_gap_summary.csv", index=False)
    rows = []
    for idx, gap in enumerate(gaps):
        rows.append(
            {
                "class_name": f"class_{idx}",
                "conditioning_key": "class",
                "source_group": "dreambooth_reference",
                "mean_clean_norm": 0.20 + gap,
                "mean_raw_norm": 0.30 + gap,
                "mean_artifact_fraction": artifact_fraction,
                "mean_artifact_cosine": 0.30,
                "n": 1,
            }
        )
    pd.DataFrame(rows).to_csv(summaries / "source_group_summary.csv", index=False)


def test_write_source_decomp_conclusion_reports_go(tmp_path):
    experiment_dir = tmp_path / "experiment"
    _write_summaries(experiment_dir, gaps=[0.15, 0.10, 0.05, -0.01], artifact_fraction=0.30)

    path = write_source_decomp_conclusion(experiment_dir)

    text = path.read_text(encoding="utf-8")
    assert "# Target-Gap Source Decomposition Conclusion" in text
    assert "Stage 1.4 Target-Gap Source Decomposition Conclusion" not in text
    assert "Subject-specific positive classes: 3 of 4" in text
    assert "Go / pivot decision: Go" in text


def test_write_source_decomp_conclusion_reports_pivot(tmp_path):
    experiment_dir = tmp_path / "experiment"
    _write_summaries(experiment_dir, gaps=[0.00, -0.02, -0.03, -0.01], artifact_fraction=0.90)

    path = write_source_decomp_conclusion(experiment_dir)

    text = path.read_text(encoding="utf-8")
    assert "Mean DreamBooth artifact fraction: 0.9000" in text
    assert "Go / pivot decision: Pivot" in text


def test_write_source_decomp_conclusion_handles_string_numeric_columns(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
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
    ).to_csv(summaries / "source_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": "dreambooth_reference",
                "mean_clean_norm": "0.35",
                "mean_raw_norm": "0.40",
                "mean_artifact_fraction": "0.20",
                "mean_artifact_cosine": "0.30",
                "n": "1",
            }
        ]
    ).to_csv(summaries / "source_group_summary.csv", index=False)

    path = write_source_decomp_conclusion(experiment_dir)

    text = path.read_text(encoding="utf-8")
    assert "Mean subject-specific gap: 0.1500" in text


def test_write_source_decomp_conclusion_selects_prompt_matched_conditioning(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "base_generated_control": 0.10,
                "ordinary_real_control": 0.12,
                "dreambooth_reference": "",
                "natural_hard_reference": "",
                "real_domain_gap": 0.02,
                "subject_specific_gap": "",
                "natural_hard_gap": "",
            }
        ]
    ).to_csv(summaries / "source_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "source_group": "ordinary_real_control",
                "mean_clean_norm": 0.12,
                "mean_raw_norm": 0.20,
                "mean_artifact_fraction": 0.30,
                "mean_artifact_cosine": 0.40,
                "n": 1,
            }
        ]
    ).to_csv(summaries / "source_group_summary.csv", index=False)

    path = write_source_decomp_conclusion(experiment_dir)

    text = path.read_text(encoding="utf-8")
    assert "prompt_matched" in text
    assert "Subject-specific positive classes: 0 of 1" in text


def test_write_source_decomp_conclusion_marks_control_only_diagnosis(tmp_path):
    experiment_dir = tmp_path / "experiment"
    summaries = experiment_dir / "summaries"
    summaries.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "base_generated_control": 0.10,
                "ordinary_real_control": 0.12,
                "dreambooth_reference": "",
                "natural_hard_reference": "",
                "real_domain_gap": 0.02,
                "subject_specific_gap": "",
                "natural_hard_gap": "",
            }
        ]
    ).to_csv(summaries / "source_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "source_group": "ordinary_real_control",
                "mean_clean_norm": 0.12,
                "mean_raw_norm": 0.20,
                "mean_artifact_fraction": 0.30,
                "mean_artifact_cosine": 0.40,
                "n": 1,
            }
        ]
    ).to_csv(summaries / "source_group_summary.csv", index=False)

    path = write_source_decomp_conclusion(experiment_dir)

    text = path.read_text(encoding="utf-8")
    assert "DreamBooth reference rows present: False" in text
    assert "Go / pivot decision: Control-only diagnosis" in text
