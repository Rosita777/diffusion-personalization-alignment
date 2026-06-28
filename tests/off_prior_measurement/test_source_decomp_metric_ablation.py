import pandas as pd

from scripts.off_prior_measurement.source_decomp_metric_ablation import (
    combine_metric_ablation_gaps,
    summarize_metric_ablation,
)


def test_summarize_metric_ablation_computes_projected_artifact_gap(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "source_group": "base_generated_control",
                "raw_norm": 0.10,
                "clean_norm": 0.02,
                "artifact_fraction": 0.90,
                "artifact_cosine": 0.95,
                "clean_fraction": 0.20,
            },
            {
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "source_group": "ordinary_real_control",
                "raw_norm": 0.15,
                "clean_norm": 0.05,
                "artifact_fraction": 0.80,
                "artifact_cosine": 0.90,
                "clean_fraction": 0.33,
            },
        ]
    ).to_csv(raw, index=False)

    paths = summarize_metric_ablation(raw, tmp_path / "experiment", label="stage15c")

    source_summary = pd.read_csv(paths["source_metric_summary"])
    ordinary = source_summary[source_summary["source_group"] == "ordinary_real_control"].iloc[0]
    assert round(ordinary["projected_artifact_norm"], 6) == 0.12

    gaps = pd.read_csv(paths["metric_gap_summary"])
    raw_gap = gaps[gaps["metric_name"] == "raw_norm"].iloc[0]
    clean_gap = gaps[gaps["metric_name"] == "clean_norm"].iloc[0]
    artifact_gap = gaps[gaps["metric_name"] == "projected_artifact_norm"].iloc[0]
    assert round(raw_gap["real_domain_gap"], 6) == 0.05
    assert round(clean_gap["real_domain_gap"], 6) == 0.03
    assert round(artifact_gap["real_domain_gap"], 6) == 0.03


def test_summarize_metric_ablation_preserves_label(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        [
            {
                "class_name": "cat",
                "conditioning_key": "class",
                "source_group": "base_generated_control",
                "raw_norm": 0.20,
                "clean_norm": 0.04,
                "artifact_fraction": 0.95,
                "artifact_cosine": 0.97,
                "clean_fraction": 0.20,
            }
        ]
    ).to_csv(raw, index=False)

    paths = summarize_metric_ablation(raw, tmp_path / "experiment", label="class_only")

    summary = pd.read_csv(paths["source_metric_summary"])
    assert summary["experiment_label"].tolist() == ["class_only"]
    assert paths["source_metric_summary"].name == "source_metric_summary_class_only.csv"


def test_combine_metric_ablation_gaps_writes_mean_comparison(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        [
            {
                "experiment_label": "first",
                "metric_name": "raw_norm",
                "class_name": "dog",
                "conditioning_key": "class",
                "real_domain_gap": 0.10,
                "subject_specific_gap": -0.02,
            },
            {
                "experiment_label": "first",
                "metric_name": "raw_norm",
                "class_name": "cat",
                "conditioning_key": "class",
                "real_domain_gap": 0.20,
                "subject_specific_gap": -0.04,
            },
        ]
    ).to_csv(first, index=False)
    pd.DataFrame(
        [
            {
                "experiment_label": "second",
                "metric_name": "clean_norm",
                "class_name": "dog",
                "conditioning_key": "prompt_matched",
                "real_domain_gap": 0.03,
                "subject_specific_gap": "",
            }
        ]
    ).to_csv(second, index=False)

    paths = combine_metric_ablation_gaps([first, second], tmp_path / "combined")

    combined = pd.read_csv(paths["combined_gap_summary"])
    means = pd.read_csv(paths["mean_comparison"])
    assert len(combined) == 3
    first_mean = means[
        (means["experiment_label"] == "first") & (means["metric_name"] == "raw_norm")
    ].iloc[0]
    assert round(first_mean["mean_real_domain_gap"], 6) == 0.15
    assert round(first_mean["mean_subject_specific_gap"], 6) == -0.03
