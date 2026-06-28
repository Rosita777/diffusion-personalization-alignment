import pandas as pd

from scripts.off_prior_measurement.source_decomp_summarize import summarize_source_decomp


def test_summarize_source_decomp_computes_gap_columns(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = []
    for source, clean in [
        ("base_generated_control", 0.10),
        ("ordinary_real_control", 0.20),
        ("dreambooth_reference", 0.35),
        ("natural_hard_reference", 0.45),
    ]:
        rows.append(
            {
                "subject_id": "dog",
                "class_name": "dog",
                "source_group": source,
                "conditioning_key": "class",
                "timestep": 50,
                "clean_norm": clean,
                "raw_norm": clean + 0.05,
                "artifact_fraction": 0.20,
                "artifact_cosine": 0.30,
                "dct_clean_low": clean,
                "dct_clean_mid": clean / 2,
                "dct_clean_high": clean / 4,
                "dct_artifact_low": 0.01,
                "dct_artifact_mid": 0.02,
                "dct_artifact_high": 0.03,
            }
        )
    pd.DataFrame(rows).to_csv(raw, index=False)

    paths = summarize_source_decomp(raw, tmp_path / "experiment")

    gaps = pd.read_csv(paths["source_gap_summary"])
    row = gaps.iloc[0]
    assert round(row["real_domain_gap"], 6) == 0.10
    assert round(row["subject_specific_gap"], 6) == 0.15
    assert round(row["natural_hard_gap"], 6) == 0.10
    assert paths["source_group_summary"].exists()
    assert paths["timestep_frequency_summary"].exists()


def test_summarize_source_decomp_writes_regime_summary(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "dog",
                "class_name": "dog",
                "source_group": "ordinary_real_control",
                "reference_regime": "ordinary_matched_context",
                "hardness_axis": "matched_context",
                "conditioning_key": "class",
                "timestep": 50,
                "clean_norm": 0.20,
                "raw_norm": 0.30,
                "artifact_fraction": 0.40,
                "artifact_cosine": 0.50,
                "dct_clean_low": 0.10,
                "dct_clean_mid": 0.05,
                "dct_clean_high": 0.02,
                "dct_artifact_low": 0.01,
                "dct_artifact_mid": 0.02,
                "dct_artifact_high": 0.03,
            },
            {
                "subject_id": "dog",
                "class_name": "dog",
                "source_group": "ordinary_real_control",
                "reference_regime": "ordinary_cluttered_scene",
                "hardness_axis": "cluttered_scene",
                "conditioning_key": "class",
                "timestep": 50,
                "clean_norm": 0.60,
                "raw_norm": 0.70,
                "artifact_fraction": 0.80,
                "artifact_cosine": 0.90,
                "dct_clean_low": 0.10,
                "dct_clean_mid": 0.05,
                "dct_clean_high": 0.02,
                "dct_artifact_low": 0.01,
                "dct_artifact_mid": 0.02,
                "dct_artifact_high": 0.03,
            },
        ]
    ).to_csv(raw, index=False)

    paths = summarize_source_decomp(raw, tmp_path / "experiment")

    regime = pd.read_csv(paths["regime_summary"])
    assert regime["reference_regime"].tolist() == [
        "ordinary_cluttered_scene",
        "ordinary_matched_context",
    ]
    assert regime["mean_clean_norm"].tolist() == [0.60, 0.20]
