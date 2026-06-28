import pandas as pd

from scripts.off_prior_measurement.source_decomp_fine_grained import summarize_fine_grained


def test_summarize_fine_grained_computes_timestep_gaps(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = []
    for timestep, values in [
        (
            50,
            {
                "base_generated_control": 0.10,
                "ordinary_real_control": 0.20,
                "dreambooth_reference": 0.35,
            },
        ),
        (
            200,
            {
                "base_generated_control": 0.20,
                "ordinary_real_control": 0.25,
                "dreambooth_reference": 0.22,
            },
        ),
    ]:
        for source_group, clean_norm in values.items():
            rows.append(
                {
                    "class_name": "dog",
                    "conditioning_key": "class",
                    "source_group": source_group,
                    "timestep": timestep,
                    "clean_norm": clean_norm,
                    "raw_norm": clean_norm + 0.10,
                    "artifact_fraction": 0.80,
                    "dct_clean_low": clean_norm * 0.50,
                    "dct_clean_mid": clean_norm * 0.25,
                    "dct_clean_high": clean_norm * 0.10,
                }
            )
    pd.DataFrame(rows).to_csv(raw, index=False)

    paths = summarize_fine_grained(raw, tmp_path / "experiment", label="stage15d")

    gaps = pd.read_csv(paths["gap_by_timestep"])
    early = gaps[gaps["timestep"] == 50].iloc[0]
    late = gaps[gaps["timestep"] == 200].iloc[0]
    assert round(early["real_domain_gap"], 6) == 0.10
    assert round(early["subject_specific_gap"], 6) == 0.15
    assert round(late["subject_specific_gap"], 6) == -0.03
    assert paths["source_timestep_summary"].name == "source_timestep_summary_stage15d.csv"


def test_summarize_fine_grained_ranks_frequency_signal_candidates(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        [
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": "ordinary_real_control",
                "timestep": 50,
                "clean_norm": 0.20,
                "raw_norm": 0.30,
                "artifact_fraction": 0.80,
                "dct_clean_low": 0.10,
                "dct_clean_mid": 0.20,
                "dct_clean_high": 0.30,
            },
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": "dreambooth_reference",
                "timestep": 50,
                "clean_norm": 0.25,
                "raw_norm": 0.35,
                "artifact_fraction": 0.80,
                "dct_clean_low": 0.40,
                "dct_clean_mid": 0.21,
                "dct_clean_high": 0.10,
            },
            {
                "class_name": "dog",
                "conditioning_key": "class",
                "source_group": "base_generated_control",
                "timestep": 50,
                "clean_norm": 0.10,
                "raw_norm": 0.20,
                "artifact_fraction": 0.80,
                "dct_clean_low": 0.05,
                "dct_clean_mid": 0.10,
                "dct_clean_high": 0.20,
            },
        ]
    ).to_csv(raw, index=False)

    paths = summarize_fine_grained(raw, tmp_path / "experiment", label="stage15d")

    frequency = pd.read_csv(paths["frequency_gap_summary"])
    low = frequency[frequency["frequency_band"] == "low"].iloc[0]
    high = frequency[frequency["frequency_band"] == "high"].iloc[0]
    assert round(low["subject_specific_gap"], 6) == 0.30
    assert round(high["subject_specific_gap"], 6) == -0.20

    candidates = pd.read_csv(paths["signal_candidates"])
    first = candidates.iloc[0]
    assert first["frequency_band"] == "low"
    assert round(first["subject_specific_gap"], 6) == 0.30
