from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


SOURCE_COLUMNS = [
    "base_generated_control",
    "ordinary_real_control",
    "dreambooth_reference",
    "natural_hard_reference",
]


def _ensure_source_columns(frame):
    for column in SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def summarize_source_decomp(raw_metrics_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    raw = read_csv_preserve_strings(raw_metrics_path)
    if "reference_regime" not in raw.columns:
        raw["reference_regime"] = "unknown"
    if "hardness_axis" not in raw.columns:
        raw["hardness_axis"] = "none"
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    source_group_summary = (
        raw.groupby(["class_name", "conditioning_key", "source_group"], as_index=False)
        .agg(
            mean_clean_norm=("clean_norm", "mean"),
            mean_raw_norm=("raw_norm", "mean"),
            mean_artifact_fraction=("artifact_fraction", "mean"),
            mean_artifact_cosine=("artifact_cosine", "mean"),
            n=("clean_norm", "size"),
        )
    )
    source_group_path = summaries_dir / "source_group_summary.csv"
    source_group_summary.to_csv(source_group_path, index=False)

    regime_summary = (
        raw.groupby(
            ["class_name", "conditioning_key", "source_group", "reference_regime", "hardness_axis"],
            as_index=False,
        )
        .agg(
            mean_clean_norm=("clean_norm", "mean"),
            mean_raw_norm=("raw_norm", "mean"),
            mean_artifact_fraction=("artifact_fraction", "mean"),
            mean_artifact_cosine=("artifact_cosine", "mean"),
            n=("clean_norm", "size"),
        )
    )
    regime_path = summaries_dir / "source_regime_summary.csv"
    regime_summary.to_csv(regime_path, index=False)

    wide = source_group_summary.pivot_table(
        index=["class_name", "conditioning_key"],
        columns="source_group",
        values="mean_clean_norm",
        aggfunc="mean",
    ).reset_index()
    wide = _ensure_source_columns(wide)
    wide["real_domain_gap"] = wide["ordinary_real_control"] - wide["base_generated_control"]
    wide["subject_specific_gap"] = wide["dreambooth_reference"] - wide["ordinary_real_control"]
    wide["natural_hard_gap"] = wide["natural_hard_reference"] - wide["dreambooth_reference"]
    gap_path = summaries_dir / "source_gap_summary.csv"
    wide.to_csv(gap_path, index=False)

    timestep_frequency_summary = (
        raw.groupby(["source_group", "conditioning_key", "timestep"], as_index=False)
        .agg(
            mean_clean_norm=("clean_norm", "mean"),
            mean_raw_norm=("raw_norm", "mean"),
            mean_artifact_fraction=("artifact_fraction", "mean"),
            mean_dct_clean_low=("dct_clean_low", "mean"),
            mean_dct_clean_mid=("dct_clean_mid", "mean"),
            mean_dct_clean_high=("dct_clean_high", "mean"),
            mean_dct_artifact_low=("dct_artifact_low", "mean"),
            mean_dct_artifact_mid=("dct_artifact_mid", "mean"),
            mean_dct_artifact_high=("dct_artifact_high", "mean"),
            n=("clean_norm", "size"),
        )
    )
    timestep_path = summaries_dir / "timestep_frequency_summary.csv"
    timestep_frequency_summary.to_csv(timestep_path, index=False)

    return {
        "source_group_summary": source_group_path,
        "regime_summary": regime_path,
        "source_gap_summary": gap_path,
        "timestep_frequency_summary": timestep_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = summarize_source_decomp(args.raw_metrics, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
