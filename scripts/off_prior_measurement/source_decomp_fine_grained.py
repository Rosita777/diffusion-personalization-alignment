from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


SOURCE_COLUMNS = [
    "base_generated_control",
    "ordinary_real_control",
    "dreambooth_reference",
    "natural_hard_reference",
]

FREQUENCY_COLUMNS = {
    "low": "dct_clean_low",
    "mid": "dct_clean_mid",
    "high": "dct_clean_high",
}


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _ensure_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _add_gap_columns(wide: pd.DataFrame) -> pd.DataFrame:
    wide = _ensure_source_columns(wide)
    wide["real_domain_gap"] = wide["ordinary_real_control"] - wide["base_generated_control"]
    wide["subject_specific_gap"] = wide["dreambooth_reference"] - wide["ordinary_real_control"]
    wide["natural_hard_gap"] = wide["natural_hard_reference"] - wide["dreambooth_reference"]
    return wide


def _frequency_long(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for band, column in FREQUENCY_COLUMNS.items():
        frame = raw[
            [
                "experiment_label",
                "class_name",
                "conditioning_key",
                "source_group",
                "timestep",
                column,
            ]
        ].copy()
        frame["frequency_band"] = band
        frame["frequency_value"] = frame[column]
        frames.append(frame.drop(columns=[column]))
    return pd.concat(frames, ignore_index=True)


def summarize_fine_grained(
    raw_metrics_path: str | Path,
    output_dir: str | Path,
    label: str,
) -> dict[str, Path]:
    raw = read_csv_preserve_strings(raw_metrics_path)
    raw["experiment_label"] = str(label)
    raw = _coerce_numeric(
        raw,
        [
            "timestep",
            "clean_norm",
            "raw_norm",
            "artifact_fraction",
            "dct_clean_low",
            "dct_clean_mid",
            "dct_clean_high",
        ],
    )

    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    source_timestep_summary = (
        raw.groupby(["experiment_label", "class_name", "conditioning_key", "source_group", "timestep"], as_index=False)
        .agg(
            mean_clean_norm=("clean_norm", "mean"),
            mean_raw_norm=("raw_norm", "mean"),
            mean_artifact_fraction=("artifact_fraction", "mean"),
            mean_dct_clean_low=("dct_clean_low", "mean"),
            mean_dct_clean_mid=("dct_clean_mid", "mean"),
            mean_dct_clean_high=("dct_clean_high", "mean"),
            n=("clean_norm", "size"),
        )
    )
    source_timestep_path = summaries_dir / f"source_timestep_summary_{label}.csv"
    source_timestep_summary.to_csv(source_timestep_path, index=False)

    timestep_wide = source_timestep_summary.pivot_table(
        index=["experiment_label", "class_name", "conditioning_key", "timestep"],
        columns="source_group",
        values="mean_clean_norm",
        aggfunc="mean",
    ).reset_index()
    timestep_gap = _add_gap_columns(timestep_wide)
    timestep_gap_path = summaries_dir / f"gap_by_timestep_{label}.csv"
    timestep_gap.to_csv(timestep_gap_path, index=False)

    frequency_source_summary = (
        _frequency_long(raw)
        .groupby(
            [
                "experiment_label",
                "class_name",
                "conditioning_key",
                "timestep",
                "frequency_band",
                "source_group",
            ],
            as_index=False,
        )
        .agg(mean_frequency_value=("frequency_value", "mean"), n=("frequency_value", "size"))
    )
    frequency_wide = frequency_source_summary.pivot_table(
        index=["experiment_label", "class_name", "conditioning_key", "timestep", "frequency_band"],
        columns="source_group",
        values="mean_frequency_value",
        aggfunc="mean",
    ).reset_index()
    frequency_gap = _add_gap_columns(frequency_wide)
    frequency_gap_path = summaries_dir / f"frequency_gap_summary_{label}.csv"
    frequency_gap.to_csv(frequency_gap_path, index=False)

    candidates = frequency_gap.sort_values(
        by=["subject_specific_gap", "real_domain_gap"],
        ascending=[False, False],
        na_position="last",
    )
    candidates_path = summaries_dir / f"signal_candidates_{label}.csv"
    candidates.to_csv(candidates_path, index=False)

    return {
        "source_timestep_summary": source_timestep_path,
        "gap_by_timestep": timestep_gap_path,
        "frequency_gap_summary": frequency_gap_path,
        "signal_candidates": candidates_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    paths = summarize_fine_grained(args.raw_metrics, args.output_dir, args.label)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
