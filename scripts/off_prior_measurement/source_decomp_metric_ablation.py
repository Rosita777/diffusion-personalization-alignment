from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


METRIC_COLUMNS = [
    "raw_norm",
    "projected_artifact_norm",
    "clean_norm",
    "artifact_fraction",
    "artifact_cosine",
    "clean_fraction",
]

SOURCE_COLUMNS = [
    "base_generated_control",
    "ordinary_real_control",
    "dreambooth_reference",
    "natural_hard_reference",
]


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


def _add_metric_columns(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _coerce_numeric(
        raw,
        [
            "raw_norm",
            "clean_norm",
            "artifact_fraction",
            "artifact_cosine",
            "clean_fraction",
        ],
    )
    raw["projected_artifact_norm"] = raw["raw_norm"] * raw["artifact_fraction"]
    return raw


def _metric_gap_summary(source_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metric_name in ["raw_norm", "projected_artifact_norm", "clean_norm"]:
        wide = source_summary.pivot_table(
            index=["experiment_label", "class_name", "conditioning_key"],
            columns="source_group",
            values=metric_name,
            aggfunc="mean",
        ).reset_index()
        wide = _ensure_source_columns(wide)
        wide.insert(1, "metric_name", metric_name)
        wide["real_domain_gap"] = wide["ordinary_real_control"] - wide["base_generated_control"]
        wide["subject_specific_gap"] = wide["dreambooth_reference"] - wide["ordinary_real_control"]
        wide["natural_hard_gap"] = wide["natural_hard_reference"] - wide["dreambooth_reference"]
        rows.append(wide)
    return pd.concat(rows, ignore_index=True)


def summarize_metric_ablation(
    raw_metrics_path: str | Path,
    output_dir: str | Path,
    label: str,
) -> dict[str, Path]:
    raw = _add_metric_columns(read_csv_preserve_strings(raw_metrics_path))
    raw["experiment_label"] = str(label)
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    source_summary = (
        raw.groupby(["experiment_label", "class_name", "conditioning_key", "source_group"], as_index=False)
        .agg(
            raw_norm=("raw_norm", "mean"),
            projected_artifact_norm=("projected_artifact_norm", "mean"),
            clean_norm=("clean_norm", "mean"),
            artifact_fraction=("artifact_fraction", "mean"),
            artifact_cosine=("artifact_cosine", "mean"),
            clean_fraction=("clean_fraction", "mean"),
            n=("raw_norm", "size"),
        )
    )
    source_path = summaries_dir / f"source_metric_summary_{label}.csv"
    source_summary.to_csv(source_path, index=False)

    gap_summary = _metric_gap_summary(source_summary)
    gap_path = summaries_dir / f"metric_gap_summary_{label}.csv"
    gap_summary.to_csv(gap_path, index=False)

    return {
        "source_metric_summary": source_path,
        "metric_gap_summary": gap_path,
    }


def combine_metric_ablation_gaps(
    gap_summary_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([read_csv_preserve_strings(path) for path in gap_summary_paths], ignore_index=True)
    for column in ["real_domain_gap", "subject_specific_gap", "natural_hard_gap"]:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = _coerce_numeric(
        combined,
        [
            "real_domain_gap",
            "subject_specific_gap",
            "natural_hard_gap",
        ],
    )
    combined_path = summaries_dir / "metric_gap_summary_combined.csv"
    combined.to_csv(combined_path, index=False)

    mean_comparison = (
        combined.groupby(["experiment_label", "metric_name"], as_index=False)
        .agg(
            mean_real_domain_gap=("real_domain_gap", "mean"),
            mean_subject_specific_gap=("subject_specific_gap", "mean"),
            mean_natural_hard_gap=("natural_hard_gap", "mean"),
        )
    )
    mean_path = summaries_dir / "metric_gap_mean_comparison.csv"
    mean_comparison.to_csv(mean_path, index=False)
    return {
        "combined_gap_summary": combined_path,
        "mean_comparison": mean_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label")
    parser.add_argument("--combine-gap-summaries", nargs="+")
    args = parser.parse_args()
    if args.combine_gap_summaries:
        paths = combine_metric_ablation_gaps(
            gap_summary_paths=args.combine_gap_summaries,
            output_dir=args.output_dir,
        )
    else:
        if not args.raw_metrics or not args.label:
            raise ValueError("Either provide --combine-gap-summaries, or provide both --raw-metrics and --label")
        paths = summarize_metric_ablation(
            raw_metrics_path=args.raw_metrics,
            output_dir=args.output_dir,
            label=args.label,
        )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
