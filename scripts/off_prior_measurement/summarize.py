from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def summarize_metrics(raw_metrics_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    raw = read_csv_preserve_strings(raw_metrics_path)
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    floor_keys = ["subject_id", "conditioning_key", "timestep"]
    base_floor = (
        raw[raw["source_group"] == "base_easy_control"]
        .groupby(floor_keys, as_index=False)["normalized_l2"]
        .median()
        .rename(columns={"normalized_l2": "base_floor_l2"})
    )
    scored = raw.merge(base_floor, on=floor_keys, how="left")
    scored["floor_adjusted_l2"] = scored["normalized_l2"] - scored["base_floor_l2"]
    scored_path = summaries_dir / "scored_metrics.csv"
    scored.to_csv(scored_path, index=False)

    base_floor_path = summaries_dir / "base_floor.csv"
    base_floor.to_csv(base_floor_path, index=False)

    regime_summary = (
        scored.groupby(["source_group", "reference_regime", "conditioning_key", "timestep"], as_index=False)
        .agg(
            mean_normalized_l2=("normalized_l2", "mean"),
            median_normalized_l2=("normalized_l2", "median"),
            mean_floor_adjusted_l2=("floor_adjusted_l2", "mean"),
            median_floor_adjusted_l2=("floor_adjusted_l2", "median"),
            mean_cosine_distance=("cosine_distance", "mean"),
            n=("normalized_l2", "size"),
        )
    )
    regime_summary_path = summaries_dir / "regime_summary.csv"
    regime_summary.to_csv(regime_summary_path, index=False)

    subject_summary = (
        scored.groupby(["subject_id", "source_group", "conditioning_key"], as_index=False)
        .agg(
            mean_floor_adjusted_l2=("floor_adjusted_l2", "mean"),
            median_floor_adjusted_l2=("floor_adjusted_l2", "median"),
            mean_cosine_distance=("cosine_distance", "mean"),
            n=("normalized_l2", "size"),
        )
    )
    subject_summary_path = summaries_dir / "subject_summary.csv"
    subject_summary.to_csv(subject_summary_path, index=False)

    return {
        "scored_metrics": scored_path,
        "base_floor": base_floor_path,
        "regime_summary": regime_summary_path,
        "subject_summary": subject_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = summarize_metrics(args.raw_metrics, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
