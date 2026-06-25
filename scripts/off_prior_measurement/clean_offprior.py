from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings

PAIR_KEYS = ["subject_id", "conditioning_key", "timestep", "noise_seed"]


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype=str)


def _standard_image_key(frame: pd.DataFrame) -> pd.Series:
    source = _series_or_empty(frame, "source_standard_image")
    path = _series_or_empty(frame, "image_path")
    image = _series_or_empty(frame, "image_id")
    return source.where(source != "", path.where(path != "", image))


def _roundtrip_baseline(scored: pd.DataFrame) -> pd.DataFrame:
    roundtrip = scored[scored["source_group"] == "vae_roundtrip_control"].copy()
    roundtrip["standard_image_key"] = _standard_image_key(roundtrip)
    return roundtrip[PAIR_KEYS + ["standard_image_key", "floor_adjusted_l2"]].rename(
        columns={"floor_adjusted_l2": "roundtrip_baseline_l2"}
    )


def compute_clean_scores(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    for column in ["source_standard_image", "variant_id", "hardness_axis"]:
        if column not in scored.columns:
            scored[column] = ""
    scored["standard_image_key"] = _standard_image_key(scored)

    clean = scored.merge(_roundtrip_baseline(scored), on=PAIR_KEYS + ["standard_image_key"], how="left")
    subject_baseline = (
        scored[scored["source_group"] == "vae_roundtrip_control"]
        .groupby(PAIR_KEYS, as_index=False)["floor_adjusted_l2"]
        .median()
        .rename(columns={"floor_adjusted_l2": "subject_roundtrip_baseline_l2"})
    )
    clean = clean.merge(subject_baseline, on=PAIR_KEYS, how="left")
    clean["roundtrip_baseline_l2"] = clean["roundtrip_baseline_l2"].fillna(
        clean["subject_roundtrip_baseline_l2"]
    )
    clean["clean_pair_l2"] = clean["floor_adjusted_l2"] - clean["roundtrip_baseline_l2"]
    clean["clean_subject_l2"] = clean["floor_adjusted_l2"] - clean["subject_roundtrip_baseline_l2"]
    clean["roundtrip_ratio"] = clean["roundtrip_baseline_l2"].abs() / (
        clean["floor_adjusted_l2"].abs() + 1e-8
    )
    return clean


def _write_summaries(clean: pd.DataFrame, summaries_dir: Path) -> dict[str, Path]:
    regime_summary = (
        clean.groupby(
            ["source_group", "reference_regime", "hardness_axis", "conditioning_key", "timestep"],
            as_index=False,
        )
        .agg(
            mean_raw_l2=("floor_adjusted_l2", "mean"),
            mean_clean_pair_l2=("clean_pair_l2", "mean"),
            mean_clean_subject_l2=("clean_subject_l2", "mean"),
            mean_roundtrip_ratio=("roundtrip_ratio", "mean"),
            n=("floor_adjusted_l2", "size"),
        )
    )
    subject_summary = (
        clean.groupby(["subject_id", "source_group", "reference_regime", "conditioning_key"], as_index=False)
        .agg(
            mean_raw_l2=("floor_adjusted_l2", "mean"),
            mean_clean_pair_l2=("clean_pair_l2", "mean"),
            mean_clean_subject_l2=("clean_subject_l2", "mean"),
            mean_roundtrip_ratio=("roundtrip_ratio", "mean"),
            n=("floor_adjusted_l2", "size"),
        )
    )

    ladder_base = (
        clean.groupby(["subject_id", "conditioning_key", "reference_regime"], as_index=False)
        .agg(
            mean_raw_l2=("floor_adjusted_l2", "mean"),
            mean_clean_pair_l2=("clean_pair_l2", "mean"),
            mean_roundtrip_ratio=("roundtrip_ratio", "mean"),
        )
    )
    raw_wide = ladder_base.pivot_table(
        index=["subject_id", "conditioning_key"],
        columns="reference_regime",
        values="mean_raw_l2",
        aggfunc="mean",
    ).add_prefix("raw_")
    clean_wide = ladder_base.pivot_table(
        index=["subject_id", "conditioning_key"],
        columns="reference_regime",
        values="mean_clean_pair_l2",
        aggfunc="mean",
    ).add_prefix("clean_")
    ratio_wide = ladder_base.pivot_table(
        index=["subject_id", "conditioning_key"],
        columns="reference_regime",
        values="mean_roundtrip_ratio",
        aggfunc="mean",
    ).add_prefix("ratio_")
    ladder = pd.concat([raw_wide, clean_wide, ratio_wide], axis=1).reset_index()
    for column in [
        "raw_easy_control",
        "raw_standard_reference",
        "raw_hard_reference",
        "raw_roundtrip_control",
        "clean_standard_reference",
        "clean_hard_reference",
        "ratio_standard_reference",
    ]:
        if column not in ladder.columns:
            ladder[column] = np.nan
    ladder["raw_standard_minus_easy"] = ladder["raw_standard_reference"] - ladder["raw_easy_control"]
    ladder["raw_hard_minus_standard"] = ladder["raw_hard_reference"] - ladder["raw_standard_reference"]
    ladder["clean_hard_minus_standard"] = ladder["clean_hard_reference"] - ladder["clean_standard_reference"]

    paths = {
        "clean_regime_summary": summaries_dir / "clean_regime_summary.csv",
        "clean_subject_summary": summaries_dir / "clean_subject_summary.csv",
        "clean_ladder_summary": summaries_dir / "clean_ladder_summary.csv",
    }
    regime_summary.to_csv(paths["clean_regime_summary"], index=False)
    subject_summary.to_csv(paths["clean_subject_summary"], index=False)
    ladder.to_csv(paths["clean_ladder_summary"], index=False)
    return paths


def compute_clean_offprior(
    scored_metrics_path: str | Path,
    output_dir: str | Path,
    source_experiment: str,
) -> dict[str, Path]:
    scored = read_csv_preserve_strings(scored_metrics_path)
    output_dir = Path(output_dir)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    clean = compute_clean_scores(scored)
    clean_path = summaries_dir / "clean_scored_metrics.csv"
    clean.to_csv(clean_path, index=False)

    paths = {"clean_scored_metrics": clean_path}
    paths.update(_write_summaries(clean, summaries_dir))
    config_path = output_dir / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "source_experiment": source_experiment,
                "scored_metrics": str(scored_metrics_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["config_resolved"] = config_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-experiment", required=True)
    args = parser.parse_args()
    paths = compute_clean_offprior(args.scored_metrics, args.output_dir, args.source_experiment)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
