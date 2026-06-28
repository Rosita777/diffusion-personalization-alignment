from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _coerce_numeric(frame, columns: list[str]):
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def create_source_decomp_figures(summary_dir: str | Path, figures_dir: str | Path) -> dict[str, Path]:
    summary_dir = Path(summary_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    gaps = read_csv_preserve_strings(summary_dir / "source_gap_summary.csv")
    source_groups = read_csv_preserve_strings(summary_dir / "source_group_summary.csv")
    timestep = read_csv_preserve_strings(summary_dir / "timestep_frequency_summary.csv")
    gaps = _coerce_numeric(
        gaps,
        [
            "real_domain_gap",
            "subject_specific_gap",
            "natural_hard_gap",
        ],
    )
    source_groups = _coerce_numeric(source_groups, ["mean_artifact_fraction"])
    timestep = _coerce_numeric(timestep, ["timestep", "mean_clean_norm"])

    source_gap_bars = figures_dir / "source_gap_bars.png"
    gap_means = gaps[["real_domain_gap", "subject_specific_gap", "natural_hard_gap"]].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(gap_means)), gap_means.to_numpy())
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(gap_means)), [item.replace("_", "\n") for item in gap_means.index])
    plt.ylabel("mean clean-norm gap")
    plt.tight_layout()
    plt.savefig(source_gap_bars, dpi=200)
    plt.close()

    artifact_fraction_by_source = figures_dir / "artifact_fraction_by_source.png"
    artifact = source_groups.groupby("source_group", as_index=False)["mean_artifact_fraction"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(artifact)), artifact["mean_artifact_fraction"])
    plt.axhline(0.75, color="red", linestyle="--", linewidth=1.0, label="0.75 gate")
    plt.xticks(np.arange(len(artifact)), [item.replace("_", "\n") for item in artifact["source_group"]])
    plt.ylabel("mean artifact fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_fraction_by_source, dpi=200)
    plt.close()

    clean_timestep_curves = figures_dir / "clean_timestep_curves.png"
    plt.figure(figsize=(8, 4))
    for source_group in timestep["source_group"].drop_duplicates():
        rows = timestep[timestep["source_group"] == source_group]
        curve = rows.groupby("timestep", as_index=False)["mean_clean_norm"].mean()
        plt.plot(curve["timestep"], curve["mean_clean_norm"], marker="o", label=source_group)
    plt.xlabel("timestep")
    plt.ylabel("mean clean norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(clean_timestep_curves, dpi=200)
    plt.close()

    return {
        "source_gap_bars": source_gap_bars,
        "artifact_fraction_by_source": artifact_fraction_by_source,
        "clean_timestep_curves": clean_timestep_curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()
    paths = create_source_decomp_figures(args.summary_dir, args.figures_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
