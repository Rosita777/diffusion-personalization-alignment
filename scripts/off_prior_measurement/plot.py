from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _group_labels(groups: list[str]) -> list[str]:
    return [group.replace("_", "\n") for group in groups]


def create_figures(scored_metrics_path: str | Path, figures_dir: str | Path) -> dict[str, Path]:
    scored = read_csv_preserve_strings(scored_metrics_path)
    if "hardness_axis" not in scored.columns:
        scored["hardness_axis"] = "none"
    scored["hardness_axis"] = scored["hardness_axis"].replace("", "none").fillna("none")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    control_distribution = figures_dir / "control_distribution.png"
    groups = list(scored["source_group"].drop_duplicates())
    data = [scored[scored["source_group"] == group]["normalized_l2"].dropna().to_numpy() for group in groups]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=_group_labels(groups))
    plt.ylabel("normalized_l2")
    plt.tight_layout()
    plt.savefig(control_distribution, dpi=200)
    plt.close()

    timestep_curves = figures_dir / "timestep_curves.png"
    plt.figure(figsize=(8, 4))
    for group in groups:
        subset = scored[scored["source_group"] == group]
        curve = subset.groupby("timestep", as_index=False)["floor_adjusted_l2"].mean()
        plt.plot(curve["timestep"], curve["floor_adjusted_l2"], marker="o", label=group)
    plt.xlabel("timestep")
    plt.ylabel("mean floor_adjusted_l2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(timestep_curves, dpi=200)
    plt.close()

    ladder_timestep_heatmap = figures_dir / "ladder_timestep_heatmap.png"
    ladder_data = scored.pivot_table(
        index="reference_regime",
        columns="timestep",
        values="floor_adjusted_l2",
        aggfunc="mean",
    ).fillna(0.0)
    plt.figure(figsize=(8, 4))
    plt.imshow(ladder_data.to_numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="mean floor_adjusted_l2")
    plt.xticks(np.arange(len(ladder_data.columns)), ladder_data.columns)
    plt.yticks(np.arange(len(ladder_data.index)), ladder_data.index)
    plt.xlabel("timestep")
    plt.ylabel("reference_regime")
    plt.tight_layout()
    plt.savefig(ladder_timestep_heatmap, dpi=200)
    plt.close()

    reference = scored[scored["source_group"] == "dreambooth_reference"]
    if reference.empty:
        reference = scored
    heatmap_data = (
        reference.groupby("timestep")[["dct_delta_low", "dct_delta_mid", "dct_delta_high"]]
        .mean()
        .rename(columns={"dct_delta_low": "low", "dct_delta_mid": "mid", "dct_delta_high": "high"})
    )
    frequency_heatmap = figures_dir / "frequency_heatmap.png"
    plt.figure(figsize=(6, 4))
    values = heatmap_data.to_numpy()
    plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(label="mean DCT residual energy")
    plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
    plt.xlabel("frequency band")
    plt.ylabel("timestep")
    plt.tight_layout()
    plt.savefig(frequency_heatmap, dpi=200)
    plt.close()

    hardness_frequency_heatmap = figures_dir / "hardness_frequency_heatmap.png"
    hardness_data = (
        scored.groupby("hardness_axis")[["dct_delta_low", "dct_delta_mid", "dct_delta_high"]]
        .mean()
        .rename(columns={"dct_delta_low": "low", "dct_delta_mid": "mid", "dct_delta_high": "high"})
    )
    plt.figure(figsize=(7, 4))
    plt.imshow(hardness_data.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="mean DCT residual energy")
    plt.xticks(np.arange(len(hardness_data.columns)), hardness_data.columns)
    plt.yticks(np.arange(len(hardness_data.index)), hardness_data.index)
    plt.xlabel("frequency band")
    plt.ylabel("hardness_axis")
    plt.tight_layout()
    plt.savefig(hardness_frequency_heatmap, dpi=200)
    plt.close()

    return {
        "control_distribution": control_distribution,
        "timestep_curves": timestep_curves,
        "frequency_heatmap": frequency_heatmap,
        "ladder_timestep_heatmap": ladder_timestep_heatmap,
        "hardness_frequency_heatmap": hardness_frequency_heatmap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-metrics", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()
    paths = create_figures(args.scored_metrics, args.figures_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
