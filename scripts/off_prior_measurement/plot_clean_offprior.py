from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def _label(value: str) -> str:
    return value.replace("_", "\n")


def create_clean_figures(clean_scored_metrics_path: str | Path, figures_dir: str | Path) -> dict[str, Path]:
    clean = read_csv_preserve_strings(clean_scored_metrics_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_vs_clean_ladder = figures_dir / "raw_vs_clean_ladder.png"
    ladder = (
        clean.groupby("reference_regime", as_index=False)
        .agg(raw=("floor_adjusted_l2", "mean"), clean=("clean_pair_l2", "mean"))
        .sort_values("reference_regime")
    )
    x = np.arange(len(ladder))
    width = 0.36
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, ladder["raw"], width=width, label="raw")
    plt.bar(x + width / 2, ladder["clean"], width=width, label="clean")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, [_label(regime) for regime in ladder["reference_regime"]])
    plt.ylabel("mean floor-adjusted L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(raw_vs_clean_ladder, dpi=200)
    plt.close()

    roundtrip_attribution_by_subject = figures_dir / "roundtrip_attribution_by_subject.png"
    reference = clean[clean["source_group"] == "dreambooth_reference"]
    if reference.empty:
        reference = clean
    ratio = reference.groupby("subject_id", as_index=False)["roundtrip_ratio"].mean().sort_values("subject_id")
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(ratio)), ratio["roundtrip_ratio"])
    plt.axhline(0.75, color="red", linestyle="--", linewidth=1.0, label="0.75 gate")
    plt.xticks(np.arange(len(ratio)), ratio["subject_id"], rotation=30, ha="right")
    plt.ylabel("mean roundtrip ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roundtrip_attribution_by_subject, dpi=200)
    plt.close()

    clean_timestep_curves = figures_dir / "clean_timestep_curves.png"
    plt.figure(figsize=(8, 4))
    for regime in clean["reference_regime"].drop_duplicates():
        subset = clean[clean["reference_regime"] == regime]
        curve = subset.groupby("timestep", as_index=False)["clean_pair_l2"].mean()
        plt.plot(curve["timestep"], curve["clean_pair_l2"], marker="o", label=regime)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("timestep")
    plt.ylabel("mean clean_pair_l2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(clean_timestep_curves, dpi=200)
    plt.close()

    clean_frequency_heatmap = figures_dir / "clean_frequency_heatmap.png"
    frequency = (
        clean.groupby("reference_regime")[["dct_delta_low", "dct_delta_mid", "dct_delta_high"]]
        .mean()
        .rename(columns={"dct_delta_low": "low", "dct_delta_mid": "mid", "dct_delta_high": "high"})
    )
    plt.figure(figsize=(7, 4))
    plt.imshow(frequency.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="mean DCT residual energy")
    plt.xticks(np.arange(len(frequency.columns)), frequency.columns)
    plt.yticks(np.arange(len(frequency.index)), frequency.index)
    plt.xlabel("frequency band")
    plt.ylabel("reference regime")
    plt.tight_layout()
    plt.savefig(clean_frequency_heatmap, dpi=200)
    plt.close()

    return {
        "raw_vs_clean_ladder": raw_vs_clean_ladder,
        "roundtrip_attribution_by_subject": roundtrip_attribution_by_subject,
        "clean_timestep_curves": clean_timestep_curves,
        "clean_frequency_heatmap": clean_frequency_heatmap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-scored-metrics", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()
    paths = create_clean_figures(args.clean_scored_metrics, args.figures_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
