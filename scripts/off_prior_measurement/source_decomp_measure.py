from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings
from scripts.off_prior_measurement.diffusion_backend import StableDiffusionBackend
from scripts.off_prior_measurement.measure import shard_manifest
from scripts.off_prior_measurement.metrics import dct_band_energy, residual_projection_metrics


def _scaled_residual(v_ref, v_base, coeff: float):
    return coeff * (v_ref - v_base)


def run_source_decomp_measurement(
    manifest_path: str | Path,
    output_dir: str | Path,
    timesteps: list[int],
    noise_seeds: list[int],
    backend,
    rank: int = 0,
    world_size: int = 1,
    output_name: str = "raw_source_decomp_metrics.csv",
) -> Path:
    manifest = read_csv_preserve_strings(manifest_path)
    manifest = shard_manifest(manifest, rank=rank, world_size=world_size)
    rows: list[dict[str, object]] = []
    for row in tqdm(list(manifest.to_dict("records")), desc="source decomposition"):
        for timestep in timesteps:
            for seed in noise_seeds:
                batch = backend.measure(row["image_path"], row["conditioning_prompt"], int(timestep), int(seed))
                rt_batch = backend.measure(
                    row["roundtrip_image_path"],
                    row["conditioning_prompt"],
                    int(timestep),
                    int(seed),
                )
                metrics = residual_projection_metrics(
                    batch.v_ref,
                    batch.v_base,
                    rt_batch.v_ref,
                    rt_batch.v_base,
                )
                artifact_residual = _scaled_residual(rt_batch.v_ref, rt_batch.v_base, metrics["artifact_coeff"])
                clean_residual = (batch.v_ref - batch.v_base) - artifact_residual
                clean_bands = dct_band_energy(clean_residual)
                artifact_bands = dct_band_energy(artifact_residual)
                rows.append(
                    {
                        **row,
                        "timestep": int(timestep),
                        "noise_seed": int(seed),
                        "snr": float(getattr(batch, "snr", float("nan"))),
                        **metrics,
                        "dct_clean_low": clean_bands["low"],
                        "dct_clean_mid": clean_bands["mid"],
                        "dct_clean_high": clean_bands["high"],
                        "dct_artifact_low": artifact_bands["low"],
                        "dct_artifact_mid": artifact_bands["mid"],
                        "dct_artifact_high": artifact_bands["high"],
                    }
                )
    output_dir = Path(output_dir)
    measurements_dir = output_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    path = measurements_dir / output_name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def merge_source_decomp_shards(
    output_dir: str | Path,
    world_size: int,
    output_name: str = "raw_source_decomp_metrics.csv",
) -> Path:
    output_dir = Path(output_dir)
    measurements_dir = output_dir / "measurements"
    shard_paths = [
        measurements_dir / f"raw_source_decomp_metrics_rank{rank}.csv" for rank in range(world_size)
    ]
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source decomposition shard files: {missing}")
    merged = pd.concat([read_csv_preserve_strings(path) for path in shard_paths], ignore_index=True)
    merged_path = measurements_dir / output_name
    merged.to_csv(merged_path, index=False)
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-name", default="raw_source_decomp_metrics.csv")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.merge_shards:
        path = merge_source_decomp_shards(
            config.output_dir,
            world_size=args.world_size,
            output_name=args.output_name,
        )
        print(path)
        return
    backend = StableDiffusionBackend(config)
    path = run_source_decomp_measurement(
        manifest_path=args.manifest,
        output_dir=config.output_dir,
        timesteps=config.timesteps,
        noise_seeds=config.noise_seeds,
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
        output_name=args.output_name,
    )
    print(path)


if __name__ == "__main__":
    main()
