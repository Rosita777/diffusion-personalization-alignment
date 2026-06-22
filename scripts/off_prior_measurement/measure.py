from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.diffusion_backend import StableDiffusionBackend
from scripts.off_prior_measurement.metrics import dct_band_energy, residual_metrics


def _prompt_from_row(row: dict[str, object]) -> str:
    prompt = row.get("conditioning_prompt", "")
    return prompt if isinstance(prompt, str) else ""


def run_measurement(
    manifest_path: str | Path,
    output_dir: str | Path,
    timesteps: list[int],
    noise_seeds: list[int],
    backend,
) -> Path:
    manifest = pd.read_csv(manifest_path)
    output_dir = Path(output_dir)
    measurements_dir = output_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for row in tqdm(list(manifest.to_dict("records")), desc="measuring residuals"):
        for timestep in timesteps:
            for seed in noise_seeds:
                batch = backend.measure(
                    image_path=row["image_path"],
                    prompt=_prompt_from_row(row),
                    timestep=int(timestep),
                    seed=int(seed),
                )
                residual = residual_metrics(batch.v_ref, batch.v_base)
                delta_bands = dct_band_energy(batch.v_ref - batch.v_base)
                ref_bands = dct_band_energy(batch.v_ref)
                base_bands = dct_band_energy(batch.v_base)
                rows.append(
                    {
                        **row,
                        "timestep": int(timestep),
                        "noise_seed": int(seed),
                        "snr": float(getattr(batch, "snr", float("nan"))),
                        **residual,
                        "dct_delta_low": delta_bands["low"],
                        "dct_delta_mid": delta_bands["mid"],
                        "dct_delta_high": delta_bands["high"],
                        "dct_ref_low": ref_bands["low"],
                        "dct_ref_mid": ref_bands["mid"],
                        "dct_ref_high": ref_bands["high"],
                        "dct_base_low": base_bands["low"],
                        "dct_base_mid": base_bands["mid"],
                        "dct_base_high": base_bands["high"],
                    }
                )

    raw_metrics_path = measurements_dir / "raw_metrics.csv"
    pd.DataFrame(rows).to_csv(raw_metrics_path, index=False)
    return raw_metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backend = StableDiffusionBackend(config)
    path = run_measurement(
        manifest_path=args.manifest,
        output_dir=config.output_dir,
        timesteps=config.timesteps,
        noise_seeds=config.noise_seeds,
        backend=backend,
    )
    print(path)


if __name__ == "__main__":
    main()
