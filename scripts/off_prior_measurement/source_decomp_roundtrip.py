from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings
from scripts.off_prior_measurement.roundtrip_controls import (
    torch_dtype,
    vae_roundtrip_image,
)


def _unique_image_pairs(manifest_path: str | Path) -> list[tuple[Path, Path]]:
    manifest = read_csv_preserve_strings(manifest_path)
    required = {"image_path", "roundtrip_image_path"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Missing source decomposition manifest columns: {sorted(missing_columns)}")

    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for row in manifest[["image_path", "roundtrip_image_path"]].drop_duplicates().to_dict("records"):
        image_path = Path(str(row["image_path"]))
        output_path = Path(str(row["roundtrip_image_path"]))
        key = (str(image_path), str(output_path))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((image_path, output_path))
    return pairs


def generate_source_decomp_roundtrips_from_manifest(
    manifest_path: str | Path,
    vae,
    resolution: int,
    device,
    dtype,
    roundtrip_fn=vae_roundtrip_image,
) -> list[Path]:
    written: list[Path] = []
    for image_path, output_path in tqdm(_unique_image_pairs(manifest_path), desc="source roundtrips"):
        if not image_path.exists():
            raise FileNotFoundError(f"Missing source decomposition image path: {image_path}")
        if output_path.exists():
            continue
        roundtrip_fn(
            vae=vae,
            image_path=image_path,
            output_path=output_path,
            resolution=resolution,
            device=device,
            dtype=dtype,
        )
        written.append(output_path)
    return written


def generate_source_decomp_roundtrips(config_path: str | Path, manifest_path: str | Path) -> list[Path]:
    import torch
    from diffusers import AutoencoderKL

    config = load_config(config_path)
    device = torch.device(config.device)
    dtype = torch_dtype(config.dtype)
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()
    return generate_source_decomp_roundtrips_from_manifest(
        manifest_path=manifest_path,
        vae=vae,
        resolution=config.resolution,
        device=device,
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    written = generate_source_decomp_roundtrips(args.config, args.manifest)
    print(f"wrote {len(written)} source decomposition roundtrip images")


if __name__ == "__main__":
    main()
