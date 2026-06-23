from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.config import load_config
from scripts.off_prior_measurement.csv_io import read_csv_preserve_strings


def torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _preprocess(image, resolution: int, device, dtype):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)


def vae_roundtrip_image(
    vae,
    image_path: str | Path,
    output_path: str | Path,
    resolution: int,
    device,
    dtype,
) -> None:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image

    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        pixels = _preprocess(image, resolution, device, dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1, 1) + 1) / 2
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_pil_image(decoded[0].cpu()).save(output_path)


def build_roundtrip_manifest(reference_manifest_path: str | Path, roundtrip_root: str | Path) -> pd.DataFrame:
    reference = read_csv_preserve_strings(reference_manifest_path)
    roundtrip_root = Path(roundtrip_root)
    rows = []
    for row in reference.to_dict("records"):
        new_row = dict(row)
        new_row["image_path"] = str(roundtrip_root / row["subject_id"] / f"{row['image_id']}.png")
        new_row["source_group"] = "vae_roundtrip_control"
        new_row["reference_regime"] = "roundtrip_control"
        new_row["hardness_axis"] = "none"
        new_row["source_standard_image"] = row["image_path"]
        new_row["variant_id"] = ""
        new_row["transform_parameters"] = "{}"
        rows.append(new_row)
    return pd.DataFrame(rows)


def generate_roundtrip_controls(config_path: str | Path, reference_manifest_path: str | Path) -> Path:
    import torch
    from diffusers import AutoencoderKL

    config = load_config(config_path)
    reference = read_csv_preserve_strings(reference_manifest_path)
    roundtrip_root = config.cache_dir / "vae_roundtrip_controls"
    device = torch.device(config.device)
    dtype = torch_dtype(config.dtype)
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()
    unique_images = reference[["subject_id", "image_id", "image_path"]].drop_duplicates()
    for row in unique_images.to_dict("records"):
        output_path = roundtrip_root / row["subject_id"] / f"{row['image_id']}.png"
        if output_path.exists():
            continue
        vae_roundtrip_image(
            vae=vae,
            image_path=row["image_path"],
            output_path=output_path,
            resolution=config.resolution,
            device=device,
            dtype=dtype,
        )
    return roundtrip_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-manifest", required=True)
    args = parser.parse_args()
    root = generate_roundtrip_controls(args.config, args.reference_manifest)
    print(root)


if __name__ == "__main__":
    main()
