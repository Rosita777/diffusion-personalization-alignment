from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class PromptMatchedGenerationJob:
    prompt: str
    seed: int
    output_path: Path


def _read_yaml(path: str | Path) -> dict:
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}


def _ordinary_rows(path: str | Path) -> list[dict[str, str]]:
    rows = _read_yaml(path).get("ordinary_real_controls", [])
    if not rows:
        raise ValueError(f"No ordinary_real_controls entries found in {path}")
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _generated_images(generated_root: Path, row: dict[str, str], seeds_per_prompt: int) -> list[Path]:
    image_dir = generated_root / row["class_name"] / row["image_id"]
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
    images = images[:seeds_per_prompt]
    if len(images) < seeds_per_prompt:
        raise FileNotFoundError(
            f"Expected {seeds_per_prompt} prompt-matched generated images in {image_dir}, found {len(images)}"
        )
    return images


def prompt_matched_generation_jobs(
    ordinary_manifest_path: str | Path,
    generated_root: str | Path,
    seeds_per_prompt: int,
) -> list[PromptMatchedGenerationJob]:
    generated_root = Path(generated_root)
    jobs: list[PromptMatchedGenerationJob] = []
    for ordinary in _ordinary_rows(ordinary_manifest_path):
        output_dir = generated_root / ordinary["class_name"] / ordinary["image_id"]
        prompt = ordinary.get("conditioning_prompt", f"a photo of a {ordinary['class_name']}")
        for seed in range(seeds_per_prompt):
            jobs.append(
                PromptMatchedGenerationJob(
                    prompt=prompt,
                    seed=seed,
                    output_path=output_dir / f"seed_{seed:04d}.png",
                )
            )
    return jobs


def _dtype_from_name(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def generate_prompt_matched_controls(
    ordinary_manifest_path: str | Path,
    generated_root: str | Path,
    model_id: str,
    device: str,
    dtype: str,
    resolution: int,
    seeds_per_prompt: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> Path:
    import torch
    from diffusers import StableDiffusionPipeline

    generated_root = Path(generated_root)
    jobs = prompt_matched_generation_jobs(
        ordinary_manifest_path=ordinary_manifest_path,
        generated_root=generated_root,
        seeds_per_prompt=seeds_per_prompt,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_from_name(dtype),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    for job in jobs:
        if job.output_path.exists():
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(job.seed)
        image = pipe(
            prompt=job.prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=resolution,
            width=resolution,
        ).images[0]
        image.save(job.output_path)
    return generated_root


def build_prompt_matched_control_manifest(
    ordinary_manifest_path: str | Path,
    generated_root: str | Path,
    seeds_per_prompt: int,
) -> pd.DataFrame:
    generated_root = Path(generated_root)
    rows: list[dict[str, str]] = []
    for ordinary in _ordinary_rows(ordinary_manifest_path):
        for image_path in _generated_images(generated_root, ordinary, seeds_per_prompt):
            rows.append(
                {
                    "subject_id": ordinary["class_name"],
                    "image_id": f"{ordinary['image_id']}_{image_path.stem}",
                    "image_path": str(image_path),
                    "source_group": "base_generated_control",
                    "reference_regime": "prompt_matched_generated",
                    "hardness_axis": ordinary.get("hardness_axis", "none"),
                    "source_standard_image": "",
                    "variant_id": "",
                    "transform_parameters": "{}",
                    "class_name": ordinary["class_name"],
                    "class_prompt": f"a photo of a {ordinary['class_name']}",
                    "class_context_prompt": ordinary.get("conditioning_prompt", f"a photo of a {ordinary['class_name']}"),
                    "conditioning_key": ordinary.get("conditioning_key", "prompt_matched"),
                    "conditioning_prompt": ordinary.get("conditioning_prompt", f"a photo of a {ordinary['class_name']}"),
                    "source_dataset": "base_sd15_prompt_matched",
                    "source_license_note": "generated locally from the ordinary-real prompt-matched description",
                }
            )
    return pd.DataFrame(rows, dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--ordinary-real-manifest", required=True)
    generate_parser.add_argument("--generated-root", required=True)
    generate_parser.add_argument("--model-id", required=True)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--dtype", default="float16")
    generate_parser.add_argument("--resolution", type=int, default=512)
    generate_parser.add_argument("--seeds-per-prompt", type=int, default=2)
    generate_parser.add_argument("--num-inference-steps", type=int, default=30)
    generate_parser.add_argument("--guidance-scale", type=float, default=7.5)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--ordinary-real-manifest", required=True)
    manifest_parser.add_argument("--generated-root", required=True)
    manifest_parser.add_argument("--seeds-per-prompt", type=int, default=2)
    manifest_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "generate":
        print(
            generate_prompt_matched_controls(
                ordinary_manifest_path=args.ordinary_real_manifest,
                generated_root=args.generated_root,
                model_id=args.model_id,
                device=args.device,
                dtype=args.dtype,
                resolution=args.resolution,
                seeds_per_prompt=args.seeds_per_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        )
        return

    if args.command != "manifest":
        raise ValueError(f"Unsupported command: {args.command}")
    manifest = build_prompt_matched_control_manifest(
        ordinary_manifest_path=args.ordinary_real_manifest,
        generated_root=args.generated_root,
        seeds_per_prompt=args.seeds_per_prompt,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
