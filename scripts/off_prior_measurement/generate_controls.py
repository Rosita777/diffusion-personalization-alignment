from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.off_prior_measurement.config import SubjectSpec, load_config
from scripts.off_prior_measurement.dreambooth_data import conditioning_prompt


def _dtype_from_name(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def generate_controls(config_path: str | Path) -> Path:
    import torch
    from diffusers import StableDiffusionPipeline

    config = load_config(config_path)
    generated_root = config.cache_dir / "generated_controls"
    generated_root.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=_dtype_from_name(config.dtype),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(config.device)

    for subject in config.subjects:
        prompts = {
            "easy": subject.class_prompt,
            "hard": subject.hard_control_prompt,
        }
        for regime, prompt in prompts.items():
            out_dir = generated_root / subject.subject_id / regime
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed in range(config.control_images_per_subject):
                path = out_dir / f"seed_{seed:04d}.png"
                if path.exists():
                    continue
                generator = torch.Generator(device=config.device).manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                    height=config.resolution,
                    width=config.resolution,
                ).images[0]
                image.save(path)
    return generated_root


def build_control_manifest(
    subjects: list[SubjectSpec],
    generated_root: Path,
    conditionings: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for subject in subjects:
        for regime_dir, source_group, reference_regime, hardness_axis in [
            ("easy", "base_easy_control", "easy_control", "none"),
            ("hard", "base_hard_control", "hard_control", "clutter_background"),
        ]:
            image_dir = generated_root / subject.subject_id / regime_dir
            images = sorted(image_dir.glob("*.png"))
            if not images:
                raise FileNotFoundError(f"No generated controls found in {image_dir}")
            for image_path in images:
                for conditioning_key in conditionings:
                    rows.append(
                        {
                            "subject_id": subject.subject_id,
                            "image_id": image_path.stem,
                            "image_path": str(image_path),
                            "source_group": source_group,
                            "reference_regime": reference_regime,
                            "hardness_axis": hardness_axis,
                            "source_standard_image": "",
                            "variant_id": "",
                            "transform_parameters": "{}",
                            "class_name": subject.class_name,
                            "class_prompt": subject.class_prompt,
                            "class_context_prompt": subject.class_context_prompt,
                            "conditioning_key": conditioning_key,
                            "conditioning_prompt": conditioning_prompt(subject, conditioning_key),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generated_root = generate_controls(args.config)
    print(generated_root)


if __name__ == "__main__":
    main()
