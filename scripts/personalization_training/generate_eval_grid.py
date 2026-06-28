from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.personalization_training.config import Stage2AConfig, SubjectConfig, load_training_config
from scripts.personalization_training.train_lora_dreambooth import _disable_awq_lora_dispatch


@dataclass(frozen=True)
class EvaluationPrompt:
    kind: str
    text: str


def safe_prompt_slug(prompt: str, max_length: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "prompt"


def _subject_terms(subject: SubjectConfig) -> set[str]:
    terms = {subject.subject_id, subject.class_name}
    return {term.lower() for term in terms if term}


def _prompt_matches_subject(prompt: str, subject: SubjectConfig) -> bool:
    prompt_lower = prompt.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", prompt_lower) for term in _subject_terms(subject))


def _find_subject(config: Stage2AConfig, subject_id: str) -> SubjectConfig:
    for subject in config.subjects:
        if subject.subject_id == subject_id:
            return subject
    available = ", ".join(subject.subject_id for subject in config.subjects)
    raise ValueError(f"Unknown subject_id '{subject_id}'. Available subjects: {available}")


def evaluation_prompts_for_subject(config: Stage2AConfig, subject_id: str | None) -> list[EvaluationPrompt]:
    subject = _find_subject(config, subject_id) if subject_id else None
    prompts: list[EvaluationPrompt] = []
    for kind, prompt_list in [
        ("subject", config.evaluation.prompts),
        ("class", config.evaluation.class_prompts),
    ]:
        for prompt in prompt_list:
            if subject is None or _prompt_matches_subject(prompt, subject):
                prompts.append(EvaluationPrompt(kind=kind, text=prompt))
    if not prompts:
        target = subject_id if subject_id else "all subjects"
        raise ValueError(f"No evaluation prompts matched {target}")
    return prompts


def _output_dir_for(config: Stage2AConfig, output_dir: str | None, subject_id: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if subject_id:
        return config.output_dir / "eval_grids" / subject_id
    return config.output_dir / "eval_grids"


def _image_filename(prompt: EvaluationPrompt, prompt_index: int, image_index: int, seed: int) -> str:
    slug = safe_prompt_slug(prompt.text)
    return f"{prompt.kind}_{prompt_index:02d}_{image_index:02d}_seed{seed}_{slug}.png"


def _save_grid(image_paths: list[Path], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    tile_width, tile_height = images[0].size
    label_height = 22
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * (tile_height + label_height)
        grid.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 4), image_paths[index].stem[:80], fill=(0, 0, 0), font=font)
    grid.save(output_path)


def _load_pipeline(model_id: str, device: Any, torch_dtype: Any) -> Any:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_eval_lora_weights(pipe: Any, weights_dir: Path) -> None:
    _disable_awq_lora_dispatch()
    pipe.unet.load_lora_adapter(
        weights_dir,
        adapter_name="default",
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
    )


def generate_eval_grid(
    config: Stage2AConfig,
    *,
    subject_id: str | None,
    weights_dir: str | None,
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    prompts = evaluation_prompts_for_subject(config, subject_id=subject_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline(config.model_id, device=device, torch_dtype=torch_dtype)
    if weights_dir:
        load_eval_lora_weights(pipe, Path(weights_dir))

    records: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(config.evaluation.num_images_per_prompt):
            seed = config.training.seed + prompt_index * 100 + image_index
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                prompt.text,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            image_path = output_dir / _image_filename(prompt, prompt_index, image_index, seed)
            image.save(image_path)
            image_paths.append(image_path)
            records.append(
                {
                    **asdict(prompt),
                    "prompt_index": prompt_index,
                    "image_index": image_index,
                    "seed": seed,
                    "path": str(image_path),
                }
            )

    _save_grid(image_paths, output_dir / "grid.png")
    manifest = {
        "model_id": config.model_id,
        "subject_id": subject_id,
        "weights_dir": weights_dir,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": len(records),
        "images": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 2A evaluation grids.")
    parser.add_argument("--config", required=True, help="Path to Stage 2A YAML config.")
    parser.add_argument("--subject-id", help="Subject id to evaluate from a multi-subject config.")
    parser.add_argument("--weights-dir", help="Directory containing LoRA weights.")
    parser.add_argument("--output-dir", help="Directory for generated grids.")
    parser.add_argument("--num-inference-steps", type=int, default=25, help="Denoising steps per image.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and create output directory only.")
    args = parser.parse_args()

    config = load_training_config(Path(args.config))
    prompts = evaluation_prompts_for_subject(config, subject_id=args.subject_id)
    output_dir = _output_dir_for(config, args.output_dir, args.subject_id)
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"would write eval grids to: {output_dir}")
        print(f"prompts: {len(prompts)}")
        return
    generate_eval_grid(
        config,
        subject_id=args.subject_id,
        weights_dir=args.weights_dir,
        output_dir=output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
