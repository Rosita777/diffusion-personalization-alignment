from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.personalization_training.config import Stage2AConfig, load_training_config
from scripts.personalization_training.target_alignment import cfg_residual_cosine
from scripts.personalization_training.train_lora_dreambooth import (
    ReferenceImageDataset,
    _encode_prompts,
    _load_base_components,
    _load_base_unet,
    _move_batch,
    _training_target,
    build_training_examples,
    collate_reference_batch,
    filter_config_subjects,
)


def summarize_cosine_tensor(cosine: torch.Tensor) -> dict[str, float | int]:
    values = cosine.detach().float().flatten()
    if values.numel() == 0:
        raise ValueError("cosine tensor must contain at least one value")
    return {
        "num_values": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "positive_ratio": float((values > 0).float().mean().item()),
    }


def run_cfg_residual_diagnostic(
    config: Stage2AConfig,
    *,
    timestep: int,
    max_batches: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    if timestep < 0:
        raise ValueError("timestep must be non-negative")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    dataset = ReferenceImageDataset(build_training_examples(config), resolution=config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.train_batch_size,
        shuffle=False,
        collate_fn=collate_reference_batch,
    )

    components = _load_base_components(config, torch_dtype=torch_dtype, device=device)
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    vae = components["vae"]
    base_unet = _load_base_unet(config, torch_dtype=torch_dtype, device=device)

    cosine_chunks: list[torch.Tensor] = []
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if batches >= max_batches:
                break
            batch = _move_batch(batch, device=device, dtype=torch_dtype)
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.full(
                size=(latents.shape[0],),
                fill_value=timestep,
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            reference_target = _training_target(scheduler, latents, noise, timesteps).float()
            class_embeds = _encode_prompts(text_encoder, tokenizer, batch["class_prompts"], device)
            null_embeds = _encode_prompts(text_encoder, tokenizer, [""] * latents.shape[0], device)
            class_prediction = base_unet(noisy_latents, timesteps, encoder_hidden_states=class_embeds).sample.float()
            null_prediction = base_unet(noisy_latents, timesteps, encoder_hidden_states=null_embeds).sample.float()
            residual = reference_target - class_prediction
            class_direction = class_prediction - null_prediction
            cosine_chunks.append(cfg_residual_cosine(residual, class_direction).cpu())
            batches += 1

    if not cosine_chunks:
        raise ValueError("diagnostic did not process any batches")
    cosine = torch.cat([chunk.flatten() for chunk in cosine_chunks])
    summary = summarize_cosine_tensor(cosine)
    return {
        "condition": config.training.condition,
        "subjects": [subject.subject_id for subject in config.subjects],
        "model_id": config.model_id,
        "device": str(device),
        "timestep": int(timestep),
        "num_batches": batches,
        "cosine": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage 2C-1 CFG residual gate cosine statistics.")
    parser.add_argument("--config", required=True, type=Path, help="Path to Stage 2C-1 YAML config.")
    parser.add_argument("--subject-id", help="Subject id to diagnose from a multi-subject config.")
    parser.add_argument("--timestep", type=int, default=500, help="Fixed diffusion timestep for the diagnostic.")
    parser.add_argument("--max-batches", type=int, default=3, help="Maximum reference batches to inspect.")
    parser.add_argument("--output-json", type=Path, help="Optional path for JSON summary.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    config = filter_config_subjects(config, args.subject_id)
    summary = run_cfg_residual_diagnostic(config, timestep=args.timestep, max_batches=args.max_batches)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
