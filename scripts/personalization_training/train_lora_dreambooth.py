from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scripts.personalization_training.config import ALLOWED_CONDITIONS, Stage2AConfig, load_training_config
from scripts.personalization_training.target_alignment import apply_lf_late_alignment, apply_residual_magnitude_gate


@dataclass(frozen=True)
class TrainingExample:
    subject_id: str
    image_path: Path
    prompt: str
    class_prompt: str


def build_training_examples(config: Stage2AConfig) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for subject in config.subjects:
        for image_path in subject.image_paths:
            examples.append(
                TrainingExample(
                    subject_id=subject.subject_id,
                    image_path=image_path,
                    prompt=subject.instance_prompt,
                    class_prompt=subject.class_prompt,
                )
            )
    return examples


class ReferenceImageDataset:
    def __init__(self, examples: list[TrainingExample], resolution: int) -> None:
        if not examples:
            raise ValueError("ReferenceImageDataset needs at least one training example")
        self.examples = examples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image
        from torchvision import transforms

        example = self.examples[index]
        preprocess = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image = Image.open(example.image_path).convert("RGB")
        return {
            "pixel_values": preprocess(image),
            "prompt": example.prompt,
            "class_prompt": example.class_prompt,
            "subject_id": example.subject_id,
            "image_path": str(example.image_path),
        }


def collate_reference_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "prompts": [item["prompt"] for item in items],
        "class_prompts": [item["class_prompt"] for item in items],
        "subject_ids": [item["subject_id"] for item in items],
        "image_paths": [item["image_path"] for item in items],
    }


def config_with_condition(config: Stage2AConfig, condition: str) -> Stage2AConfig:
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"Unsupported training condition: {condition}")
    return replace(config, training=replace(config.training, condition=condition))


def config_with_max_train_steps(config: Stage2AConfig, max_train_steps: int | None) -> Stage2AConfig:
    if max_train_steps is None:
        return config
    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    return replace(config, training=replace(config.training, max_train_steps=max_train_steps))


def filter_config_subjects(config: Stage2AConfig, subject_id: str | None) -> Stage2AConfig:
    if subject_id is None:
        return config
    subjects = [subject for subject in config.subjects if subject.subject_id == subject_id]
    if not subjects:
        available = ", ".join(subject.subject_id for subject in config.subjects)
        raise ValueError(f"Unknown subject_id '{subject_id}'. Available subjects: {available}")
    return replace(config, subjects=subjects)


def training_output_dir(config: Stage2AConfig, subject_id: str | None = None, run_name: str | None = None) -> Path:
    output_dir = config.output_dir / (run_name or config.training.condition)
    if subject_id:
        output_dir = output_dir / subject_id
    return output_dir


def _single_subject_id(config: Stage2AConfig) -> str | None:
    if len(config.subjects) == 1:
        return config.subjects[0].subject_id
    return None


def target_for_condition(
    reference_target: Any,
    base_prediction: Any,
    timestep: int | Any,
    config: Stage2AConfig,
) -> Any:
    if config.training.condition == "vanilla":
        return reference_target
    if config.training.condition == "dadt_lf_late":
        return apply_lf_late_alignment(
            reference_target=reference_target,
            base_prediction=base_prediction,
            timestep=timestep,
            config=config.alignment,
        )
    if config.training.condition == "dadt_residual_gate":
        return apply_residual_magnitude_gate(
            reference_target=reference_target,
            base_prediction=base_prediction,
            config=config.alignment,
        )
    raise ValueError(f"Unsupported training condition: {config.training.condition}")


def dry_run_summary(config: Stage2AConfig, run_name: str | None = None) -> str:
    lines = [
        "Stage 2A LoRA DreamBooth dry run",
        f"model_id: {config.model_id}",
        f"output_dir: {training_output_dir(config, subject_id=_single_subject_id(config), run_name=run_name)}",
        f"condition: {config.training.condition}",
        f"max_train_steps: {config.training.max_train_steps}",
        f"lora_rank: {config.training.lora_rank}",
        f"alpha: {config.alignment.alpha}",
    ]
    for subject in config.subjects:
        lines.append(f"subject: {subject.subject_id}")
        lines.append(f"  instance_prompt: {subject.instance_prompt}")
        lines.append(f"  class_prompt: {subject.class_prompt}")
        lines.append(f"  images: {len(subject.image_paths)}")
    return "\n".join(lines)


def _freeze(module: Any) -> None:
    module.requires_grad_(False)
    module.eval()


def _tokenize_prompts(tokenizer: Any, prompts: list[str], device: Any) -> Any:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return tokens.input_ids.to(device)


def _encode_prompts(text_encoder: Any, tokenizer: Any, prompts: list[str], device: Any) -> Any:
    input_ids = _tokenize_prompts(tokenizer, prompts, device)
    return text_encoder(input_ids)[0]


def _scheduler_prediction_type(scheduler: Any) -> str:
    return str(getattr(scheduler.config, "prediction_type", "epsilon"))


def _training_target(scheduler: Any, latents: Any, noise: Any, timesteps: Any) -> Any:
    prediction_type = _scheduler_prediction_type(scheduler)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")


def _move_batch(batch: dict[str, Any], *, device: Any, dtype: Any) -> dict[str, Any]:
    return {
        **batch,
        "pixel_values": batch["pixel_values"].to(device=device, dtype=dtype),
    }


def _trainable_parameters(module: Any) -> list[Any]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _cast_trainable_parameters_to_float32(module: Any) -> None:
    for parameter in _trainable_parameters(module):
        parameter.data = parameter.data.float()


def _ensure_finite_loss(loss: Any, step: int) -> None:
    if not loss.isfinite().all().item():
        raise RuntimeError(f"Non-finite loss at training step {step}")


def _load_base_components(config: Stage2AConfig, torch_dtype: Any, device: Any) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        config.model_id,
        subfolder="vae",
        torch_dtype=torch_dtype,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        config.model_id,
        subfolder="unet",
        torch_dtype=torch_dtype,
    ).to(device)
    _freeze(text_encoder)
    _freeze(vae)
    return {
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
    }


def _attach_lora(unet: Any, rank: int) -> None:
    from peft import LoraConfig

    _disable_awq_lora_dispatch()
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.train()


def _disable_awq_lora_dispatch() -> None:
    # This environment has an incompatible AutoAWQ install; SD 1.5 UNet LoRA does not need AWQ dispatch.
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.awq as peft_lora_awq
    except Exception:
        return

    peft_import_utils.is_auto_awq_available = lambda: False
    peft_lora_awq.is_auto_awq_available = lambda: False


def _load_base_unet(config: Stage2AConfig, torch_dtype: Any, device: Any) -> Any:
    from diffusers import UNet2DConditionModel

    base_unet = UNet2DConditionModel.from_pretrained(
        config.model_id,
        subfolder="unet",
        torch_dtype=torch_dtype,
    ).to(device)
    _freeze(base_unet)
    return base_unet


def run_training(config: Stage2AConfig, run_name: str | None = None) -> None:
    # Heavy imports stay inside this function so --help and --dry-run remain cheap.
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    examples = build_training_examples(config)
    dataset = ReferenceImageDataset(examples, resolution=config.resolution)
    generator = torch.Generator().manual_seed(config.training.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        collate_fn=collate_reference_batch,
        generator=generator,
    )

    components = _load_base_components(config, torch_dtype=torch_dtype, device=device)
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    vae = components["vae"]
    unet = components["unet"]
    _freeze(unet)
    _attach_lora(unet, rank=config.training.lora_rank)
    _cast_trainable_parameters_to_float32(unet)
    trainable_parameters = _trainable_parameters(unet)
    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters were found on the UNet")

    base_unet = None
    if config.training.condition.startswith("dadt_"):
        base_unet = _load_base_unet(config, torch_dtype=torch_dtype, device=device)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.training.learning_rate)
    output_dir = training_output_dir(config, subject_id=_single_subject_id(config), run_name=run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    progress = tqdm(range(config.training.max_train_steps), desc=f"train {config.training.condition}")
    data_iter = iter(dataloader)
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = _move_batch(batch, device=device, dtype=torch_dtype)

        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps,
                size=(latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            reference_target = _training_target(scheduler, latents, noise, timesteps)
            prompt_embeds = _encode_prompts(text_encoder, tokenizer, batch["prompts"], device)

            if base_unet is None:
                base_prediction = reference_target
            else:
                class_embeds = _encode_prompts(text_encoder, tokenizer, batch["class_prompts"], device)
                base_prediction = base_unet(noisy_latents, timesteps, encoder_hidden_states=class_embeds).sample
            target = target_for_condition(
                reference_target=reference_target.float(),
                base_prediction=base_prediction.float(),
                timestep=timesteps,
                config=config,
            ).to(dtype=torch_dtype)

        model_prediction = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
        loss = F.mse_loss(model_prediction.float(), target.float(), reduction="mean")
        _ensure_finite_loss(loss, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    unet.save_lora_adapter(output_dir, adapter_name="default", safe_serialization=True)
    summary = {
        "condition": config.training.condition,
        "subjects": [subject.subject_id for subject in config.subjects],
        "model_id": config.model_id,
        "resolution": config.resolution,
        "max_train_steps": config.training.max_train_steps,
        "train_batch_size": config.training.train_batch_size,
        "learning_rate": config.training.learning_rate,
        "lora_rank": config.training.lora_rank,
        "seed": config.training.seed,
        "prediction_type": _scheduler_prediction_type(scheduler),
        "loss_first": losses[0] if losses else math.nan,
        "loss_last": losses[-1] if losses else math.nan,
        "loss_mean": sum(losses) / len(losses) if losses else math.nan,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter and summary to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2A LoRA DreamBooth training.")
    parser.add_argument("--config", required=False, help="Path to Stage 2A YAML config.")
    parser.add_argument("--condition", choices=sorted(ALLOWED_CONDITIONS), help="Override training condition.")
    parser.add_argument("--subject-id", help="Train only one subject from a multi-subject config.")
    parser.add_argument("--run-name", help="Output label for sweep variants; defaults to condition.")
    parser.add_argument("--max-train-steps", type=int, help="Override config max_train_steps for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned run without loading SD.")
    args = parser.parse_args()

    if not args.config:
        if args.dry_run:
            raise ValueError("--config is required for --dry-run")
        parser.print_help()
        return

    config = load_training_config(Path(args.config))
    if args.condition:
        config = config_with_condition(config, args.condition)
    config = config_with_max_train_steps(config, args.max_train_steps)
    config = filter_config_subjects(config, args.subject_id)
    if args.dry_run:
        print(dry_run_summary(config, run_name=args.run_name))
        return
    run_training(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
