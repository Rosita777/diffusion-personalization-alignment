from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.off_prior_measurement.config import ExperimentConfig


def torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def training_target(scheduler, prediction_type: str, latents, noise, timesteps):
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


@dataclass
class MeasurementBatch:
    v_ref: Any
    v_base: Any
    noisy_latents: Any
    latents: Any
    snr: float


class StableDiffusionBackend:
    def __init__(self, config: ExperimentConfig):
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from torchvision import transforms
        from transformers import CLIPTextModel, CLIPTokenizer

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch_dtype(config.dtype)
        self.scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            config.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.model_id,
            subfolder="unet",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((config.resolution, config.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def encode_prompt(self, prompt: str):
        import torch

        with torch.no_grad():
            tokens = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            return self.text_encoder(tokens)[0].to(dtype=self.dtype)

    def encode_image(self, image_path: str | Path):
        import torch
        from PIL import Image

        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            pixels = self.preprocess(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
            posterior = self.vae.encode(pixels).latent_dist
            return posterior.sample() * self.vae.config.scaling_factor

    def measure(self, image_path: str | Path, prompt: str, timestep: int, seed: int) -> MeasurementBatch:
        import torch

        with torch.no_grad():
            latents = self.encode_image(image_path)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
            timesteps = torch.tensor([timestep], device=self.device, dtype=torch.long)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            prompt_embeds = self.encode_prompt(prompt)
            v_base = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
            v_ref = training_target(self.scheduler, self.config.prediction_type, latents, noise, timesteps)
            alpha = self.scheduler.alphas_cumprod.to(self.device)[timesteps].float()
            snr = (alpha / (1.0 - alpha).clamp_min(1e-8)).item()
            return MeasurementBatch(
                v_ref=v_ref.float(),
                v_base=v_base.float(),
                noisy_latents=noisy_latents.float(),
                latents=latents.float(),
                snr=float(snr),
            )
