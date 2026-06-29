from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.personalization_training.dct_target import FrequencyBands, split_frequency_bands


@dataclass(frozen=True)
class ResidualGateConfig:
    residual_gate_quantile: float
    residual_gate_keep: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.residual_gate_quantile <= 1.0:
            raise ValueError("residual_gate_quantile must be in [0, 1]")
        if not 0.0 <= self.residual_gate_keep <= 1.0:
            raise ValueError("residual_gate_keep must be in [0, 1]")


@dataclass(frozen=True)
class LFLateAlignmentConfig:
    alpha: float
    late_timestep_threshold: int
    low_radius: int
    mid_radius: int
    residual_gate_quantile: float = 0.75
    residual_gate_keep: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.late_timestep_threshold < 0:
            raise ValueError("late_timestep_threshold must be non-negative")
        if self.low_radius <= 0:
            raise ValueError("low_radius must be positive")
        if self.mid_radius <= self.low_radius:
            raise ValueError("mid_radius must be greater than low_radius")
        ResidualGateConfig(
            residual_gate_quantile=self.residual_gate_quantile,
            residual_gate_keep=self.residual_gate_keep,
        )

    def split(self, x: torch.Tensor) -> FrequencyBands:
        return split_frequency_bands(x, low_radius=self.low_radius, mid_radius=self.mid_radius)


def _is_late_timestep(timestep: int | torch.Tensor, threshold: int) -> bool:
    if isinstance(timestep, torch.Tensor):
        return bool((timestep >= threshold).any().item())
    return int(timestep) >= threshold


def apply_lf_late_alignment(
    reference_target: torch.Tensor,
    base_prediction: torch.Tensor,
    timestep: int | torch.Tensor,
    config: LFLateAlignmentConfig,
) -> torch.Tensor:
    if reference_target.shape != base_prediction.shape:
        raise ValueError("reference_target and base_prediction must have the same shape")
    if config.alpha == 0.0 or not _is_late_timestep(timestep, config.late_timestep_threshold):
        return reference_target

    ref_bands = config.split(reference_target)
    base_bands = config.split(base_prediction)
    low = (1.0 - config.alpha) * ref_bands.low + config.alpha * base_bands.low
    return low + ref_bands.mid + ref_bands.high


def apply_residual_magnitude_gate(
    reference_target: torch.Tensor,
    base_prediction: torch.Tensor,
    config: ResidualGateConfig,
) -> torch.Tensor:
    if reference_target.shape != base_prediction.shape:
        raise ValueError("reference_target and base_prediction must have the same shape")
    if config.residual_gate_keep == 1.0:
        return reference_target

    residual = reference_target - base_prediction
    magnitude = residual.abs().mean(dim=1, keepdim=True)
    flat = magnitude.flatten(start_dim=1)
    threshold = torch.quantile(flat.float(), q=config.residual_gate_quantile, dim=1)
    view_shape = [magnitude.shape[0]] + [1] * (magnitude.ndim - 1)
    threshold = threshold.reshape(view_shape).to(device=magnitude.device, dtype=magnitude.dtype)
    gate = torch.where(
        magnitude > threshold,
        torch.full_like(magnitude, config.residual_gate_keep),
        torch.ones_like(magnitude),
    )
    return base_prediction + gate * residual
