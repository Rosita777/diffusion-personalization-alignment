from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.personalization_training.dct_target import FrequencyBands, split_frequency_bands


@dataclass(frozen=True)
class LFLateAlignmentConfig:
    alpha: float
    late_timestep_threshold: int
    low_radius: int
    mid_radius: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.late_timestep_threshold < 0:
            raise ValueError("late_timestep_threshold must be non-negative")
        if self.low_radius <= 0:
            raise ValueError("low_radius must be positive")
        if self.mid_radius <= self.low_radius:
            raise ValueError("mid_radius must be greater than low_radius")

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
