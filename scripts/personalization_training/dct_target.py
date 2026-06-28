from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrequencyMasks:
    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


@dataclass(frozen=True)
class FrequencyBands:
    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


def _dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(torch.pi / size * (n + 0.5) * k)
    matrix[0, :] *= (1.0 / size) ** 0.5
    if size > 1:
        matrix[1:, :] *= (2.0 / size) ** 0.5
    return matrix


def dct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("dct2 expects at least two spatial dimensions")
    height, width = x.shape[-2:]
    left = _dct_matrix(height, device=x.device, dtype=x.dtype)
    right = _dct_matrix(width, device=x.device, dtype=x.dtype)
    flat = x.reshape(-1, height, width)
    transformed = left @ flat @ right.transpose(0, 1)
    return transformed.reshape_as(x)


def idct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("idct2 expects at least two spatial dimensions")
    height, width = x.shape[-2:]
    left = _dct_matrix(height, device=x.device, dtype=x.dtype)
    right = _dct_matrix(width, device=x.device, dtype=x.dtype)
    flat = x.reshape(-1, height, width)
    reconstructed = left.transpose(0, 1) @ flat @ right
    return reconstructed.reshape_as(x)


def _validate_radii(height: int, width: int, low_radius: int, mid_radius: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if low_radius <= 0:
        raise ValueError("low_radius must be positive")
    if mid_radius <= low_radius:
        raise ValueError("mid_radius must be greater than low_radius")


def frequency_masks(
    height: int,
    width: int,
    low_radius: int,
    mid_radius: int,
    device: torch.device | str | None = None,
) -> FrequencyMasks:
    _validate_radii(height, width, low_radius, mid_radius)
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    y = torch.arange(height, device=resolved_device).unsqueeze(1)
    x = torch.arange(width, device=resolved_device).unsqueeze(0)
    distance = y + x
    low = distance < low_radius
    mid = (distance >= low_radius) & (distance < mid_radius)
    high = distance >= mid_radius
    return FrequencyMasks(low=low, mid=mid, high=high)


def split_frequency_bands(x: torch.Tensor, low_radius: int, mid_radius: int) -> FrequencyBands:
    coeffs = dct2(x)
    masks = frequency_masks(
        height=x.shape[-2],
        width=x.shape[-1],
        low_radius=low_radius,
        mid_radius=mid_radius,
        device=x.device,
    )
    low = idct2(coeffs * masks.low.to(dtype=x.dtype))
    mid = idct2(coeffs * masks.mid.to(dtype=x.dtype))
    high = idct2(coeffs * masks.high.to(dtype=x.dtype))
    return FrequencyBands(low=low, mid=mid, high=high)
