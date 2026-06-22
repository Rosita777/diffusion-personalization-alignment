from __future__ import annotations

import numpy as np
from scipy.fft import dctn


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def residual_metrics(v_ref, v_base) -> dict[str, float]:
    ref = _to_numpy(v_ref).astype(np.float64).reshape(-1)
    base = _to_numpy(v_base).astype(np.float64).reshape(-1)
    delta = ref - base
    ref_norm = max(float(np.linalg.norm(ref)), 1e-8)
    base_norm = max(float(np.linalg.norm(base)), 1e-8)
    normalized_l2 = float(np.linalg.norm(delta) / ref_norm)
    cosine_similarity = float(np.dot(ref, base) / (ref_norm * base_norm))
    cosine_similarity = min(1.0, max(-1.0, cosine_similarity))
    cosine_distance = 1.0 - cosine_similarity
    base_to_ref_norm_ratio = base_norm / ref_norm
    return {
        "normalized_l2": normalized_l2,
        "cosine_distance": cosine_distance,
        "base_to_ref_norm_ratio": base_to_ref_norm_ratio,
    }


def dct_band_energy(tensor) -> dict[str, float]:
    array = _to_numpy(tensor).astype(np.float64)
    transformed = dctn(array, axes=(-2, -1), norm="ortho")
    energy = np.mean(transformed**2, axis=tuple(range(transformed.ndim - 2)))
    height, width = energy.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    radius = np.sqrt((yy / max(height - 1, 1)) ** 2 + (xx / max(width - 1, 1)) ** 2)
    low_mask = radius <= 0.2
    mid_mask = (radius > 0.2) & (radius <= 0.6)
    high_mask = radius > 0.6
    return {
        "low": float(energy[low_mask].mean()),
        "mid": float(energy[mid_mask].mean()),
        "high": float(energy[high_mask].mean()),
    }
