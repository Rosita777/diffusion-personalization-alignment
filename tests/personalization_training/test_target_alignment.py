import pytest
import torch

from scripts.personalization_training.target_alignment import (
    LFLateAlignmentConfig,
    apply_lf_late_alignment,
)


def test_alignment_alpha_zero_returns_reference_target():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=0.0, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    aligned = apply_lf_late_alignment(ref, base, timestep=900, config=cfg)

    assert torch.allclose(aligned, ref)


def test_alignment_skips_early_timestep():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=0.5, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    aligned = apply_lf_late_alignment(ref, base, timestep=200, config=cfg)

    assert torch.allclose(aligned, ref)


def test_alignment_replaces_only_low_frequency_at_alpha_one():
    ref = torch.randn(1, 2, 8, 8)
    base = torch.randn(1, 2, 8, 8)
    cfg = LFLateAlignmentConfig(alpha=1.0, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    aligned = apply_lf_late_alignment(ref, base, timestep=900, config=cfg)

    ref_bands = cfg.split(ref)
    base_bands = cfg.split(base)
    aligned_bands = cfg.split(aligned)
    assert torch.allclose(aligned_bands.low, base_bands.low, atol=1e-5)
    assert torch.allclose(aligned_bands.mid, ref_bands.mid, atol=1e-5)
    assert torch.allclose(aligned_bands.high, ref_bands.high, atol=1e-5)


def test_alignment_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        LFLateAlignmentConfig(alpha=1.1, late_timestep_threshold=800, low_radius=2, mid_radius=4)
