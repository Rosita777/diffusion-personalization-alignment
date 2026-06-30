import pytest
import torch

from scripts.personalization_training.target_alignment import (
    LFLateAlignmentConfig,
    ResidualGateConfig,
    apply_cfg_residual_gate,
    apply_lf_late_alignment,
    apply_residual_magnitude_gate,
    cfg_residual_cosine,
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


def test_residual_gate_suppresses_only_large_residual_locations():
    ref = torch.tensor([[[[0.0, 10.0], [2.0, 4.0]]]])
    base = torch.zeros_like(ref)
    cfg = ResidualGateConfig(residual_gate_quantile=0.75, residual_gate_keep=0.5)

    aligned = apply_residual_magnitude_gate(ref, base, cfg)

    assert torch.allclose(aligned, torch.tensor([[[[0.0, 5.0], [2.0, 4.0]]]]))


def test_residual_gate_keep_one_returns_reference_target():
    ref = torch.randn(1, 2, 4, 4)
    base = torch.randn(1, 2, 4, 4)
    cfg = ResidualGateConfig(residual_gate_quantile=0.75, residual_gate_keep=1.0)

    aligned = apply_residual_magnitude_gate(ref, base, cfg)

    assert torch.allclose(aligned, ref)


def test_residual_gate_rejects_invalid_values():
    with pytest.raises(ValueError, match="residual_gate_quantile"):
        ResidualGateConfig(residual_gate_quantile=1.5, residual_gate_keep=0.5)
    with pytest.raises(ValueError, match="residual_gate_keep"):
        ResidualGateConfig(residual_gate_quantile=0.75, residual_gate_keep=-0.1)


def test_cfg_residual_gate_suppresses_class_aligned_residuals_only():
    class_prediction = torch.zeros(1, 1, 1, 4)
    null_prediction = torch.full_like(class_prediction, -1.0)
    reference = torch.tensor([[[[10.0, -10.0, 0.0, 4.0]]]])
    cfg = LFLateAlignmentConfig(alpha=0.5, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    aligned = apply_cfg_residual_gate(reference, class_prediction, null_prediction, cfg)

    assert torch.allclose(aligned, torch.tensor([[[[5.0, -10.0, 0.0, 2.0]]]]))


def test_cfg_residual_gate_alpha_zero_returns_reference_target():
    reference = torch.randn(1, 2, 4, 4)
    class_prediction = torch.randn(1, 2, 4, 4)
    null_prediction = torch.randn(1, 2, 4, 4)
    cfg = LFLateAlignmentConfig(alpha=0.0, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    aligned = apply_cfg_residual_gate(reference, class_prediction, null_prediction, cfg)

    assert torch.allclose(aligned, reference)


def test_cfg_residual_gate_rejects_mismatched_shapes():
    reference = torch.randn(1, 2, 4, 4)
    class_prediction = torch.randn(1, 2, 4, 4)
    null_prediction = torch.randn(1, 2, 2, 2)
    cfg = LFLateAlignmentConfig(alpha=0.5, late_timestep_threshold=800, low_radius=2, mid_radius=4)

    with pytest.raises(ValueError, match="same shape"):
        apply_cfg_residual_gate(reference, class_prediction, null_prediction, cfg)


def test_cfg_residual_cosine_reports_per_location_alignment():
    residual = torch.tensor([[[[1.0, -1.0]], [[0.0, 0.0]]]])
    class_direction = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])

    cosine = cfg_residual_cosine(residual, class_direction)

    assert torch.allclose(cosine, torch.tensor([[[[1.0, -1.0]]]]), atol=1e-5)
