import numpy as np

from scripts.off_prior_measurement.metrics import dct_band_energy, residual_metrics, residual_projection_metrics


def test_residual_metrics_zero_when_vectors_match():
    v_ref = np.ones((1, 4, 4, 4), dtype=np.float32)
    v_base = np.ones((1, 4, 4, 4), dtype=np.float32)

    metrics = residual_metrics(v_ref, v_base)

    assert metrics["normalized_l2"] == 0.0
    assert metrics["cosine_distance"] == 0.0


def test_residual_metrics_positive_when_vectors_differ():
    v_ref = np.ones((1, 4, 4, 4), dtype=np.float32)
    v_base = np.zeros((1, 4, 4, 4), dtype=np.float32)

    metrics = residual_metrics(v_ref, v_base)

    assert metrics["normalized_l2"] > 0.9
    assert metrics["cosine_distance"] >= 0.0


def test_dct_band_energy_returns_three_bands():
    tensor = np.ones((1, 4, 8, 8), dtype=np.float32)

    bands = dct_band_energy(tensor)

    assert set(bands) == {"low", "mid", "high"}
    assert bands["low"] >= 0.0
    assert bands["mid"] >= 0.0
    assert bands["high"] >= 0.0


def test_dct_band_energy_handles_tiny_tensors_without_nan():
    tensor = np.ones((1, 1, 2, 2), dtype=np.float32)

    bands = dct_band_energy(tensor)

    assert np.isfinite(bands["low"])
    assert np.isfinite(bands["mid"])
    assert np.isfinite(bands["high"])


def test_residual_projection_metrics_zero_clean_when_residuals_match():
    v_ref = np.array([2.0, 0.0], dtype=np.float32)
    v_base = np.array([0.0, 0.0], dtype=np.float32)
    rt_ref = np.array([2.0, 0.0], dtype=np.float32)
    rt_base = np.array([0.0, 0.0], dtype=np.float32)

    metrics = residual_projection_metrics(v_ref, v_base, rt_ref, rt_base)

    assert round(metrics["artifact_fraction"], 6) == 1.0
    assert round(metrics["clean_norm"], 6) == 0.0
    assert round(metrics["artifact_cosine"], 6) == 1.0


def test_residual_projection_metrics_keeps_orthogonal_clean_residual():
    v_ref = np.array([0.0, 2.0], dtype=np.float32)
    v_base = np.array([0.0, 0.0], dtype=np.float32)
    rt_ref = np.array([2.0, 0.0], dtype=np.float32)
    rt_base = np.array([0.0, 0.0], dtype=np.float32)

    metrics = residual_projection_metrics(v_ref, v_base, rt_ref, rt_base)

    assert round(metrics["artifact_fraction"], 6) == 0.0
    assert round(metrics["clean_fraction"], 6) == 1.0
    assert round(metrics["artifact_cosine"], 6) == 0.0
