import numpy as np

from scripts.off_prior_measurement.metrics import dct_band_energy, residual_metrics


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
