import torch

from scripts.personalization_training.diagnose_cfg_residual_gate import summarize_cosine_tensor


def test_summarize_cosine_tensor_reports_basic_stats():
    cosine = torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]])

    summary = summarize_cosine_tensor(cosine)

    assert summary["num_values"] == 4
    assert summary["mean"] == 0.125
    assert summary["min"] == -1.0
    assert summary["max"] == 1.0
    assert summary["positive_ratio"] == 0.5
