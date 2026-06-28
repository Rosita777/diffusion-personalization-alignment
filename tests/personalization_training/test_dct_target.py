import torch

from scripts.personalization_training.dct_target import (
    dct2,
    frequency_masks,
    idct2,
    split_frequency_bands,
)


def test_dct_roundtrip_reconstructs_tensor():
    x = torch.randn(2, 3, 8, 8)

    reconstructed = idct2(dct2(x))

    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=1e-5)


def test_frequency_masks_are_disjoint_and_complete():
    masks = frequency_masks(
        height=8,
        width=8,
        low_radius=2,
        mid_radius=4,
        device=torch.device("cpu"),
    )

    total = masks.low.to(torch.int) + masks.mid.to(torch.int) + masks.high.to(torch.int)

    assert torch.equal(total, torch.ones_like(total))
    assert masks.low.sum().item() > 0
    assert masks.mid.sum().item() > 0
    assert masks.high.sum().item() > 0


def test_split_frequency_bands_reconstructs_input():
    x = torch.randn(1, 4, 8, 8)

    bands = split_frequency_bands(x, low_radius=2, mid_radius=4)
    reconstructed = bands.low + bands.mid + bands.high

    assert torch.allclose(reconstructed, x, atol=1e-5)
