import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import augment, normalize  # noqa: E402


def test_augment_keeps_shape_and_pixels():
    x = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    for _ in range(20):
        out = augment(x)
        assert out.shape == x.shape
        # Flips and 90-degree rotations only move pixels around, never change them.
        assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)


def test_normalize_output():
    x = torch.full((2, 3, 64, 64), 128, dtype=torch.uint8)
    out = normalize(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.dtype == torch.float32
    assert abs(out.mean().item()) < 1.0
