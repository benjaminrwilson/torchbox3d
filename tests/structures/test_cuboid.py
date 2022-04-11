"""Unit tests for the cuboid module."""

import pytest
import torch
from torch import Tensor

from torchbox3d.structures.cuboids import Cuboids


@pytest.mark.parametrize(
    "cuboids, vertices_m_",
    [
        pytest.param(
            Cuboids(
                params=torch.as_tensor(
                    [[1, 1, 1, 1, 2, 3, 1, 0, 0, 0]], dtype=torch.float
                ),
                categories=torch.zeros(10),
                scores=torch.ones(10),
            ),
            torch.as_tensor([]),
        )
    ],
)
def test_cuboid(cuboids: Cuboids, vertices_m_: Tensor) -> None:
    """Unit test for cuboid instance methods."""
    cuboids.vertices_m
