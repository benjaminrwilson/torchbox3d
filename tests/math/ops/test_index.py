"""Unit tests for raveling and unraveling indices."""

from typing import List

import pytest
import torch
from torch import Tensor

from torchbox3d.math.ops.index import ravel_multi_index, unravel_index


@pytest.mark.parametrize(
    "unraveled_coords, shape, raveled_indices_",
    [
        pytest.param(
            torch.as_tensor([[2]]),
            [10],
            torch.as_tensor([2]),
        ),
        pytest.param(
            torch.as_tensor([[2, 3]]),
            [10, 10],
            torch.as_tensor([2 * 10 + 3 * 1]),
        ),
        pytest.param(
            torch.as_tensor([[3, 6, 9]]),
            [5, 10, 15],
            torch.as_tensor([3 * (10 * 15) + 6 * 15 + 9 * 1]),
        ),
    ],
)
def test_ravel_multi_index(
    unraveled_coords: Tensor, shape: List[int], raveled_indices_: Tensor
) -> None:
    """Unit tests for raveling a multi-index (i.e., unraveled coordinates).

    Args:
        unraveled_coords: Indexed tensor of coordinates.
        shape: Length of each dimension w.r.t. the grid.
        raveled_indices_: Expected tensor of coordinates.
    """
    raveled_indices = ravel_multi_index(unraveled_coords, shape)
    torch.testing.assert_allclose(raveled_indices, raveled_indices_)


@pytest.mark.parametrize(
    "raveled_indices, shape, unraveled_coords_",
    [
        pytest.param(
            torch.as_tensor([2]),
            [10],
            torch.as_tensor([[2]]),
        ),
        pytest.param(
            torch.as_tensor([2 * 10 + 3 * 1]),
            [10, 10],
            torch.as_tensor([[2, 3]]),
        ),
        pytest.param(
            torch.as_tensor([3 * (10 * 15) + 6 * 15 + 9 * 1]),
            [5, 10, 15],
            torch.as_tensor([[3, 6, 9]]),
        ),
    ],
)
def test_unravel_indices(
    raveled_indices: Tensor, shape: List[int], unraveled_coords_: Tensor
) -> None:
    """Unit test for converting raveled indices to an unraveled index."""
    unraveled_coords = unravel_index(raveled_indices, shape)
    torch.testing.assert_allclose(unraveled_coords, unraveled_coords_)
