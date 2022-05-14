"""Unit tests for the RegularGrid class."""

from math import pi

import pytest
import torch
from torch import Tensor

from torchbox3d.structures.regular_grid import RegularGrid


@pytest.mark.parametrize(
    "RegularGrid,points,dims,range_m,scaled_points,quantized_points,grid_coords",
    [
        pytest.param(
            RegularGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.1),
            ),
            torch.as_tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]]),
            (100, 100, 100),
            (10.0, 10.0, 10.0),
            torch.as_tensor([[11.1, 22.2, 33.3], [44.4, 55.5, 66.6]]),
            torch.as_tensor([[1, 2, 3], [4, 6, 7]]),
            torch.as_tensor([[61, 72, 83], [94, 106, 117]]),
        ),
        pytest.param(
            RegularGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(pi / 10, pi / 10, pi / 10),
            ),
            torch.as_tensor([[10.0, 10.0, 10.0]]),
            (32, 32, 32),
            (10.0, 10.0, 10.0),
            torch.as_tensor([[10.0, 10.0, 10.0]]) / (pi / 10),
            torch.as_tensor([[10, 10, 10]]),
            (
                ((torch.as_tensor([[10.0, 10.0, 10.0]]) + 5) / (pi / 10)) + 0.5
            ).floor(),
        ),
    ],
    ids=["3d_0", "3d_odd"],
)
def test_RegularGrid(
    RegularGrid: RegularGrid,
    points: Tensor,
    dims: Tensor,
    range_m: Tensor,
    scaled_points: Tensor,
    quantized_points: Tensor,
    grid_coords: Tensor,
) -> None:
    """Unit tests for the RegularGrid class."""
    assert RegularGrid.dims == dims
    assert RegularGrid.range_m == range_m

    torch.testing.assert_allclose(
        RegularGrid.scale_points(points), scaled_points
    )
    torch.testing.assert_allclose(
        RegularGrid.quantize_points(points), quantized_points
    )

    torch.testing.assert_allclose(
        RegularGrid.transform_to_grid_coordinates(points)[0], grid_coords
    )
