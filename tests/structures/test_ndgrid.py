"""Unit tests for the NDGrid class."""

from math import pi

import pytest
import torch
from torch import Tensor

from torchbox3d.structures.ndgrid import NDGrid


@pytest.mark.parametrize(
    "ndgrid,points,dims,range_m,scaled_points,quantized_points,grid_coords",
    [
        pytest.param(
            NDGrid(
                min_range_m=(-5.0, -5.0, -5.0),
                max_range_m=(+5.0, +5.0, +5.0),
                resolution_m_per_cell=(+0.1, +0.1, +0.1),
            ),
            torch.as_tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]]),
            (101, 101, 101),
            (10.0, 10.0, 10.0),
            torch.as_tensor([[11.1, 22.2, 33.3], [44.4, 55.5, 66.6]]),
            torch.as_tensor([[1, 2, 3], [4, 6, 7]]),
            torch.as_tensor([[61, 72, 83], [94, 106, 117]]),
        ),
        pytest.param(
            NDGrid(
                min_range_m=(-5.0, -5.0, -5.0),
                max_range_m=(+5.0, +5.0, +5.0),
                resolution_m_per_cell=(pi / 10, pi / 10, pi / 10),
            ),
            torch.as_tensor([[10.0, 10.0, 10.0]]),
            (33, 33, 33),
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
def test_ndgrid(
    ndgrid: NDGrid,
    points: Tensor,
    dims: Tensor,
    range_m: Tensor,
    scaled_points: Tensor,
    quantized_points: Tensor,
    grid_coords: Tensor,
) -> None:
    """Unit tests for the NDGrid class."""
    assert ndgrid.dims == dims
    assert ndgrid.range_m == range_m

    torch.testing.assert_allclose(ndgrid.scale_points(points), scaled_points)
    torch.testing.assert_allclose(
        ndgrid.quantize_points(points), quantized_points
    )
    torch.testing.assert_allclose(
        ndgrid.transform_to_grid_coordinates(points), grid_coords
    )
