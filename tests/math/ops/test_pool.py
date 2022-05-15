"""Unit tests for the pooling subpackage."""

from typing import Dict

import pytest
import torch
from torch import Tensor

from torchbox3d.math.ops.cluster import ClusterType, cluster_grid
from torchbox3d.math.ops.index import unique_indices


def _construct_dummy_voxels() -> Dict[str, Tensor]:
    """Construct dummy voxels."""
    indices = torch.as_tensor(
        [[0, 0, 0], [0, 0, 0], [5, 5, 5], [5, 5, 5]], dtype=torch.int64
    )
    values = torch.as_tensor(
        [[0.0, 0.0, 0.0], [2.2, 2.2, 2.2], [5.3, 5.3, 5.3], [5.4, 5.4, 5.4]],
        dtype=torch.float32,
    )
    return {"indices": indices, "values": values}


def test_voxel_pool() -> None:
    """Test voxel pooling method."""
    voxels = _construct_dummy_voxels()

    # Grab indices and values.
    indices = voxels["indices"]
    values = voxels["values"]

    # Declare voxel grid size.
    size = [6, 6, 6]

    # Apply mean pooling.
    indices, values, _ = cluster_grid(
        indices, values, size, cluster_type=ClusterType.MEAN
    )

    # Declare expected indices and values.
    indices_expected = torch.as_tensor(
        [[0, 0, 0], [5, 5, 5]], dtype=torch.int64
    )
    values_expected = torch.as_tensor(
        [[1.1, 1.1, 1.1], [5.35, 5.35, 5.35]], dtype=torch.float32
    )

    # Check if all indices and values match.
    torch.testing.assert_allclose(indices_expected, indices)
    torch.testing.assert_allclose(values_expected, values)


@pytest.mark.parametrize(
    "indices, indices_, inv_",
    [
        (
            torch.as_tensor(
                [
                    [5, 5, 0],  # 0
                    [1, 2, 3],  # 1
                    [1, 2, 4],  # 2
                    [0, 0, 0],  # 3
                    [5, 5, 0],
                    [5, 5, 0],
                    [7, 7, 1],  # 5
                ]
            ),
            torch.as_tensor(
                [
                    [5, 5, 0],  # 0
                    [1, 2, 3],  # 1
                    [1, 2, 4],  # 2
                    [0, 0, 0],  # 3
                    [7, 7, 1],  # 5
                ],
                dtype=torch.float,
            ),
            torch.as_tensor(
                [0, 1, 2, 3, 6],
                dtype=torch.float,
            ),
        ),
        (
            torch.as_tensor(
                [
                    [5, 15, 0],  # 0
                    [5, 15, 0],  # 1
                    [4, 15, 0],  # 2
                    [2, 5, 0],  # 3
                    [1, 5, 0],
                    [5, 5, 0],
                    [5, 15, 0],  # 5
                ]
            ),
            torch.as_tensor(
                [
                    [5, 15, 0],  # 0
                    [4, 15, 0],  # 1
                    [2, 5, 0],  # 3
                    [1, 5, 0],
                    [5, 5, 0],
                ],
                dtype=torch.float,
            ),
            torch.as_tensor(
                [0, 2, 3, 4, 5],
                dtype=torch.float,
            ),
        ),
        (
            torch.as_tensor(
                [
                    [0, 0, 0],  # 0
                    [0, 0, 0],  # 1
                    [1, 1, 1],  # 2
                    [1, 1, 1],  # 3
                    [0, 0, 0],
                    [1, 1, 1],
                    [10, 10, 10],  # 5
                    [0, 0, 0],  # 5
                ]
            ),
            torch.as_tensor(
                [
                    [0, 0, 0],  # 0
                    [1, 1, 1],  # 2
                    [10, 10, 10],  # 5
                ],
                dtype=torch.float,
            ),
            torch.as_tensor(
                [0, 2, 6],
                dtype=torch.float,
            ),
        ),
    ],
    ids=["test1", "test2", "test3"],
)
def test_unique_indices(
    indices: Tensor, indices_: Tensor, inv_: Tensor
) -> None:
    """Unit test for computing unique indices of a tensor."""
    inv = unique_indices(indices)
    torch.testing.assert_allclose(indices[inv], indices_)
    torch.testing.assert_allclose(inv, inv_)
