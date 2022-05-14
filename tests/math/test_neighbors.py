"""Unit tests for the neighbors subpackage."""

from typing import Any, Callable

import pytest
import torch
from torch import Tensor

from torchbox3d.math.neighbors import voxelize
from torchbox3d.math.ops.cluster import (
    VoxelizationType,
    voxelize_concatenate_kernel,
)
from torchbox3d.structuresgrid import BEVGrid, VoxelGrid


@pytest.mark.parametrize(
    "voxel_grid, points_xyz, indices_, values_, counts_",
    [
        pytest.param(
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
            torch.as_tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.05, 0.05, 0.05],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.08, 0.0],
                ]
            ),
            torch.as_tensor(
                [
                    [
                        50,
                        50,
                        25,
                    ],  # (0.00, 0.00, 0.00) / (0.1, 0.1, 0.2) + (50, 50, 25)
                    [
                        51,
                        51,
                        25,
                    ],  # (0.05, 0.05, 0.05) / (0.1, 0.1, 0.2) + (50, 50, 25)
                    [
                        50,
                        60,
                        25,
                    ],  # (0.00, 1.00, 0.00) / (0.1, 0.1, 0.2) + (50, 50, 25)
                    [
                        50,
                        61,
                        25,
                    ],  # (0.00, 1.08, 0.00) / (0.1, 0.1, 0.2) + (50, 50, 25)
                ],
                dtype=torch.int32,
            ),  # 50 + 101 - 50
            torch.as_tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.05, 0.05, 0.05],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.08, 0.0],
                ]
            ),
            torch.as_tensor([1, 1, 1, 1]),
        ),
        pytest.param(
            VoxelGrid(
                min_world_coordinates_m=(0.0, 0.0, 0.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+2.5, +2.5, +2.5),
            ),
            torch.as_tensor(
                [
                    [-1.0, +0.0, +0.0],  # Cropped
                    [
                        +0.0,
                        +0.0,
                        +0.0,
                    ],
                    [
                        +2.0,
                        +2.0,
                        +2.0,
                    ],
                    [+2.5, +2.5, +2.5],
                ]
            ),
            torch.as_tensor(
                [
                    [
                        0,
                        0,
                        0,
                    ],  # (0.0, 0.0, 0.0) / (2.5, 2.5, 2.5) + (1, 1, 1)
                    [1, 1, 1],  # (2.5, 2.5, 2.5) / (2.5, 2.5, 2.5) + (0, 0, 0)
                ],
                dtype=torch.int32,
            ),
            torch.as_tensor([[0.0, 0.0, 0.0], [2.25, 2.25, 2.25]]),
            torch.as_tensor([1, 2]),
        ),
    ],
    ids=["test0", "test1"],
)
def test_voxelize(
    voxel_grid: VoxelGrid,
    points_xyz: Tensor,
    indices_: Tensor,
    values_: Tensor,
    counts_: Tensor,
) -> None:
    """Test voxelizing a point cloud."""
    indices, values, counts, _ = voxelize(points_xyz, points_xyz, voxel_grid)
    torch.testing.assert_allclose(indices, indices_)
    torch.testing.assert_allclose(values, values_)
    torch.testing.assert_allclose(counts, counts_)


@pytest.mark.parametrize(
    "voxel_grid, points_m, indices_, values_, counts_",
    [
        pytest.param(
            # voxel_grid
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
            # points_m
            torch.as_tensor(
                [
                    [1.55, 0.0, 0.0],
                    [0, 0, 0],
                    [0.5, 0.5, 0.5],
                    [0.55, 0.55, 0.55],
                    [1.5, 0.0, 0.0],
                    [2.1, 2.1, 2.0],
                ]
            ),
            # indices_
            torch.as_tensor(
                [
                    [
                        50 + 16,
                        50,
                        25,
                    ],
                    [
                        50,
                        50,
                        25,
                    ],
                    [
                        55,
                        55,
                        28,
                    ],
                    [
                        56,
                        56,
                        28,
                    ],
                    [
                        65,
                        50,
                        25,
                    ],
                    [
                        71,
                        71,
                        35,
                    ],
                ],
                dtype=torch.int32,
            ),
            # values_
            torch.as_tensor(
                [
                    [[1.5500, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[0.5000, 0.5000, 0.5000], [0.0000, 0.0000, 0.0000]],
                    [[0.5500, 0.5500, 0.5500], [0.0000, 0.0000, 0.0000]],
                    [[1.5000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[2.1000, 2.1000, 2.0000], [0.0000, 0.0000, 0.0000]],
                ]
            ),
            # counts_
            torch.as_tensor([1, 1, 1, 1, 1, 1]),
        ),
        # Test 2 parameters.
        pytest.param(
            BEVGrid(
                min_world_coordinates_m=(-5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1),
            ),
            # points_m
            torch.as_tensor(
                [
                    [1.55, 0.0, 0.0],
                    [0, 0, 0],
                    [0.5, 0.5, 0.5],
                    [0.55, 0.55, 0.55],
                    [1.5, 0.0, 0.0],
                    [2.1, 2.1, 2.0],
                ]
            ),
            # indices_
            # indices_
            torch.as_tensor(
                [
                    [
                        50 + 16,
                        50,
                    ],
                    [
                        50,
                        50,
                    ],
                    [
                        55,
                        55,
                    ],
                    [
                        56,
                        56,
                    ],
                    [
                        65,
                        50,
                    ],
                    [
                        71,
                        71,
                    ],
                ],
                dtype=torch.int32,
            ),
            # values_
            torch.as_tensor(
                [
                    [[1.5500, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[0.5000, 0.5000, 0.5000], [0.0000, 0.0000, 0.0000]],
                    [[0.5500, 0.5500, 0.5500], [0.0000, 0.0000, 0.0000]],
                    [[1.5000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
                    [[2.1000, 2.1000, 2.0000], [0.0000, 0.0000, 0.0000]],
                ]
            ),
            # counts_
            torch.as_tensor([1, 1, 1, 1, 1, 1]),
        ),
    ],
    ids=[
        "Test clustering points into voxels.",
        "Test clustering points into pillars.",
    ],
)
def test_voxelize_concatenate_kernel(
    voxel_grid: VoxelGrid,
    points_m: Tensor,
    indices_: Tensor,
    values_: Tensor,
    counts_: Tensor,
) -> None:
    """Test voxelization with truncated reduction."""
    indices, values, counts, _ = voxelize_concatenate_kernel(
        points_m,
        points_m,
        voxel_grid,
        max_num_pts=2,
    )
    torch.testing.assert_allclose(indices, indices_)
    torch.testing.assert_allclose(values, values_)
    torch.testing.assert_allclose(counts, counts_)


@pytest.mark.parametrize(
    "points_xyz, voxel_grid",
    [
        pytest.param(
            torch.rand((100000, 3)),
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
        )
    ],
    ids=["Benchmark the voxelization concatenate op with 100,000 points."],
)
def test_benchmark_voxelize_concatenate(
    benchmark: Callable[..., Any], points_xyz: Tensor, voxel_grid: VoxelGrid
) -> None:
    """Benchmark concatenate kernel with 100k points."""
    benchmark(
        voxelize,
        points_xyz,
        points_xyz,
        voxel_grid,
        cluster_type=VoxelizationType.CONCATENATE,
    )


@pytest.mark.parametrize(
    "points_xyz, voxel_grid",
    [
        pytest.param(
            torch.rand((100000, 3)),
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
        ),
        pytest.param(
            torch.rand((500000, 3)),
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
        ),
    ],
    ids=[
        "Benchmark voxelization with mean pooling on 100,000 points.",
        "Benchmark voxelization with mean pooling on 500,000 points.",
    ],
)
def test_benchmark_voxelize_pool(
    benchmark: Callable[..., Any], points_xyz: Tensor, voxel_grid: VoxelGrid
) -> None:
    """Benchmark mean-pooling voxelization on 100k points."""
    benchmark(
        voxelize,
        points_xyz,
        points_xyz,
        voxel_grid,
        cluster_type=VoxelizationType.POOL,
    )
