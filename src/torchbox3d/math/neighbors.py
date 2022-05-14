"""Nearest neighbor methods."""

from typing import Tuple

from torch import Tensor

from torchbox3d.math.ops.voxelize import (
    Reduction,
    voxelize_concatenate_kernel,
    voxelize_pool_kernel,
)
from torchbox3d.structures.regular_grid import RegularGrid


def grid_cluster(
    indices: Tensor,
    values: Tensor,
    grid: RegularGrid,
    reduction: Reduction = Reduction.MEAN_POOL,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster a point cloud into a grid of voxels.

    Args:
        indices: (N,3) Spatial indices.
        values: (N,F) Features associated with the points.
        grid: Voxel grid metadata.
        reduction: The reduction applied after clustering.

    Returns:
        The voxel indices, values, counts, and cropping mask.

    Raises:
        NotImplementedError: If the voxelization mode is not implemented.
    """
    if reduction.upper() == Reduction.MEAN_POOL:
        return voxelize_pool_kernel(
            indices,
            values,
            grid,
        )
    elif reduction.upper() == Reduction.CONCATENATE:
        return voxelize_concatenate_kernel(indices, values, grid)
    else:
        raise NotImplementedError(
            f"The reduction, {reduction}, is not implemented!"
        )
