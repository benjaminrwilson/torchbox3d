"""Nearest neighbor methods."""

from typing import Optional, Tuple

from torch import Tensor

from torchbox3d.math.ops.voxelize import (
    VoxelizationPoolingType,
    VoxelizationType,
    voxelize_concatenate_kernel,
    voxelize_pool_kernel,
)
from torchbox3d.structures.ndgrid import VoxelGrid


def voxelize(
    points_xyz: Tensor,
    values: Tensor,
    voxel_grid: VoxelGrid,
    voxelization_type: VoxelizationType = VoxelizationType.POOL,
    voxelization_pooling_type: Optional[
        VoxelizationPoolingType
    ] = VoxelizationPoolingType.MEAN,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster a point cloud into a grid of voxels.

    Args:
        points_xyz: (N,3) Coordinates (x,y,z).
        values: (N,F) Features associated with the points.
        voxel_grid: Voxel grid metadata.
        voxelization_type: The voxelization mode (e.g., "pool").
        voxelization_pooling_type: Pooling method for collisions.

    Returns:
        The voxel indices, values, counts, and cropping mask.

    Raises:
        NotImplementedError: If the voxelization mode is not implemented.
    """
    if voxelization_type.upper() == VoxelizationType.POOL:
        return voxelize_pool_kernel(
            points_xyz,
            values,
            voxel_grid,
            voxelization_pooling_type,
        )
    elif voxelization_type.upper() == VoxelizationType.CONCATENATE:
        return voxelize_concatenate_kernel(points_xyz, values, voxel_grid)
    else:
        raise NotImplementedError(
            f"The voxelization type {voxelization_type} is not implemented!"
        )
