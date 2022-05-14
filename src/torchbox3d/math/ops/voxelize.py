"""Nearest neighbor methods."""

from enum import Enum, unique
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torchbox3d.math.crop import crop_points
from torchbox3d.math.ops.index import ravel_multi_index, unravel_index
from torchbox3d.math.ops.pool import mean_pool
from torchbox3d.structures.regular_grid import RegularGrid


@unique
class Reduction(str, Enum):
    """The type of reduction performed during voxelization."""

    CONCATENATE = "CONCATENATE"
    MEAN_POOL = "MEAN_POOL"


@unique
class VoxelizationPoolingType(str, Enum):
    """The pooling method used for 'pooling' voxelization."""

    MEAN = "MEAN"


# @torch.jit.script
def voxelize_pool_kernel(
    xyz: Tensor,
    values: Tensor,
    voxel_grid: RegularGrid,
    pool_mode: Optional[str] = "mean",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster a point cloud into a grid of voxels.

    Args:
        xyz: (N,3) Coordinates (x,y,z).
        values: (N,F) Features associated with the points.
        voxel_grid: Voxel grid metadata.
        pool_mode: Pooling method for collisions.

    Returns:
        Voxel indices, values, counts, and cropping mask.
    """
    points_xyz, mask = crop_points(
        xyz,
        list(voxel_grid.min_world_coordinates_m),
        list(voxel_grid.max_world_coordinates_m),
    )
    values = values[mask]
    indices, mask = voxel_grid.transform_from(points_xyz)
    indices = indices[mask]

    counts = torch.ones_like(indices[:, 0])
    if pool_mode is not None and pool_mode == VoxelizationPoolingType.MEAN:
        indices, values, counts = mean_pool(
            indices, values, list(voxel_grid.grid_size)
        )
    return indices.int(), values, counts, mask


# @torch.jit.script
def voxelize_concatenate_kernel(
    pos: Tensor, values: Tensor, voxel_grid: RegularGrid, max_num_pts: int = 20
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Places a set of points in R^3 into a voxel grid.

    NOTE: This will not pool the points that fall into a voxel bin. Instead,
    this function will concatenate the points until they exceed a maximium
    size defined by max_num_pts.

    Args:
        pos: (N,3) Coordinates (x,y,z).
        values: (N,F) Features associated with the points.
        voxel_grid: Voxel grid metadata.
        max_num_pts: Max number of points per bin location.

    Returns:
        Voxel indices, values, counts, and cropping mask.
    """
    pos, roi_mask = crop_points(
        pos,
        list(voxel_grid.min_world_coordinates_m),
        list(voxel_grid.max_world_coordinates_m),
    )

    # Filter the values.
    values = values[roi_mask]

    indices, mask = voxel_grid.transform_from(pos)
    indices = indices[mask]
    raveled_indices = ravel_multi_index(indices, list(voxel_grid.grid_size))

    # Find indices which make bucket indices contiguous.
    permutation_sorted = torch.argsort(raveled_indices)
    indices = indices[permutation_sorted]
    raveled_indices = raveled_indices[permutation_sorted]
    values = values[permutation_sorted]

    # Compute unique values, inverse, and counts.
    out: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        raveled_indices, return_inverse=True, return_counts=True
    )

    output, inverse_indices, counts = out

    # Initialize vectors at each voxel (max_num_pts,F).
    # Instead of applying an information destroying reduction,
    # we concatenate the features until we reach a maximum size.
    voxelized_values = torch.zeros(
        (len(output), max_num_pts, values.shape[-1])
    )

    # Concatenating collisions requires counting how many collisions there are.
    # This computes offsets for all of the collisions in a vectorized fashion.
    offset = F.pad(counts, pad=[1, 0], mode="constant", value=0.0).cumsum(
        dim=0
    )[inverse_indices]

    index = torch.arange(0, len(inverse_indices)) - offset
    out_inv: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        inverse_indices,
        return_inverse=True,
        return_counts=True,
    )
    offset, _, _ = out_inv
    is_valid = index < max_num_pts

    inverse_indices = inverse_indices[is_valid]
    index = index[is_valid].long()
    voxelized_values[inverse_indices, index] = values[is_valid]

    inv_perm = torch.argsort(permutation_sorted[offset])
    voxelized_indices = unravel_index(output, list(voxel_grid.grid_size)).int()
    voxelized_indices = voxelized_indices[inv_perm]
    voxelized_values = voxelized_values[inv_perm]
    counts = counts[inv_perm]
    roi_mask = roi_mask[inv_perm]
    return voxelized_indices, voxelized_values, counts, roi_mask
