"""Nearest neighbor methods."""

from enum import Enum, unique
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torchbox3d.math.ops.index import ravel_multi_index, unravel_index


@unique
class ClusterType(str, Enum):
    """The type of reduction performed during voxelization."""

    CONCATENATE = "CONCATENATE"
    MEAN = "MEAN"


def mean_cluster_grid(
    indices: Tensor,
    values: Tensor,
    grid_size: List[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    indices, values, counts = _mean_cluster_grid_kernel(
        indices, values, grid_size
    )
    return indices, values, counts


def concatenate_cluster_grid(
    indices: Tensor,
    values: Tensor,
    grid_size: List[int],
    max_num_values: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    indices, values, counts = _concatenate_cluster_grid_kernel(
        indices,
        values,
        grid_size,
        max_num_values=max_num_values,
    )
    return indices, values, counts


@torch.jit.script
def _mean_cluster_grid_kernel(
    indices: Tensor, values: Tensor, grid_size: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a pooling operation on a voxel grid.

    Args:
        indices: (N,3) Spatial indices.
        values: (N,F) Spatial values.
        size: (3,) Length, width, and height of the grid.

    Returns:
        The binned voxel coordinates, features, and counts after pooling.
    """
    raveled_indices = ravel_multi_index(indices, grid_size)
    permutation_sorted = torch.argsort(raveled_indices)

    raveled_indices = raveled_indices[permutation_sorted]
    values = values[permutation_sorted]

    # Compute unique values, inverse, and counts.
    out: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        raveled_indices, return_inverse=True, return_counts=True
    )
    output, inverse_indices, counts = out

    # Scatter with sum reduction.
    last_index_exclusive = int(inverse_indices.max()) + 1
    pooled_values: Tensor = torch.zeros_like(values).scatter_add_(
        dim=0,
        index=inverse_indices[:, None].repeat(1, values.shape[-1]),
        src=values,
    )[:last_index_exclusive]
    pooled_values /= counts[:, None]

    # Unravel unique linear indices into unique multi indices.
    unraveled_coords = unravel_index(output, grid_size)

    out_inv: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        inverse_indices,
        return_inverse=True,
        return_counts=True,
    )
    offset, _, counts = out_inv
    offset += F.pad(
        counts[:-1] - 1, pad=[1, 0], mode="constant", value=0.0
    ).cumsum(dim=0)

    # Respect original ordering.
    inverse_indices = torch.argsort(permutation_sorted[offset])
    return (
        unraveled_coords[inverse_indices],
        pooled_values[inverse_indices],
        counts[inverse_indices],
    )


@torch.jit.script
def _concatenate_cluster_grid_kernel(
    indices: Tensor,
    values: Tensor,
    grid_size: List[int],
    max_num_values: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Places a set of points in R^3 into a voxel grid.

    NOTE: This will not pool the points that fall into a voxel bin. Instead,
    this function will concatenate the points until they exceed a maximium
    size defined by max_num_pts.

    Args:
        indices: (N,3) Spatial indices.
        values: (N,F) Spatial values.
        size: (3,) Length, width, and height of the grid.
        max_num_values: Max number of values per index location.

    Returns:
        Voxel indices, values, counts, and cropping mask.
    """
    raveled_indices = ravel_multi_index(indices, grid_size)

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
        (len(output), max_num_values, values.shape[-1])
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
    is_valid = index < max_num_values

    inverse_indices = inverse_indices[is_valid]
    index = index[is_valid].long()
    voxelized_values[inverse_indices, index] = values[is_valid]

    inv_perm = torch.argsort(permutation_sorted[offset])
    voxelized_indices = unravel_index(output, grid_size).int()
    voxelized_indices = voxelized_indices[inv_perm]
    voxelized_values = voxelized_values[inv_perm]
    counts = counts[inv_perm]
    return voxelized_indices, voxelized_values, counts


def unique_indices(indices: Tensor, dim: int = 0) -> Tensor:
    """Compute the indices corresponding to the unique value.

    Args:
        indices: (N,K) Coordinate inputs.
        dim: Dimension to compute unique operation over.

    Returns:
        The indices corresponding to the selected values.
    """
    out: Tuple[Tensor, Tensor] = torch.unique(
        indices, return_inverse=True, dim=dim
    )
    unique, inverse = out
    perm = torch.arange(
        inverse.size(dim), dtype=inverse.dtype, device=inverse.device
    )
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    inv = inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
    inv, _ = inv.sort()
    return inv
