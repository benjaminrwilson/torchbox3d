"""Pooling methods."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torchbox3d.math.ops.index import ravel_multi_index, unravel_index


@torch.jit.script
def mean_pool(
    indices: Tensor, values: Tensor, size: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a pooling operation on a voxel grid.

    Args:
        indices: (N,3) Voxel coordinates.
        values: (N,F) Voxel features.
        size: (3,) Length, width, and height of the voxel grid.

    Returns:
        The binned voxel coordinates, features, and counts after pooling.
    """
    raveled_indices = ravel_multi_index(indices, size)
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
    unraveled_coords = unravel_index(output, size)

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
