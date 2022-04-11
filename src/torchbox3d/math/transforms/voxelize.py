"""Voxelization transformation for a point cloud."""

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torchbox3d.math.neighbors import voxelize
from torchbox3d.math.ops.voxelize import VoxelizationType
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.ndgrid import VoxelGrid
from torchbox3d.structures.sparse_tensor import SparseTensor


@dataclass
class Voxelize:
    """Construct a voxelization transformation.

    Args:
        min_range_m: (3,) Minimum range along the x,y,z axes in meters.
        max_range_m: (3,) Maximum range along the x,y,z axes in meters.
        resolution_m_per_cell: (3,) Ratio of meters to cell in meters.
        voxelization_type: Voxelization type used in the transformation
            (e.g., pooling).
    """

    min_range_m: Tuple[float, float, float]
    max_range_m: Tuple[float, float, float]
    resolution_m_per_cell: Tuple[float, float, float]
    voxelization_type: VoxelizationType

    @cached_property
    def voxel_grid(self) -> VoxelGrid:
        """Return the voxel grid associated with the transformation."""
        return VoxelGrid(
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            resolution_m_per_cell=self.resolution_m_per_cell,
        )

    def __call__(self, x: RegularGridData) -> Data:
        """Voxelize the points in the data object.

        Args:
            x: Data object containing the points.

        Returns:
            The data with voxelized points.
        """
        x.grid = self.voxel_grid
        x.x = torch.cat((x.pos, x.x), dim=-1)

        indices, values, _, _ = voxelize(
            x.pos, x.x, self.voxel_grid, self.voxelization_type
        )
        x.voxels = SparseTensor(feats=values, coords=indices)
        return x
