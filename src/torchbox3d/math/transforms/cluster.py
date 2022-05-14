"""Voxelization transformation for a point cloud."""

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torchbox3d.math.neighbors import grid_cluster
from torchbox3d.math.ops.voxelize import Reduction
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.regular_grid import BEVGrid, VoxelGrid
from torchbox3d.structures.sparse_tensor import SparseTensor


@dataclass
class Voxelize:
    """Construct a voxelization transformation.

    Args:
        min_world_coordinates_m: (3,) Minimum range along the x,y,z axes in meters.
        max_world_coordinates_m: (3,) Maximum range along the x,y,z axes in meters.
        delta_m_per_cell: (3,) Ratio of meters to cell in meters.
        voxelization_type: Voxelization type used in the transformation
            (e.g., pooling).
    """

    min_world_coordinates_m: Tuple[float, float, float]
    max_world_coordinates_m: Tuple[float, float, float]
    delta_m_per_cell: Tuple[float, float, float]
    voxelization_type: Reduction

    @cached_property
    def voxel_grid(self) -> VoxelGrid:
        """Return the voxel grid associated with the transformation."""
        return VoxelGrid(
            min_world_coordinates_m=self.min_world_coordinates_m,
            max_world_coordinates_m=self.max_world_coordinates_m,
            delta_m_per_cell=self.delta_m_per_cell,
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

        indices, values, _, _ = grid_cluster(
            x.pos, x.x, self.voxel_grid, self.voxelization_type
        )
        x.voxels = SparseTensor(feats=values, coords=indices)
        return x


@dataclass
class Pillarize:
    """Construct a voxelization transformation.

    Args:
        min_world_coordinates_m: (2,) Minimum range along the x,y axes in meters.
        max_world_coordinates_m: (2,) Maximum range along the x,y axes in meters.
        delta_m_per_cell: (2,) Ratio of meters to cell in meters.
        voxelization_type: Voxelization type used in the transformation
            (e.g., pooling).
    """

    min_world_coordinates_m: Tuple[float, float]
    max_world_coordinates_m: Tuple[float, float]
    delta_m_per_cell: Tuple[float, float]
    voxelization_type: Reduction

    @cached_property
    def bev_grid(self) -> BEVGrid:
        """Return the voxel grid associated with the transformation."""
        return BEVGrid(
            min_world_coordinates_m=self.min_world_coordinates_m,
            max_world_coordinates_m=self.max_world_coordinates_m,
            delta_m_per_cell=self.delta_m_per_cell,
        )

    def __call__(self, x: RegularGridData) -> Data:
        """Voxelize the points in the data object.

        Args:
            x: Data object containing the points.

        Returns:
            The data with voxelized points.
        """
        x.grid = self.bev_grid
        x.x = torch.cat((x.pos, x.x), dim=-1)

        indices, values, _, _ = grid_cluster(
            x.pos, x.x, self.bev_grid, self.voxelization_type
        )
        x.voxels = SparseTensor(feats=values, coords=indices)
        return x
