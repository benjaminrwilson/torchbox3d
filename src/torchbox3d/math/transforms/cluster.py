"""Voxelization transformation for a point cloud."""

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torchbox3d.math.ops.cluster import ClusterType
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.sparse_tensor import SparseTensor
from torchbox3d.structuresgrid import BEVGrid, VoxelGrid


@dataclass
class Voxelize:
    """Construct a voxelization transformation.

    Args:
        min_world_coordinates_m: (3,) Minimum range along the x,y,z axes in meters.
        max_world_coordinates_m: (3,) Maximum range along the x,y,z axes in meters.
        delta_m_per_cell: (3,) Ratio of meters to cell in meters.
        cluster_type: Cluster type used in the transformation.
    """

    min_world_coordinates_m: Tuple[float, float, float]
    max_world_coordinates_m: Tuple[float, float, float]
    delta_m_per_cell: Tuple[float, float, float]
    cluster_type: ClusterType

    @cached_property
    def grid(self) -> VoxelGrid:
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
        x.grid = self.grid
        x.values = torch.cat((x.pos, x.values), dim=-1)

        indices, values, _ = x.grid.cluster(x.pos, x.values, self.cluster_type)
        x.voxels = SparseTensor(values=values, indices=indices)
        return x


@dataclass
class Pillarize:
    """Construct a pillarize transformation.

    Args:
        min_world_coordinates_m: (2,) Minimum range along the x,y axes in meters.
        max_world_coordinates_m: (2,) Maximum range along the x,y axes in meters.
        delta_m_per_cell: (2,) Ratio of meters to cell in meters.
        cluster_type: Cluster type used in the transformation
            (e.g., pooling).
    """

    min_world_coordinates_m: Tuple[float, float]
    max_world_coordinates_m: Tuple[float, float]
    delta_m_per_cell: Tuple[float, float]
    cluster_type: ClusterType

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
        x.values = torch.cat((x.pos, x.values), dim=-1)

        pos_cropped, mask = x.grid.crop_points(x.pos)
        indices, values, _ = x.grid.cluster(
            pos_cropped[..., :2], x.values[mask], self.cluster_type
        )
        x.voxels = SparseTensor(values=values, indices=indices)
        return x
