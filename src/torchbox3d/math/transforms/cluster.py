"""Voxelization transformation for a point cloud."""

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torchbox3d.math.ops.cluster import ClusterType
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.grid import BEVGrid, VoxelGrid
from torchbox3d.structures.sparse_tensor import SparseTensor


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
        values = torch.cat((x.coordinates_m, x.values), dim=-1)
        indices, mask = self.grid.transform_from(x.coordinates_m)
        indices, values, _ = self.grid.cluster(
            indices[mask], values[mask], self.cluster_type
        )

        x.grid = self.grid
        x.values = values
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
    def grid(self) -> BEVGrid:
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
        values = torch.cat((x.coordinates_m, x.values), dim=-1)
        indices, mask = self.grid.transform_from(x.coordinates_m)
        indices, values, counts = self.grid.cluster(
            indices[mask], values[mask], cluster_type=ClusterType.CONCATENATE
        )

        x.grid = self.grid
        x.values = values
        x.voxels = SparseTensor(values=values, indices=indices)
        return x
