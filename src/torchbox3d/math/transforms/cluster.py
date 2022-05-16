"""Voxelization transformation for a point cloud."""

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torchbox3d.math.ops.cluster import ClusterType
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.grid import BEVGrid, VoxelGrid
from torchbox3d.structures.sparse_tensor import SparseTensor


@dataclass(frozen=True)
class Voxelize:
    """Construct a voxelization transformation.

    Args:
        min_world_coordinates_m: (2,) Minimum world coordinates in meters.
        max_world_coordinates_m: (2,) Maximum world coordinates in meters.
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

    def __call__(self, grid_data: RegularGridData) -> Data:
        """Voxelize the points in the data object.

        Args:
            grid_data: Data object containing the points.

        Returns:
            The data with voxelized points.
        """
        values = torch.cat((grid_data.coordinates_m, grid_data.values), dim=-1)
        indices, mask = self.grid.convert_world_coordinates_to_grid(
            grid_data.coordinates_m
        )
        indices, values, counts = self.grid.cluster(
            indices[mask], values[mask], self.cluster_type
        )

        grid_data.grid = self.grid
        grid_data.values = values
        grid_data.cells = SparseTensor(
            values=values, indices=indices, counts=counts
        )
        return grid_data


@dataclass(frozen=True)
class Pillarize:
    """Construct a pillarization transformation.

    Args:
        min_world_coordinates_m: (2,) Minimum world coordinates in meters.
        max_world_coordinates_m: (2,) Maximum world coordinates in meters.
        delta_m_per_cell: (2,) Ratio of meters to cell in meters.
        cluster_type: Cluster type used in the transformation.
    """

    min_world_coordinates_m: Tuple[float, float]
    max_world_coordinates_m: Tuple[float, float]
    delta_m_per_cell: Tuple[float, float]
    cluster_type: ClusterType

    @cached_property
    def grid(self) -> BEVGrid:
        """Return the bird's-eye view grid."""
        return BEVGrid(
            min_world_coordinates_m=self.min_world_coordinates_m,
            max_world_coordinates_m=self.max_world_coordinates_m,
            delta_m_per_cell=self.delta_m_per_cell,
        )

    def __call__(self, grid_data: RegularGridData) -> Data:
        """Cluster the points in the data object.

        Args:
            grid_data: Grid data object.

        Returns:
            The clustered grid data.
        """
        values = torch.cat((grid_data.coordinates_m, grid_data.values), dim=-1)
        indices, mask = self.grid.convert_world_coordinates_to_grid(
            grid_data.coordinates_m
        )
        indices, values, counts = self.grid.cluster(
            indices[mask], values[mask], cluster_type=ClusterType.CONCATENATE
        )

        grid_data.grid = self.grid
        grid_data.values = values
        grid_data.cells = SparseTensor(
            values=values, indices=indices, counts=counts
        )
        return grid_data
