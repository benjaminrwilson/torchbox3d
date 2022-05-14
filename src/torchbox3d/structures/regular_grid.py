"""An N-Dimensional grid class."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import torch
from torch import Tensor

from torchbox3d.math.crop import crop_points
from torchbox3d.math.ops.cluster import (
    ClusterType,
    concatenate_cluster,
    mean_cluster,
)
from torchbox3d.rendering.ops.shaders import align_corners


@dataclass
class RegularGrid:
    """Models an N-dimensional grid.

    Reference: https://en.wikipedia.org/wiki/Regular_grid

    Args:
        min_world_coordinates_m: (N,) Minimum world coordinates (in meters).
        max_world_coordinates_m: (N,) Maximum world coordinates (in meters).
        delta_m_per_cell: (N,) Ratio of meters to cell in each dimension.
    """

    min_world_coordinates_m: Tuple[float, ...]
    max_world_coordinates_m: Tuple[float, ...]
    delta_m_per_cell: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the NDGrid sizes."""
        D_min = len(self.min_world_coordinates_m)
        D_max = len(self.max_world_coordinates_m)
        D_res = len(self.delta_m_per_cell)

        if D_min != D_max and D_max != D_res:
            raise ValueError(
                "`min_world_coordinates_m`, `max_world_coordinates_m` and `delta_m_per_cell` "
                "must have the same dimension!"
            )

    @cached_property
    def N(self) -> int:
        """Return the dimension of the grid."""
        return len(self.min_world_coordinates_m)

    @cached_property
    def grid_size(self) -> Tuple[int, ...]:
        """Size of the grid _after_ bucketing."""
        min_world_coordinates_m = torch.as_tensor(self.min_world_coordinates_m)
        max_world_coordinates_m = torch.as_tensor(self.max_world_coordinates_m)
        range_m = max_world_coordinates_m - min_world_coordinates_m
        dims: List[int] = self.scale_and_quantize_points(range_m).tolist()
        return tuple(dims)

    def quantize_points(self, points_m: Tensor) -> Tensor:
        """Quantize the points to integer coordinates.

        Args:
            points_m: (N,D) List of points in meters.

        Returns:
            (N,D) List of quantized points.
        """
        # Add half-bucket offset.
        centered_points_m: Tensor = align_corners(points_m)
        quantized_points_m: Tensor = centered_points_m.floor().long()
        return quantized_points_m

    def scale_and_quantize_points(self, points_m: Tensor) -> Tensor:
        """Scale _and_ quantize the points.

        Args:
            points_m: (N,D) Points in meters.

        Returns:
            The scaled, quantized points.
        """
        N = min(self.N, points_m.shape[-1])
        delta_m_per_cell = torch.as_tensor(
            self.delta_m_per_cell,
            device=points_m.device,
            dtype=torch.float,
        )
        scaled_points: Tensor = points_m[..., :N] / delta_m_per_cell[:N]
        quantized_points: Tensor = self.quantize_points(scaled_points)
        return quantized_points

    def transform_from(self, points_m: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform points from world coordinates to grid coordinates (in meters).

        Args:
            points_m: (N,D) list of points.

        Returns:
            (N,D) list of quantized grid coordinates.
        """
        D = min(points_m.shape[-1], len(self.min_world_coordinates_m))
        offset_m = torch.zeros_like(points_m[0])
        offset_m[:D] = torch.as_tensor(self.min_world_coordinates_m[:D]).abs()

        quantized_points_grid = self.scale_and_quantize_points(
            points_m + offset_m
        )

        upper = [float(x) for x in self.grid_size]
        _, mask = crop_points(quantized_points_grid, [0.0, 0.0, 0.0], upper)
        return quantized_points_grid, mask

    def downsample(self, stride: int) -> Tuple[int, ...]:
        """Downsample the grid coordinates."""
        downsampled_dims = [int(d / stride) for d in self.grid_size]
        return tuple(downsampled_dims)

    @cached_property
    def grid_offset_m(self) -> Tuple[float, ...]:
        """Return the grid offset from the lower bound to the grid origin."""
        min_world_coordinates_m = [
            abs(x) for x in self.min_world_coordinates_m
        ]
        return tuple(min_world_coordinates_m)

    def cluster(
        self,
        pos: Tensor,
        values: Tensor,
        reduction: ClusterType = ClusterType.MEAN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        if reduction.upper() == ClusterType.MEAN:
            return mean_cluster(
                pos,
                values,
                list(self.grid_size),
            )
        elif reduction.upper() == ClusterType.CONCATENATE:
            return concatenate_cluster(pos, values, list(self.grid_size))
        else:
            raise NotImplementedError(
                f"The reduction, {reduction}, is not implemented!"
            )

    def crop_points(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        points, mask = crop_points(
            points,
            list(self.min_world_coordinates_m),
            list(self.max_world_coordinates_m),
        )
        return points, mask


@dataclass
class Grid:
    vertices: Tensor
    edges: Tensor


@dataclass
class BEVGrid(RegularGrid):
    """Representation of a bird's-eye view grid."""

    min_world_coordinates_m: Tuple[float, float]
    max_world_coordinates_m: Tuple[float, float]
    delta_m_per_cell: Tuple[float, float]


@dataclass
class VoxelGrid(RegularGrid):
    """Representation of a voxel grid."""

    min_world_coordinates_m: Tuple[float, float, float]
    max_world_coordinates_m: Tuple[float, float, float]
    delta_m_per_cell: Tuple[float, float, float]
