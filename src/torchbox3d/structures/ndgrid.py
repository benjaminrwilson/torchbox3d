"""An N-Dimensional grid class."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import torch
from torch import Tensor

from torchbox3d.math.crop import crop_points
from torchbox3d.rendering.ops.shaders import align_corners


@dataclass
class NDGrid:
    """Models an N-dimensional grid.

    Args:
        min_range_m: (N,) Minimum coordinates (in meters).
        max_range_m: (N,) Maximum coordinates (in meters).
        resolution_m_per_cell: (N,) Ratio of meters to cell in each dimension.
    """

    min_range_m: Tuple[float, ...]
    max_range_m: Tuple[float, ...]
    resolution_m_per_cell: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the NDGrid sizes."""
        D_min = len(self.min_dim_m)
        D_max = len(self.max_range_m)
        D_res = len(self.resolution_m_per_cell)

        if D_min != D_max and D_max != D_res:
            raise ValueError(
                "`min_range_m`, `max_range_m` and `resolution_m_per_cell` "
                "must have the same dimension!"
            )

    @cached_property
    def N(self) -> int:
        """Return the dimension of the grid."""
        return len(self.min_range_m)

    @cached_property
    def dims(self) -> Tuple[int, ...]:
        """Size of the grid _after_ bucketing."""
        range_m: Tensor = torch.as_tensor(self.range_m)
        dims: List[int] = self.scale_and_quantize_points(range_m).tolist()
        return tuple(dims)

    @cached_property
    def min_dim_m(self) -> Tuple[int, ...]:
        """Size of the grid _after_ bucketing."""
        min_range_m: Tensor = torch.as_tensor(self.min_range_m)
        dims: List[int] = self.scale_and_quantize_points(min_range_m).tolist()
        return tuple(dims)

    @cached_property
    def max_dim_m(self) -> Tuple[int, ...]:
        """Size of the grid _after_ bucketing."""
        max_range_m: Tensor = torch.as_tensor(self.max_range_m)
        dims: List[int] = self.scale_and_quantize_points(max_range_m).tolist()
        return tuple(dims)

    @cached_property
    def range_m(self) -> Tuple[float, ...]:
        """Size of the grid _before_ bucketing."""
        min_range_m = torch.as_tensor(self.min_range_m)
        max_range_m = torch.as_tensor(self.max_range_m)
        range_m: List[float] = (max_range_m - min_range_m).tolist()
        return tuple(range_m)

    def scale_points(self, points: Tensor) -> Tensor:
        """Scale points by the (1/`resolution_m_per_cell`).

        Args:
            points: (N,D) list of points.

        Returns:
            (N,D) list of scaled points.
        """
        N = min(self.N, points.shape[-1])
        resolution_m_per_cell = torch.as_tensor(
            self.resolution_m_per_cell,
            device=points.device,
            dtype=torch.float,
        )
        scaled_points: Tensor = points[..., :N] / resolution_m_per_cell[:N]
        return scaled_points

    def quantize_points(self, points: Tensor) -> Tensor:
        """Quantize the points to integer coordinates.

        Args:
            points: (N,D) List of points.

        Returns:
            (N,D) List of quantized points.
        """
        # Add half-bucket offset.
        centered_points: Tensor = align_corners(points)
        quantized_points: Tensor = centered_points.floor().long()
        return quantized_points

    def scale_and_quantize_points(self, points_m: Tensor) -> Tensor:
        """Scale _and_ quantize the points.

        Args:
            points_m: (N,D) Points in meters.

        Returns:
            The scaled, quantized points.
        """
        scaled_points = self.scale_points(points_m)
        quantized_points: Tensor = self.quantize_points(scaled_points)
        return quantized_points

    def transform_to_grid_coordinates(
        self, points_m: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Transform points to grid coordinates (in meters).

        Args:
            points_m: (N,D) list of points.

        Returns:
            (N,D) list of quantized grid coordinates.
        """
        D = min(points_m.shape[-1], len(self.min_range_m))
        offset_m = torch.zeros_like(points_m[0])
        offset_m[:D] = torch.as_tensor(self.min_range_m[:D]).abs()

        quantized_points_grid = self.scale_and_quantize_points(
            points_m + offset_m
        )

        upper = [float(x) for x in self.dims]
        _, mask = crop_points(quantized_points_grid, [0.0, 0.0, 0.0], upper)
        return quantized_points_grid, mask

    def downsample(self, stride: int) -> Tuple[int, int, int]:
        """Downsample the grid coordinates."""
        downsampled_dims = [int(d / stride) for d in self.dims]
        return tuple(downsampled_dims)

    @cached_property
    def grid_offset_m(self) -> Tuple[float, float, float]:
        """Return the grid offset from the lower bound to the grid origin."""
        min_range_m = [abs(x) for x in self.min_range_m]
        return tuple(min_range_m)


@dataclass
class VoxelGrid(NDGrid):
    """Representation of a voxel grid."""

    min_range_m: Tuple[float, float, float]
    max_range_m: Tuple[float, float, float]
    resolution_m_per_cell: Tuple[float, float, float]


@dataclass
class BEVGrid(NDGrid):
    """Representation of a bird's-eye view grid."""

    min_range_m: Tuple[float, float]
    max_range_m: Tuple[float, float]
    resolution_m_per_cell: Tuple[float, float]
