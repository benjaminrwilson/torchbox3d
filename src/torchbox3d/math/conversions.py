"""Geometric conversions."""


from typing import List, Tuple, Union

import torch
from torch import Tensor

from torchbox3d.math.crop import crop_coordinates
from torchbox3d.math.ops.cluster import ClusterType, cluster_grid


@torch.jit.script
def denormalize_pixel_intensities(tensor: Tensor) -> Tensor:
    """Map intensities from [0,1] -> [0,255].

    Args:
        tensor: (...,K) K-channel tensor image.

    Returns:
        (...,K) The uint8 tensor image.
    """
    tensor = tensor.clamp(0, 1)
    tensor *= 255.0
    tensor = tensor.byte()
    return tensor


@torch.jit.script
def cart_to_sph(cart_xyz: Tensor) -> Tensor:
    """Convert Cartesian coordinates to spherical coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        cart_xyz: Cartesian coordinates (x,y,z).

    Returns:
        (N,3) Spherical coordinates (azimuth,inclination,radius).
    """
    x = cart_xyz[..., 0]
    y = cart_xyz[..., 1]
    z = cart_xyz[..., 2]

    hypot_xy = x.hypot(y)
    radius = hypot_xy.hypot(z)
    inclination = z.atan2(hypot_xy)
    azimuth = y.atan2(x)
    sph = torch.stack((azimuth, inclination, radius), dim=-1)
    return sph


@torch.jit.script
def sph_to_cart(sph_rad: Tensor) -> Tensor:
    """Convert spherical coordinates to Cartesian coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        sph_rad: Spherical coordinates (azimuth,inclination,radius).

    Returns:
        (N,3) Cartesian coordinates (x,y,z).
    """
    azimuth = sph_rad[..., 0]
    inclination = sph_rad[..., 1]
    radius = sph_rad[..., 2]

    rcos_inclination = radius * inclination.cos()
    x = rcos_inclination * azimuth.cos()
    y = rcos_inclination * azimuth.sin()
    z = radius * inclination.sin()
    cart_xyz = torch.stack((x, y, z), dim=-1)
    return cart_xyz


# @torch.jit.script
def convert_world_coordinates_to_grid(
    coordinates_m: Tensor,
    min_world_coordinates_m: Union[List[float], Tensor],
    delta_m_per_cell: Union[List[float], Tensor],
    grid_size: Union[List[int], Tensor],
    align_corners: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert world coordinates to grid coordinates.

    Args:
        coordinates_m: (N,D) Coordinates in meters.
        min_world_coordinates_m: (D,) Minimum coordinates in meters.
        delta_m_per_cell: (D,) Ratio of meters to cell in each dimension.
        grid_size: (D,) Size of the grid.
        align_corners: Boolean flag to align the world coordinates to the cell
            centers.

    Returns:
        The grid coordinates and the cropped points mask.
    """
    if isinstance(min_world_coordinates_m, List):
        min_world_coordinates_m = torch.as_tensor(
            min_world_coordinates_m, device=coordinates_m.device
        )
    if isinstance(delta_m_per_cell, List):
        delta_m_per_cell = torch.as_tensor(
            delta_m_per_cell, device=coordinates_m.device
        )
    if isinstance(grid_size, List):
        grid_size = torch.as_tensor(grid_size, device=coordinates_m.device)

    D = min(coordinates_m.shape[-1], len(min_world_coordinates_m))
    offset_m = torch.zeros_like(coordinates_m[0])
    offset_m[:D] = torch.as_tensor(min_world_coordinates_m[:D]).abs()

    indices = (coordinates_m[..., :D] + offset_m[:D]) / delta_m_per_cell[:D]
    if not align_corners:
        indices[..., :D] += 0.5
    indices = indices.long()

    upper = [float(x) for x in grid_size]
    _, mask = crop_coordinates(indices, [0.0, 0.0, 0.0], upper)
    return indices, mask


def voxelize(
    coordinates_m: Tensor,
    values: Tensor,
    min_world_coordinates_m: List[float],
    delta_m_per_cell: List[float],
    grid_size: List[int],
    align_corners: bool = False,
    max_num_values: int = 1,
    cluster_type: ClusterType = ClusterType.MEAN,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Voxelize a set of coordinates and values.

    Args:
        coordinates_m: (N,D) Coordinates in meters.
        values: (N,F) Values at each coordinate.
        min_world_coordinates_m: (D,) Minimum coordinates in meters.
        delta_m_per_cell: (D,) Ratio of meters to cell in each dimension.
        grid_size: (D,) Size of the grid.
        align_corners: Boolean flag to align the world coordinates to the cell
            centers.
        max_num_values: Max number of values per cell.
        cluster_type: Type of clustering to perform.

    Returns:
        The voxelized indices, values, and the counts per voxel.

    """
    indices, mask = convert_world_coordinates_to_grid(
        coordinates_m,
        min_world_coordinates_m,
        delta_m_per_cell,
        grid_size,
        align_corners,
    )

    indices, values, counts = cluster_grid(
        indices, values, grid_size, max_num_values, cluster_type
    )
    return indices, values, counts
