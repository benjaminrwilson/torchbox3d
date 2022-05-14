"""Geometric conversions."""


from typing import List, Tuple, Union

import torch
from torch import Tensor

from torchbox3d.math.crop import crop_coordinates
from torchbox3d.math.ops.cluster import (
    ClusterType,
    concatenate_cluster_grid,
    mean_cluster_grid,
)


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


# def sweep_to_bev(
#     points_m: Tensor,
#     grid: RegularGrid,
# ) -> Tensor:
#     """Construct an image from a point cloud.

#     Args:
#         points_m: (N,3) Tensor of Cartesian points.
#         dims: Voxel grid dimensions.

#     Returns:
#         (B,C,H,W) Bird's-eye view image.
#     """
#     cluster_type = ClusterType.MEAN
#     if points_m.ndim == 3:
#         cluster_type = ClusterType.CONCATENATE
#     breakpoint()
#     indices, _, _, _ = grid_cluster(
#         points_m, points_m, grid, reduction=cluster_type
#     )
#     # Return an empty image if no indices are available after cropping.
#     if len(indices) == 0:
#         dims = [1, 1] + list(grid.grid_size[:2])
#         return torch.zeros(
#             dims,
#             device=points_m.device,
#             dtype=points_m.dtype,
#         )

#     if indices.shape[-1] == 3:
#         indices = F.pad(indices, [0, 1], "constant", 0.0)

#     values = torch.ones_like(indices[..., 0], dtype=torch.float)
#     sparse_dims: List[int] = list(grid.grid_size)

#     dense_dims = []
#     if indices.shape[-1] > 2:
#         dense_dims = [int(indices[:, -1].max().item()) + 1]
#     size = sparse_dims + dense_dims

#     voxels: Tensor = torch.sparse_coo_tensor(
#         indices=indices.T, values=values, size=size
#     )
#     if len(sparse_dims) > 2:
#         voxels = torch.sparse.sum(voxels, dim=(-2,))
#     bev = voxels.to_dense()
#     if bev.ndim == 2:
#         bev = bev.unsqueeze(-1)
#     bev = bev.permute(2, 0, 1)[:, None]
#     return bev


# @torch.jit.script
def convert_world_coordinates_to_grid(
    coordinates_m: Tensor,
    min_world_coordinates_m: Union[List[float], Tensor],
    delta_m_per_cell: Union[List[float], Tensor],
    grid_size: Union[List[int], Tensor],
    align_corners: bool = False,
) -> Tuple[Tensor, Tensor]:
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

    indices = (coordinates_m[..., :D] + offset_m[:D]) / delta_m_per_cell
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
    indices, mask = convert_world_coordinates_to_grid(
        coordinates_m,
        min_world_coordinates_m,
        delta_m_per_cell,
        grid_size,
        align_corners,
    )

    if cluster_type.upper() == ClusterType.MEAN:
        indices, values, counts = mean_cluster_grid(indices, values, grid_size)
    elif cluster_type.upper() == ClusterType.CONCATENATE:
        indices, values, counts = concatenate_cluster_grid(
            indices, values, grid_size, max_num_values=max_num_values
        )
    else:
        raise NotImplementedError()
    return indices, values, counts
