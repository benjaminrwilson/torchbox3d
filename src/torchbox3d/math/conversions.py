"""Geometric conversions."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


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


@torch.jit.script
def sweep_to_bev(points_xyz: Tensor, dims: Tuple[int, int, int]) -> Tensor:
    """Construct an image from a point cloud.

    Args:
        points_xyz: (N,3) Tensor of Cartesian points.
        dims: Voxel grid dimensions.

    Returns:
        (B,C,H,W) Bird's-eye view image.
    """
    indices = points_xyz.long()
    # Return an empty image if no indices are available after cropping.
    if len(indices) == 0:
        return torch.zeros(
            (1, 1) + dims[:2],
            device=points_xyz.device,
            dtype=points_xyz.dtype,
        )

    if indices.shape[-1] == 3:
        indices = F.pad(indices, [0, 1], "constant", 0.0)

    values = torch.ones_like(indices[..., 0], dtype=torch.float)

    sparse_dims: List[int] = list(dims)
    dense_dims = [int(indices[:, -1].max().item()) + 1]
    size = sparse_dims + dense_dims

    voxels: Tensor = torch.sparse_coo_tensor(
        indices=indices.T, values=values, size=size
    )
    voxels = torch.sparse.sum(voxels, dim=(2,))
    bev = voxels.to_dense()
    bev = bev.permute(2, 0, 1)[:, None]
    return bev
