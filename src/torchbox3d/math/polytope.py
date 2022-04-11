"""Geometric methods for polytopes.

Reference: https://en.wikipedia.org/wiki/Polytope
"""

import torch
from torch import Tensor


@torch.jit.script
def compute_interior_points_mask(
    points_xyz: Tensor, cuboid_vertices: Tensor
) -> Tensor:
    r"""Compute the interior points within a set of _axis-aligned_ cuboids.

    Reference:
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        points_xyz: (N,3) Points in Cartesian space.
        cuboid_vertices: (K,8,3) Vertices of the cuboids.

    Returns:
        (N,) A tensor of boolean flags indicating whether the points
            are interior to the cuboid.
    """
    vertices = cuboid_vertices[:, (6, 3, 1)]
    uvw = cuboid_vertices[:, 2:3] - vertices
    reference_vertex = cuboid_vertices[:, 2:3]

    dot_uvw_reference = uvw @ reference_vertex.transpose(1, 2)
    dot_uvw_vertices = torch.diagonal(uvw @ vertices.transpose(1, 2), 0, 2)[
        ..., None
    ]
    dot_uvw_points = uvw @ points_xyz.T

    constraint_a = torch.logical_and(
        dot_uvw_reference <= dot_uvw_points, dot_uvw_points <= dot_uvw_vertices
    )
    constraint_b = torch.logical_and(
        dot_uvw_reference >= dot_uvw_points, dot_uvw_points >= dot_uvw_vertices
    )
    is_interior: Tensor = torch.logical_or(constraint_a, constraint_b).all(
        dim=1
    )
    return is_interior


@torch.jit.script
def compute_polytope_interior(
    points_xyz: Tensor, polytope: Tensor, axis_aligned: bool = True
) -> Tensor:
    """Compute the interior points within a set of polytopes.

    Args:
        points_xyz: (N,3) Points in Cartesian space.
        polytope: (K,8,3) Vertices of the polytope.
        axis_aligned: Flag indicating whether polygon is axis-aligned.

    Returns:
        (K,) Booleans indicating whether the points are interior to the cuboid.

    Raises:
        NotImplementedError: Only axis-aligned polytopes are supported.
    """
    interior_points_mask: Tensor
    if axis_aligned:
        interior_points_mask = compute_interior_points_mask(
            polytope, points_xyz
        )
        return interior_points_mask
    raise NotImplementedError("Only axis-aligned polygons are supported!")
