"""Structure which models a set of cuboids.

Reference: https://en.wikipedia.org/wiki/Cuboid
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, ItemsView, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torchbox3d.math.linalg.lie.SO3 import quat_to_mat
from torchbox3d.rendering.ops.shaders import polygon
from torchbox3d.structures.meta import TensorStruct
from torchbox3d.structures.ndgrid import VoxelGrid


@dataclass
class Cuboids(TensorStruct):
    """Models a set of cuboids.

    Args:
        params: (N,K) Tensor of cuboid parameters.
        categories: (N,) Tensor of cuboid categories (integer).
        scores: (N,) Tensor of cuboid confidence scores.
        batch: (N,) Tensor of batch indices.
    """

    params: Tensor
    categories: Tensor
    scores: Tensor
    batch: Optional[Tensor] = None

    def __len__(self) -> int:
        """Return the number of cuboids."""
        return len(self.params)

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the attribute names and values."""
        return ItemsView({k: v for k, v in self.__dict__.items()})

    def __getitem__(self, *index: Union[int, slice, Tensor]) -> Cuboids:
        """Get the items at the indices provided."""
        output: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                output[k] = v[index]
        return Cuboids(**output)

    @property
    def xyz_m(self) -> Tensor:
        """Return the cuboids centers (x,y,z) in meters."""
        return self.params[:, :3]

    @property
    def dims_lwh_m(self) -> Tensor:
        """Return the cuboids length, width, and height in meters."""
        return self.params[:, 3:6]

    @property
    def quat_wxyz(self) -> Tensor:
        """(N,4) Return the scalar first quaternion coefficients."""
        return self.params[:, 6:10]

    @property
    def mat(self) -> Tensor:
        """Return a rotation matrix representing the object's pose."""
        mat: Tensor = quat_to_mat(self.params[..., -4:])
        return mat

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the cuboids shape.."""
        shape: Tuple[int, ...] = tuple(int(x) for x in self.params.shape)
        return shape

    def cuboid_list(self) -> List[Cuboids]:
        """Get the list representation of the cuboids.

        Returns:
            The list of cuboids --- each element corresponds to the cuboids
                for a particular batch.

        Raises:
            ValueError: If the cuboids batch size does not exist.
        """
        if self.batch is None:
            raise ValueError("Batch size _must_ exist!")
        outputs: Tuple[Tensor, Tensor] = torch.unique_consecutive(
            self.batch, return_counts=True
        )

        _, counts = outputs
        count_list: List[int] = counts.tolist()

        start = 0
        cuboid_list: List[Cuboids] = []
        for count in count_list:
            batch_cuboids = self[start : start + count]
            cuboid_list.append(batch_cuboids)
            start += count
        return cuboid_list

    @cached_property
    def vertices_m(self) -> Tensor:
        r"""Return the cuboid vertices in the destination reference frame.

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

        Returns:
            (8,3) array of cuboid vertices.
        """
        unit_vertices_obj_xyz_m: Tensor = torch.as_tensor(
            [
                [+1, +1, +1],  # 0
                [+1, -1, +1],  # 1
                [+1, -1, -1],  # 2
                [+1, +1, -1],  # 3
                [-1, +1, +1],  # 4
                [-1, -1, +1],  # 5
                [-1, -1, -1],  # 6
                [-1, +1, -1],  # 7
            ],
            device=self.dims_lwh_m.device,
            dtype=self.dims_lwh_m.dtype,
        )
        dims_lwh_m = self.dims_lwh_m

        # Transform unit polygons.
        vertices_ego: Tensor = (
            dims_lwh_m[:, None] / 2.0
        ) * unit_vertices_obj_xyz_m[None]
        R = quat_to_mat(self.quat_wxyz)
        t = self.xyz_m
        vertices_ego = vertices_ego @ R.mT + t[:, None]
        return vertices_ego

    def draw_on_bev(
        self,
        voxel_grid: VoxelGrid,
        bev: Tensor,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> Tensor:
        """Draw a set of bounding boxes on a BEV image.

        Args:
            voxel_grid: Object describing voxel grid characteristics.
            bev: (3,H,W) Bird's-eye view image.
            color: 3-channel color (RGB or BGR).

        Returns:
            (3,H,W) Image with boxes drawn.
        """
        colors = torch.as_tensor(
            [
                color,
                color,
                color,
                color,
            ],
            device=bev.device,
            dtype=bev.dtype,
        )
        edge_indices = torch.as_tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], device=bev.device
        )
        vertices_uv_list = self.vertices_m[:, [2, 3, 7, 6], :2]
        for i, vertices_uv in enumerate(vertices_uv_list):
            (
                vertices_uv,
                mask,
            ) = voxel_grid.transform_to_grid_coordinates(vertices_uv)

            vertices_uv = vertices_uv.float()
            colors_uint8 = (colors * self.scores[i]).round().byte()
            polygon(
                vertices_uv,
                edge_indices[mask],
                colors=colors_uint8,
                img=bev,
                width_px=1,
            )
        return bev
