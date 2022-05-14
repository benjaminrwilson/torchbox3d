"""Pointpillars implementation.

NOTE: This implementation is NOT complete and only serves a simplified
model for development currently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BatchNorm1d, ReLU, Sequential
from torch.nn.modules.linear import Linear

from torchbox3d.math.ops.index import scatter_nd
from torchbox3d.structures.data import RegularGridData
from torchbox3d.structures.regular_grid import RegularGrid
from torchbox3d.utils.io import write_img


@dataclass(unsafe_hash=True)
class PointPillars(LightningModule):
    """Implementation of Pointpillars."""

    dim_in: int
    delta_m_per_cell: Tuple[int, int, int]
    min_world_coordinates_m: Tuple[int, int]
    max_world_coordinates_m: Tuple[int, int]
    voxelization_type: str
    debug: bool = False
    name: str = "point_pillars"

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.pointnet_layers = Sequential(
            Linear(self.dim_in, 64, bias=False),
            BatchNorm1d(64),  # type: ignore
            ReLU(inplace=True),
        )

    def pointnet(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply simple PointNet to the input.

        Args:
            x: (N,F) Tensor of point features.

        Returns:
            The PointNet encodings.
        """
        for layer in self.pointnet_layers:
            if isinstance(layer, BatchNorm1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        x, indices = x.max(dim=1)
        return x, indices

    def forward(  # type: ignore[override]
        self, data: RegularGridData
    ) -> Dict[str, Tensor]:
        """Compute PointPillars forward pass.

        Args:
            data: Input data.

        Returns:
            A dictionary of layer names to outputs.
        """
        indices = data.voxels.C.long()
        x = data.voxels.F
        x, _ = self.pointnet(x)
        canvas = self.pillar_scatter(x, indices, data.grid)

        if self.debug:
            path = Path.home() / "code" / "bev.png"
            img = canvas.clone().detach().sum(dim=1, keepdim=True)[0]
            img /= img.max()
            write_img(img.mul(255.0).byte().cpu(), str(path))

        outputs = {"out": canvas}
        return outputs

    def pillar_scatter(
        self, x: Tensor, indices: Tensor, grid: RegularGrid
    ) -> Tensor:
        """Scatter the pillars on the BEV canvas.

        Args:
            x: Input data.
            indices: Indices to emplace the input data.
            grid: Voxel grid attributes.

        Returns:
            The BEV canvas with scatter pillar encodings.
        """
        L, W = grid.grid_size[:2]
        B = int(indices[..., -1].max().item() + 1)
        D = int(x.shape[-1])

        canvas: Tensor = scatter_nd(
            indices, src=x, shape=[W, L, B, D], perm=[2, 3, 0, 1]
        )
        canvas = canvas.reshape(B, -1, L, W)
        return canvas
