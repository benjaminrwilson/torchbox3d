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
from torchbox3d.structures.grid import RegularGrid
from torchbox3d.utils.io import write_img


@dataclass(unsafe_hash=True)
class PointPillars(LightningModule):
    """Implementation of Pointpillars."""

    dim_in: int
    delta_m_per_cell: Tuple[int, int, int]
    min_world_coordinates_m: Tuple[int, int]
    max_world_coordinates_m: Tuple[int, int]
    cluster_type: str
    debug: bool = True
    name: str = "point_pillars"

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.pointnet_layers = Sequential(
            Linear(self.dim_in, 64, bias=False),
            BatchNorm1d(64),  # type: ignore
            ReLU(inplace=True),
        )

    def pointnet(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply simple PointNet to the input.

        Args:
            x: (N,F) Tensor of point features.

        Returns:
            The PointNet encodings.
        """
        for layer in self.pointnet_layers:
            if isinstance(layer, BatchNorm1d):
                features = layer(features.transpose(1, 2)).transpose(1, 2)
            else:
                features = layer(features)
        features, indices = features.max(dim=1)
        return features, indices

    def forward(  # type: ignore[override]
        self, grid_data: RegularGridData
    ) -> Dict[str, Tensor]:
        """Compute PointPillars forward pass.

        Args:
            grid_data: Input data.

        Returns:
            A dictionary of layer names to outputs.
        """
        indices = grid_data.cells.indices.long()
        values = grid_data.cells.values
        counts = grid_data.cells.counts

        values, _ = self.pointnet(values)
        canvas = pillar_scatter(indices, values, counts, grid_data.grid)

        if self.debug:
            path = Path.home() / "code" / "bev.png"
            img = canvas.clone().detach().sum(dim=1, keepdim=True)[0]
            img /= img.max()
            write_img(img.mul(255.0).byte().cpu(), str(path))

        outputs = {"out": canvas}
        return outputs


def pillar_scatter(
    indices: Tensor, values: Tensor, points_per_cell: Tensor, grid: RegularGrid
) -> Tensor:
    """Scatter the pillars on the BEV canvas.

    Args:
        indices: Indices to emplace the input data.
        values: Input data.
        points_per_cell: Number of points per cell.
        grid: Voxel grid attributes.

    Returns:
        The BEV canvas with scatter pillar encodings.
    """
    length, width = grid.grid_size[:2]
    num_batches = int(indices[..., -1].max().item() + 1)
    num_features = int(values.shape[-1])

    canvas: Tensor = scatter_nd(
        indices,
        src=values,
        grid_shape=[width, length, num_batches, num_features],
        permutation=[2, 3, 0, 1],
    )
    canvas = canvas.reshape(num_batches, -1, length, width)
    return canvas
