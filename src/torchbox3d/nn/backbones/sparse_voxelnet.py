"""Sparse 3D backbone."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import ModuleDict, Sequential

from torchbox3d.nn.blocks.sparse import ConvolutionBlock, ResidualBlock
from torchbox3d.structures.data import RegularGridData
from torchbox3d.structures.sparse_tensor import SparseTensor


@dataclass(unsafe_hash=True)
class SparseVoxelNet(LightningModule):
    """Construct the backbone.

    Args:
        dim_in: Dimension of the input features.
        resolution_m_per_cell: (3,) Ratio of meters to cell in meters.
        min_range_m: (3,) Minimum range along the x,y,z axes in meters.
        max_range_m: (3,) Maximum range along the x,y,z axes in meters.
        voxelization_type: Voxelization type used in the transformation.
    """

    name: str
    dim_in: int
    resolution_m_per_cell: Tuple[int, int, int]
    min_range_m: Tuple[int, int, int]
    max_range_m: Tuple[int, int, int]
    voxelization_type: str
    layers: ModuleDict = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the network."""
        super().__init__()
        self.layers = ModuleDict(
            {
                "encoder": ConvolutionBlock(
                    in_channels=self.dim_in, out_channels=16, kernel_size=3
                ),
                "conv1": Sequential(
                    ResidualBlock(in_channels=16, out_channels=16),
                    ResidualBlock(in_channels=16, out_channels=16),
                ),
                "conv2": Sequential(
                    ConvolutionBlock(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                    ),
                    ResidualBlock(in_channels=32, out_channels=32),
                    ResidualBlock(in_channels=32, out_channels=32),
                ),
                "conv3": Sequential(
                    ConvolutionBlock(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                    ),
                    ResidualBlock(in_channels=64, out_channels=64),
                    ResidualBlock(in_channels=64, out_channels=64),
                ),
                "conv4": Sequential(
                    ConvolutionBlock(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                    ),
                    ResidualBlock(in_channels=128, out_channels=128),
                    ResidualBlock(in_channels=128, out_channels=128),
                ),
                "conv5": Sequential(
                    ConvolutionBlock(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=(1, 1, 3),
                        stride=(1, 1, 2),
                    ),
                ),
            }
        )

    def forward(  # type: ignore[override]
        self, x: RegularGridData
    ) -> Dict[str, Union[SparseTensor, Tensor]]:
        """Compute Sparse Voxelnet forward pass.

        Args:
            x: Input data.

        Returns:
            Dense representation constructed from the convolved,
                sparse outputs.
        """
        outputs = {"out": x.voxels}
        out: SparseTensor = outputs["out"]
        for layer_name, layer in self.layers.items():
            outputs["out"] = layer(outputs["out"])
            outputs[layer_name] = SparseTensor(
                out.F.detach().clone(),
                out.C.detach().clone(),
                out.s,
            )

        out = outputs["out"]
        out = SparseTensor(out.F, out.C, out.s)

        vgrid_shape = torch.as_tensor(x.grid.dims)
        stride = torch.as_tensor(out.s)
        dims: List[int] = (vgrid_shape / stride).int().tolist()

        # Width, length, height.
        W, L, H = dims[0], dims[1], dims[2]

        # Dimension size.
        D = out.F.shape[-1]

        # Batch size.
        B = int(out.C[:, -1].max().item() + 1)

        # Size.
        size = torch.Size([W, L, H, B, D])
        return outputs | {"out": out.to_dense(size=size)}
