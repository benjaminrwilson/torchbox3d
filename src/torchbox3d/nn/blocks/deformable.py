"""Deformable network blocks."""

from dataclasses import dataclass, field

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, nn
from torchvision.ops.deform_conv import DeformConv2d


@dataclass(unsafe_hash=True)
class DeformableBlock(LightningModule):
    """Construct Deformable Convolution head.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
        groups: Number of groups.
    """

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    groups: int = 4

    dc_offset: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        dc_offset_out_channels = 2 * self.groups * self.kernel_size**2
        self.dc_offset = nn.Conv2d(self.in_channels, dc_offset_out_channels, 1)

        padding = (self.kernel_size - 1) // 2
        self.dc = DeformConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=self.groups,
        )
        self.init_offset_()

    def init_offset_(self) -> None:
        """Initialize the deformable weights and biases."""
        torch.nn.init.zeros_(self.dc_offset.weight.data)  # type: ignore
        if self.dc_offset.bias is not None:
            torch.nn.init.zeros_(self.dc_offset.bias.data)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass.

        Args:
            x: (B,C,H,W) Input tensor.

        Returns:
            The convolved inputs.
        """
        offset = self.dc_offset(x)
        out: Tensor = self.dc(x, offset).relu_()
        return out
