"""Network blocks for sparse convolution."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, Union

import torchsparse.nn as spnn
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import Sequential
from torchsparse.tensor import SparseTensor


@dataclass(unsafe_hash=True)
class ConvolutionBlock(LightningModule):
    """Construct a sparse-convolution block.

    Args:
        in_channels: Number of channels into the block.
        out_channels: Number of output channels from the block.
        kernel_size: Convolution kernel size.
        stride: Stride of the convolution kernel.
        dilation: Dilation of the convolution kernel.
        transposed: Flag for transposed sparse-convolution.
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, ...]] = 3
    stride: Union[int, Tuple[int, ...]] = 1
    dilation: int = 1
    transposed: bool = False

    net: Sequential = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.net = Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        spnn.Conv3d(
                            self.in_channels,
                            self.out_channels,
                            kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            stride=self.stride,
                            transposed=self.transposed,
                        ),
                    ),
                    ("bn", spnn.BatchNorm(self.out_channels)),
                    ("act", spnn.ReLU(True)),
                ]
            )
        )

    def forward(  # type: ignore[override]
        self, x: SparseTensor
    ) -> SparseTensor:
        """Forward pass for sparse convolution.

        Args:
            x: Sparse tensor network input.

        Returns:
            (N,F) 'Active sites' and associated features after convolution.
        """
        return self.net(x)


@dataclass(unsafe_hash=True)
class ResidualBlock(LightningModule):
    """Construct a residual, sparse-convolution block.

    Args:
        in_channels: The number of channels into the block.
        out_channels: The number of output channels from the block.
        kernel_size: Convolution kernel size.
        stride: The stride of the convolution kernel.
        dilation: The dialation of the convolution kernel.
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, ...]] = 3
    stride: Union[int, Tuple[int, ...]] = 1
    dilation: int = 1

    net: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.net = Sequential(
            spnn.Conv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=self.stride,
            ),
            spnn.BatchNorm(self.out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
            ),
            spnn.BatchNorm(self.out_channels),
        )

        self.downsample = (
            Sequential()
            if (self.in_channels == self.out_channels and self.stride == 1)
            else Sequential(
                spnn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    dilation=1,
                    stride=self.stride,
                ),
                spnn.BatchNorm(self.out_channels),
            )
        )

        self.relu = spnn.ReLU(inplace=True)

    def forward(  # type: ignore[override]
        self, x: SparseTensor
    ) -> SparseTensor:
        """Forward pass for sparse convolution with residual connection.

        Args:
            x: Network input sparse tensor.

        Returns:
            (N,F) 'Active sites' and associated features after convolution.
        """
        return self.relu(self.net(x) + self.downsample(x))
