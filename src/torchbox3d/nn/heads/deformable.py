"""Deformable detection heads."""

from dataclasses import dataclass, field

from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, ReLU, Sequential

from torchbox3d.nn.blocks.deformable import DeformableBlock
from torchbox3d.nn.heads.conv import ConvHead
from torchbox3d.structures.outputs import TaskOutputs


@dataclass(unsafe_hash=True)
class DeformableDetectionHead(LightningModule):
    """Construct deformable convolution head.

    Args:
        num_cls: Number of detection classes.
        heads: Head configuration.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        final_kernel: Number of channels in the final kernel.
        bn: Flag to enable batch normalization.
        kernel_size: Kernel size.
        groups: Number of groups.
        padding: Padding size.
        stride: Network stride.
        bias: Flag to use bias.
    """

    num_cls: int
    heads: DictConfig
    in_channels: int
    out_channels: int = 64
    final_kernel: int = 1
    bn: bool = False
    kernel_size: int = 3
    groups: int = 4
    padding: int = 1
    stride: int = 1
    bias: bool = True

    classification_head: Sequential = field(init=False)
    regression_head: Sequential = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.classification_head = Sequential(
            DeformableBlock(
                self.in_channels,
                self.in_channels,
                kernel_size=self.kernel_size,
                groups=self.groups,
            ),
            Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            ),
            BatchNorm2d(self.out_channels),  # type: ignore
            ReLU(inplace=True),
            Conv2d(
                self.out_channels,
                self.num_cls,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
        )

        self.regression_head = Sequential(
            DeformableBlock(
                self.in_channels,
                self.in_channels,
                kernel_size=self.kernel_size,
                groups=self.groups,
            ),
            ConvHead(
                self.heads,
                self.in_channels,
                out_channels=self.out_channels,
                bn=self.bn,
                final_kernel=self.final_kernel,
            ),
        )

    def forward(self, x: Tensor) -> TaskOutputs:  # type: ignore[override]
        """Network forward pass.

        Args:
            x: (B,C,H,W) Tensor of network inputs.

        Returns:
            Classification and regression heatmaps.
        """
        logits = self.classification_head(x)
        regressands = self.regression_head(x)
        return TaskOutputs(logits=logits, regressands=regressands)
