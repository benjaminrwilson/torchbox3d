"""Convolutional heads."""
from typing import List

import torch
from omegaconf import DictConfig
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, nn
from torch.nn import Sequential


class ConvHead(LightningModule):
    """Convolution head class."""

    def __init__(
        self,
        heads: DictConfig,
        in_channels: int,
        out_channels: int = 64,
        final_kernel: int = 1,
        bn: bool = False,
    ) -> None:
        """Construct a convolutional head.

        Args:
            heads: Head configuration.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            final_kernel: Number of channels in the final kernel.
            bn: Flag for batch normalization.
        """
        super().__init__()

        padding = final_kernel // 2
        self.task_heads = nn.ModuleDict({})
        for head_name, (ncls, head_dims) in dict(heads).items():
            layers: List[nn.Module] = []

            n = head_dims - 1
            for _ in range(n):
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=padding,
                        bias=True,
                    )
                )
                if bn:
                    layers.append(nn.BatchNorm2d(out_channels))  # type: ignore
                layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Conv2d(
                    out_channels,
                    ncls,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=padding,
                    bias=True,
                )
            )
            self.task_heads[str(head_name)] = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Network forward pass.

        Args:
            x: (B,C,H,W) Input tensor.

        Returns:
            Convolutional head output.
        """
        return torch.cat(
            [task_head(x) for _, task_head in self.task_heads.items()],
            dim=1,
        )
