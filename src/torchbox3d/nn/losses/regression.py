"""Regression losses."""

from dataclasses import dataclass

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules.loss import L1Loss

from torchbox3d.math.ops.index import ravel_multi_index


@dataclass(unsafe_hash=True)
class RegressionLoss(LightningModule):
    """Class used for computing grid-based regression losses."""

    def __post_init__(self) -> None:
        """Initialize LightningModule."""
        super().__init__()

    def __init__(self) -> None:
        """Construct regression loss."""
        super().__init__()
        self.loss = L1Loss(reduction="none")
        self.eps = 1e-4

    def forward(  # type: ignore[override]
        self, src: Tensor, targets: Tensor, mask: Tensor
    ) -> Tensor:
        """Loss computation.

        Args:
            src: (B,C,H,W) Tensor of network regression outputs.
            targets: (B,C,H,W) Tensor of target regression parameters.
            mask: (B,1,H,W) Tensor of target centers (binary).

        Returns:
            (B,C,H,W) Regression losses.
        """
        # npos = mask.sum(dim=[1, 2, 3], keepdim=True)

        mask = mask.repeat_interleave(src.shape[1], dim=1)
        indices = ravel_multi_index(mask.nonzero(), shape=list(mask.shape))
        src = (
            src.flatten().gather(dim=-1, index=indices).view(-1, src.shape[1])
        )
        targets = (
            targets.flatten()
            .gather(dim=-1, index=indices)
            .view(-1, targets.shape[1])
        )
        loss = self.loss(src, targets) / (targets.shape[0] + self.eps)
        reduced_loss: Tensor = loss.sum(dim=0)
        return reduced_loss[None]
