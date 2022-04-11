"""Classification losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor


@dataclass(unsafe_hash=True)
class FocalLoss(LightningModule):
    """Focal Loss class."""

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        y: Tensor,
        task_offsets: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for computing classification loss.

        Args:
            x: (B,C,H,W) Tensor of network outputs.
            y: (B,1,H,W) Tensor of target scores.
            task_offsets: (B,1,H,W) Tensor of task offsets.
            mask: (B,1,H,W) Tensor of target centers (binary mask).

        Returns:
            (B,1,H,W) Classification loss.
        """
        pos_loss: Tensor
        neg_loss: Tensor
        pos_loss, neg_loss = focal_loss(x, y, task_offsets, mask)
        return pos_loss, neg_loss


@torch.jit.script
def focal_loss(
    x: Tensor, y: Tensor, task_offsets: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute focal loss over the BEV grid.

    Args:
        x: (B,C,H,W) Tensor of network outputs.
        y: (B,1,H,W) Tensor of target scores.
        task_offsets: (B,1,H,W) Tensor of task offsets.
        mask: (B,1,H,W) Tensor of target centers (binary mask).

    Returns:
        (B,) Positive loss and (B,) negative loss.
    """
    neg_loss = (1 - x).log_() * (x**2) * (1 - y) ** 4

    x = torch.gather(x, dim=1, index=task_offsets)
    pos_loss = x.log_() * (1 - x) ** 2 * mask

    dim = [1, 2, 3]
    npos = torch.clamp(mask.sum(dim=dim), min=1)
    neg_loss = -torch.sum(neg_loss, dim=dim) / npos
    pos_loss = -torch.sum(pos_loss, dim=dim) / npos
    return pos_loss, neg_loss
