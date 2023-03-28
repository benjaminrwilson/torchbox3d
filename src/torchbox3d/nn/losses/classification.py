"""Classification losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from pytorch_lightning.core import LightningModule
from torch import Tensor

from torchbox3d.math.ops.index import ravel_multi_index


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
    src: Tensor, targets: Tensor, task_offsets: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute focal loss over the BEV grid.

    Args:
        src: (B,C,H,W) Tensor of network outputs.
        targets: (B,1,H,W) Tensor of target scores.
        task_offsets: (B,1,H,W) Tensor of task offsets.
        mask: (B,1,H,W) Tensor of target centers (binary mask).

    Returns:
        (B,) Positive loss and (B,) negative loss.
    """
    index = ravel_multi_index(mask.nonzero(), shape=list(mask.shape))

    negative_loss = (1 - src).log_() * (src**2) * (1 - targets) ** 4
    src = src.gather(dim=1, index=task_offsets)
    src = src.flatten().gather(dim=-1, index=index)

    npos = mask.flatten(1, -1).sum(dim=-1)
    src = src.log_() * (1 - src) ** 2

    positive_loss = torch.zeros(
        (len(npos), int(npos.max())), device=src.device
    )
    batch_lengths: List[int] = npos.tolist()

    batch_losses = torch.split(src, batch_lengths)  # type: ignore
    for i, loss in enumerate(batch_losses):
        positive_loss[i, : len(loss)] = loss

    dim = [1, 2, 3]
    npos = torch.clamp(mask.sum(dim=dim), min=1)
    negative_loss = -torch.sum(negative_loss, dim=dim) / npos
    positive_loss = -torch.sum(positive_loss, dim=-1) / npos
    return positive_loss, negative_loss
