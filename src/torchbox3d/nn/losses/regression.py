"""Regression losses."""

from dataclasses import dataclass

from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules.loss import L1Loss


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
        npos = mask.sum(dim=[1, 2, 3], keepdim=True)
        loss = self.loss(src, targets) / (npos + self.eps)
        reduced_loss: Tensor = loss.sum(dim=[2, 3])
        return reduced_loss
