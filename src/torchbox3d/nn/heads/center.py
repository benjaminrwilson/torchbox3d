"""CenterPointHead for keypoint classification and regression."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from torch import nn
from torch.functional import Tensor

from torchbox3d.nn.heads.deformable import DeformableDetectionHead
from torchbox3d.nn.losses.classification import FocalLoss
from torchbox3d.nn.losses.regression import RegressionLoss
from torchbox3d.structures.data import RegularGridData
from torchbox3d.structures.outputs import TaskOutputs
from torchbox3d.structures.targets import CenterPointLoss


@dataclass(unsafe_hash=True)
class CenterHead(LightningModule):
    """CenterHead class for keypoint classification and regression."""

    tasks_cfg: DictConfig
    weight: float
    in_channels: int
    task_in_channels: int
    common_heads: DictConfig

    tasks: nn.ModuleList = field(init=False)
    shared_conv: nn.Sequential = field(init=False)

    cls_loss: nn.Module = field(init=False)
    bbox_loss: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.task_in_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.task_in_channels),  # type: ignore
            nn.ReLU(inplace=True),
        )

        self.tasks = nn.ModuleList(
            [
                DeformableDetectionHead(
                    len(task),
                    deepcopy(self.common_heads),
                    self.task_in_channels,
                    bn=True,
                    final_kernel=3,
                )
                for task in self.tasks_cfg.values()
            ]
        )
        self.cls_loss = FocalLoss()
        self.bbox_loss = RegressionLoss()
        self.eps = 1e-4

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        y: RegularGridData,
    ) -> Tuple[List[TaskOutputs], CenterPointLoss]:
        """Network forward pass.

        Args:
            x: (B,C,H,W) Network outputs.
            y: Data passed to the network --- includes targets.

        Returns:
            Processed data.
        """
        outputs = self.shared_conv(x)
        data = [task(outputs) for task in self.tasks]
        return self.loss(data, y)

    def loss(
        self, task_outputs: List[TaskOutputs], grid_data: RegularGridData
    ) -> Tuple[List[TaskOutputs], CenterPointLoss]:
        """Compute the classification and regression losses.

        Args:
            task_outputs: Outputs from the network.
            grid_data: Target ground truth data.

        Returns:
            Dictionary of losses, outputs, and targets.
        """
        targets = grid_data.targets
        losses: List[CenterPointLoss] = []
        for task_idx, _ in enumerate(self.tasks):
            scores = targets.scores[:, task_idx]
            task_offsets = targets.offsets[:, task_idx].long()
            mask = targets.mask[:, task_idx]

            outputs_encoding = task_outputs[task_idx].regressands
            targets_encoding = targets.encoding[:, task_idx]

            heatmap = (
                task_outputs[task_idx]
                .logits.sigmoid_()
                .clamp(min=self.eps, max=1 - self.eps)
            )

            positive_loss, negative_loss = self.cls_loss(
                heatmap, scores, task_offsets, mask
            )

            coordinate_loss, dimension_loss, rotation_loss = self.bbox_loss(
                outputs_encoding, targets_encoding, mask
            ).split([3, 3, 2], dim=-1)

            loss = CenterPointLoss(
                positive_loss=positive_loss,
                negative_loss=negative_loss,
                coordinate_loss=coordinate_loss,
                dimension_loss=dimension_loss,
                rotation_loss=rotation_loss,
                regression_weight=self.weight,
            )
            losses.append(loss)
        return task_outputs, CenterPointLoss.stack(losses)
