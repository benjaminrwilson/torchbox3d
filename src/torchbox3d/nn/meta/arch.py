"""General object detection model."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningModule
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler  # type: ignore

from torchbox3d.structures.data import Data


@dataclass(unsafe_hash=True)
class Detector(LightningModule):
    """Construct a general detector object.

    Args:
        backbone_cfg: 3D processing block of the network.
        neck_cfg: Pseudo-3D processing block of the network.
        head_cfg: Classification / regression block of the network.
        tasks_cfg: Set of tasks describing sets of target objects.
        epochs: Number of epochs to train.
        lr: Learning rate of the experiment.
        batch_size: Size of the batches during training, validation, etc.
        devices:
        debug: Boolean flag to enable debugging.
    """

    backbone_cfg: DictConfig
    neck_cfg: DictConfig
    head_cfg: DictConfig
    tasks_cfg: DictConfig

    batch_size: int
    devices: Union[List[str], str]
    network_stride: int
    epochs: int
    lr: float
    debug: bool

    dataset_name: str = MISSING
    steps_per_epoch: int = MISSING

    train_transforms_cfg: Optional[Callable[[Data], Data]] = MISSING
    val_transforms_cfg: Optional[Callable[[Data], Data]] = MISSING
    test_transforms_cfg: Optional[Callable[[Data], Data]] = MISSING

    backbone: LightningModule = field(init=False)
    neck: LightningModule = field(init=False)
    head: LightningModule = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.backbone = instantiate(self.backbone_cfg)
        self.neck = instantiate(self.neck_cfg)
        self.head = instantiate(self.head_cfg)

    @property
    def max_lr(self) -> float:
        """Maximum learning rate during training."""
        return self.lr * self.batch_size

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizers and learning rate scheduler for training.

        Returns:
            The optimizers and learning rate scheduler.

        Raises:
            RuntimeError: If trainer is `None`.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer must not be `None`!")

        if torch.cuda.is_available():
            self.cuda()

        self.trainer.reset_train_dataloader()
        optimizer = AdamW(self.parameters(), lr=self.max_lr)
        lr_dict: Dict[str, Any] = {}
        if not self.debug:
            scheduler: _LRScheduler = OneCycleLR(
                optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
                max_lr=self.max_lr,
                div_factor=self.div_factor,
                pct_start=self.pct_start,
            )
            lr_dict = {
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "metrics/lr",
                }
            }
        return {"optimizer": optimizer} | lr_dict
