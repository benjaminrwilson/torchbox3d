"""Datamodule metaclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision.transforms import Compose


@dataclass
class DataModule(LightningDataModule):
    """Construct a general datamodule object.

    Args:
        tasks_cfg: List of tasks.
        num_workers: Number of dataloader workers.
        batch_size: Dataloader batch size.
        src_dir: The default data directory.
        dst_dir: The default data directory.
        train_transforms_cfg: Train transforms config.
        val_transforms_cfg: Val transforms config.
        test_transforms_cfg: Test transforms config.
    """

    train_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]
    val_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]
    test_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]

    tasks_cfg: DictConfig
    num_workers: int
    batch_size: int
    src_dir: Path
    dst_dir: Path
    name: str

    prepare_data_per_node: bool = False

    _train_transforms: Callable[..., Any] = field(init=False)
    _val_transforms: Callable[..., Any] = field(init=False)
    _test_transforms: Callable[..., Any] = field(init=False)

    def __post_init__(self) -> None:
        """Compose the data transforms (if the configurations exist)."""
        if self.train_transforms_cfg is not None:
            self._train_transforms = Compose(
                list(self.train_transforms_cfg.values())
            )
        if self.val_transforms_cfg is not None:
            self._val_transforms = Compose(
                list(self.val_transforms_cfg.values())
            )
        if self.test_transforms_cfg is not None:
            self._test_transforms = Compose(
                list(self.test_transforms_cfg.values())
            )

        self.save_hyperparameters(
            ignore=[
                "tasks_cfg",
                "train_transforms_cfg",
                "val_transforms_cfg",
                "test_transforms_cfg",
            ]
        )
