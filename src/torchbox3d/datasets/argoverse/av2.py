"""PyTorch implementation of an Argoverse 2 (AV2), 3D detection dataloader."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch.utils.data as data_utils
from av2.datasets.sensor.sensor_dataloader import LIDAR_PATTERN
from av2.datasets.sensor.utils import convert_path_to_named_record
from torch.utils.data.dataloader import DataLoader

from torchbox3d.datasets.argoverse.constants import DATASET_TO_TAXONOMY
from torchbox3d.datasets.argoverse.utils import read_sweep_data
from torchbox3d.datasets.datamodule import DataModule
from torchbox3d.datasets.dataset import Dataset
from torchbox3d.structures.data import Data
from torchbox3d.utils.collater import collate
from torchbox3d.utils.collections import flatten

logger = logging.getLogger(__name__)


@dataclass
class AV2(Dataset, data_utils.Dataset[Data]):
    """Argoverse 2 sensor dataset.

    Args:
        transform: Callable transformation applied to each datum.
        sensor_cache:
        label_to_idx:
    """

    transform: Optional[Callable[[Data], Data]] = None
    sensor_cache: pd.DataFrame = field(init=False)
    label_to_idx: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize instance variables."""
        super().__post_init__()
        self.label_to_idx = DATASET_TO_TAXONOMY[self.name]

        src_dir = Path(self.dataset_dir) / self.split
        lidar_paths = sorted(
            src_dir.glob(LIDAR_PATTERN), key=lambda x: int(x.stem)
        )
        records = [convert_path_to_named_record(p) for p in lidar_paths]

        self.sensor_cache = pd.DataFrame(records)
        self.sensor_cache.set_index(
            ["log_id", "sensor_name", "timestamp_ns"], inplace=True
        )
        self.sensor_cache.sort_index(inplace=True)

    def __len__(self) -> int:
        """Return the length of the dataset records."""
        return len(self.sensor_cache)

    def __getitem__(self, index: int) -> Data:
        """Load an item of the dataset and return it.

        Args:
            index: The dataset item index.

        Returns:
            An item of the dataset.
        """
        record: Tuple[str, str, int] = self.sensor_cache.iloc[index].name
        log_id, sensor_name, timestamp_ns = record
        datum = read_sweep_data(
            self.dataset_dir,
            self.split,
            log_id,
            sensor_name,
            timestamp_ns,
            self.label_to_idx,
        )
        if self.transform:
            datum = self.transform(datum)
        return datum


@dataclass
class ArgoverseDataModule(DataModule):
    """Construct an Argoverse datamodule.

    Args:
        num_workers: The number of `pytorch` dataloader workers.
        batch_size: The batch size for the `pytorch` dataloader.
        src_dir: Source data directory.
        dst_dir: Destination data directory.
        name: Name of the dataset.
    """

    def __post_init__(self) -> None:
        """Initialize the meta-datamodule."""
        return super().__post_init__()

    def setup(self, stage: Optional[str] = None) -> None:
        """Dataset setup for requested splits.

        Args:
            stage: The name representing the type of workflow (e.g., fit).
        """
        classes: Tuple[str, ...] = tuple(
            flatten(list(self.tasks_cfg.values()))
        )
        self.train_dataset = AV2(
            dataset_dir=self.src_dir,
            name=self.name,
            split="train",
            classes=classes,
            transform=self._train_transforms,
        )

        self.val_dataset = AV2(
            dataset_dir=self.src_dir,
            name=self.name,
            split="val",
            classes=classes,
            transform=self._val_transforms,
        )

    def train_dataloader(self) -> DataLoader[Data]:
        """Return the _train_ dataloader.

        Returns:
            The PyTorch _train_ dataloader.
        """
        dataloader: DataLoader[Data] = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            shuffle=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def val_dataloader(self) -> DataLoader[Data]:
        """Return the _validation_ dataloader.

        Returns:
            The PyTorch _validation_ dataloader.
        """
        dataloader: DataLoader[Data] = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader[Data]:
        """Return the _predict_ dataloader.

        Returns:
            The PyTorch _predict_ dataloader.
        """
        dataloader: DataLoader[Data] = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader
