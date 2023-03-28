"""Launch a training job."""

import logging
from pathlib import Path
from typing import Final, cast

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from torchbox3d.datasets.argoverse.av2 import ArgoverseDataModule

logging.getLogger("torch").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

HYDRA_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "conf"


@hydra.main(
    config_path=str(HYDRA_PATH),
    config_name="config",
    version_base=None,
)
def train(cfg: DictConfig) -> None:
    """Training entrypoint.

    Args:
        cfg: Training configuration.
    """
    rank_zero_info("Initializing training ...")

    # Set job options if debugging.
    if cfg.debug:
        logger.info("Using debug mode ...")

        cfg.num_workers = 0
        cfg.batch_size = 1
        cfg.trainer.max_epochs = 1000
        if cfg.trainer.devices == "auto":
            cfg.trainer.devices = 1

    datamodule = get_datamodule(cfg)
    trainer = instantiate(cfg.trainer)
    model = get_model(cfg, datamodule, 1)
    trainer.fit(model, datamodule=datamodule)


def get_model(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    num_devices: int,
) -> LightningModule:
    """Get the model to be trained.

    Args:
        cfg: Model configuration.
        datamodule: Associated datamodule for learning schedule configuration.
        num_devices: Number of devices used during training.

    Returns:
        The 3D detection model.arch_cfg.
    """
    datamodule.setup("fit")
    dataloader = cast(ArgoverseDataModule, datamodule.train_dataloader())
    model: LightningModule = instantiate(
        cfg.model.arch_cfg,
        steps_per_epoch=len(dataloader),
        dataset_name=datamodule.name,
        num_devices=num_devices,
        _recursive_=False,
    )
    return model


def get_datamodule(cfg: DictConfig) -> LightningDataModule:
    """Get the datamodule for training."""
    if cfg.num_workers == "auto":
        cfg.num_workers = mp.cpu_count()
        if torch.cuda.is_available():
            cfg.num_workers //= torch.cuda.device_count()

    datamodule: LightningDataModule = instantiate(
        cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        src_dir=cfg.src_dir,
        dst_dir=cfg.dst_dir,
        _convert_="all",
    )
    return datamodule


if __name__ == "__main__":
    train()
