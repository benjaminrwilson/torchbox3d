"""Launch a training job."""

import logging
from pathlib import Path
from typing import Final, Optional, cast

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from torchbox3d.datasets.argoverse.av2 import ArgoverseDataModule

logging.getLogger("torch").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

HYDRA_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "conf"


@hydra.main(
    config_path=str(HYDRA_PATH),
    config_name="config",
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
        cfg.batch_size = 2
        cfg.trainer.max_epochs = 1000
        if cfg.trainer.devices == "auto":
            cfg.trainer.devices = 1

    datamodule = get_datamodule(cfg)
    trainer = get_trainer(cfg)
    model = get_model(cfg, datamodule, trainer.num_devices)
    trainer.fit(model, datamodule=datamodule)


def get_trainer(cfg: DictConfig) -> Trainer:
    """Get the trainer for training.

    Args:
        cfg: Trainer configuration.

    Returns:
        Trainer: The PyTorch Lightning trainer.
    """
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(
        ".",
        name="",
        version="",
        default_hp_metric=False,
    )

    strategy: Optional[DDPStrategy] = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    if not torch.cuda.is_available():
        cfg.trainer.devices = 1
        cfg.trainer.accelerator = "cpu"
        strategy = None

    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        strategy=strategy,
        callbacks=[lr_monitor],
    )
    return trainer


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
        import torch.multiprocessing as mp

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
