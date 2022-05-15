"""Launch a training job."""

import logging
from pathlib import Path
from typing import Final

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from torchbox3d.nn.arch.centerpoint import CenterPoint

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
    rank_zero_info("Initializing validation ...")

    datamodule = get_datamodule(cfg)
    trainer = get_trainer(cfg)
    model: LightningModule = CenterPoint.load_from_checkpoint(
        "/home/ubuntu/code/torchbox-3d/scripts/experiments/centerpoint/"
        "2022-04-07-03-40-52/checkpoints/test.ckpt"
    )
    trainer.validate(model, datamodule=datamodule)


def get_trainer(cfg: DictConfig) -> Trainer:
    """Get the trainer for training.

    Args:
        cfg: Trainer configuration.

    Returns:
        Trainer: The PyTorch Lightning trainer.
    """

    trainer = Trainer(
        **cfg.trainer,
    )
    return trainer


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
