"""Implementation of CenterPoint."""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from av2.evaluation.detection.constants import CompetitionCategories
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import DetectionCfg
from av2.utils.io import read_feather
from omegaconf import MISSING
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor

from torchbox3d.datasets.argoverse.constants import (
    DATASET_TO_TAXONOMY,
    LABEL_ATTR,
)
from torchbox3d.math.ops.coding import decode
from torchbox3d.nn.meta.arch import Detector
from torchbox3d.rendering.tensorboard import to_tensorboard
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.outputs import NetworkOutputs, TaskOutputs
from torchbox3d.structures.regular_grid import VoxelGrid
from torchbox3d.structures.targets import CenterPointLoss

logger = logging.getLogger(__name__)


@dataclass(unsafe_hash=True)
class CenterPoint(Detector):
    """An implementation of 'Center-based 3D Object Detection and Tracking'.

    Url: https://arxiv.org/abs/2006.11275.
    """

    lr: float = MISSING
    epochs: int = MISSING
    div_factor: float = MISSING
    pct_start: float = MISSING
    max_k: int = MISSING
    train_log_freq: int = MISSING
    val_log_freq: int = MISSING
    debug: bool = MISSING

    src_dir: str = MISSING
    dst_dir: str = MISSING

    num_devices: int = MISSING

    def __post_init__(self) -> None:
        """Initialize network."""
        super().__post_init__()

        rank_zero_info("Initializing CenterPoint ...")
        LABEL_TO_INDEX = DATASET_TO_TAXONOMY[self.dataset_name]
        self.to_nonconsecutive = torch.as_tensor(
            [
                LABEL_TO_INDEX[label]
                for _, task in self.tasks_cfg.items()
                for label in task
            ]
        )
        self.to_label = {v: k for k, v in LABEL_TO_INDEX.items()}
        self.labels_cache: Optional[pd.DataFrame] = None
        if self.debug:
            self.train_log_freq = 1
            self.val_log_freq = 1

        self.save_hyperparameters(  # type: ignore
            ignore=["backbone", "neck", "head", "task"]
        )

    def forward(self, x: Data) -> CenterPointOutputs:  # type: ignore[override]
        """Compute CenterPoint forward pass."""
        backbone = self.backbone(x)
        neck = self.neck(backbone["out"])
        head, losses = self.head(neck, x)
        outputs = CenterPointOutputs(
            backbone=backbone, neck=neck, head=head, losses=losses
        )
        return outputs

    def on_train_start(self) -> None:
        """Initialize Tensorboard hyperparameters."""
        if self.logger is not None:
            self.logger.log_hyperparams(
                Namespace(kwargs=self.hparams),
                {
                    "hp/CDS": 0,
                    "hp/AP": 0,
                    "hp/ATE": 0,
                    "hp/ASE": 0,
                    "hp/AOE": 0,
                },
            )

    def training_step(  # type: ignore[override]
        self, data: RegularGridData, idx: int
    ) -> Dict[str, Any]:
        """Take one training step.

        Args:
            data: Input data.
            idx: Batch index.

        Returns:
            The reduced loss.
        """
        outputs = self.forward(data)
        losses: CenterPointLoss = outputs.losses
        self.log_dict(
            losses.as_dict(),
            prog_bar=True,
            batch_size=self.batch_size,
        )
        reduced_loss: Tensor = losses.loss.mean(dim=-1).sum()
        return {"loss": reduced_loss, "outputs": outputs}

    def on_train_batch_end(  # type: ignore[override]
        self,
        outputs: Dict[str, Any],
        batch: RegularGridData,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Log visualizations at the end of the training batch."""
        if batch_idx % self.train_log_freq == 0:
            network_outputs = outputs["outputs"]
            dts = self.predict_step(
                network_outputs.head, batch.grid, batch_idx
            )
            if self.trainer is not None:
                to_tensorboard(dts, batch, network_outputs, self.trainer)

    @torch.no_grad()  # type: ignore
    def validation_step(  # type: ignore[override]
        self, data: RegularGridData, idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Take a network validation step.

        Args:
            data: Input data.
            idx: Batch index.

        Returns:
            The validation outputs.
        """
        outputs = self.forward(data)
        dts = self.predict_step(outputs.head, data.grid, idx)
        if idx % self.val_log_freq == 0:
            if self.trainer is not None:
                to_tensorboard(dts, data, outputs, self.trainer)
        return {"uuids": data.uuids, "dts": dts.cpu()}

    @torch.no_grad()  # type: ignore
    def predict_step(  # type: ignore[override]
        self,
        outputs_list: List[TaskOutputs],
        voxel_grid: VoxelGrid,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Cuboids:
        """Compute the network predictions.

        Args:
            outputs_list: (T,) List of network task outputs.
            voxel_grid: Parameters of the voxel grid.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.

        Returns:
            The network predictions.
        """
        # Decode predictions.
        cuboids: Cuboids = decode(
            outputs_list,
            voxel_grid=voxel_grid,
            network_stride=self.network_stride,
            max_k=self.max_k,
            to_nonconsecutive=self.to_nonconsecutive,
        )
        return cuboids

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        """Run validation epoch end."""
        if self.trainer is None:
            return
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return

        dts = pd.concat(
            [
                _torchbox_to_av2(o["dts"], o["uuids"], self.to_label)
                for o in outputs
            ]
        )

        try:
            from torch.distributed import is_initialized

            if is_initialized():  # type: ignore
                # Gather detections from all gpus.
                gathered_outputs: List[Optional[pd.DataFrame]] = [
                    None
                ] * dist.get_world_size()  # type: ignore
                dist.all_gather_object(gathered_outputs, dts)  # type: ignore
                dts = pd.concat(gathered_outputs)
        except ImportError:
            pass

        # Only evaluate on one device.
        if self.trainer is not None and self.local_rank == 0:
            # TODO: Handle duplicates from uneven batches.
            dts = dts.sort_values("score", ascending=False).reset_index()
            log_ids = set(dts["log_id"].unique().tolist())

            dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
            split = "val"

            logger.info(f"Evaluating on the following splits: {split}.")
            logger.info("Loading validation data ...")

            dataset_dir = Path(self.src_dir) / split
            annotation_paths = sorted(
                dataset_dir.glob("*/annotations.feather")
            )

            annotation_paths = list(
                filter(lambda x: x.parent.stem in log_ids, annotation_paths)
            )

            data: List[pd.DataFrame] = []
            for p in annotation_paths:
                datum = read_feather(p)
                datum["log_id"] = p.parent.stem
                data.append(datum)
            gts = (
                pd.concat(data)
                .set_index(["log_id", "timestamp_ns"])
                .sort_values("category")
            )

            valid_uuids_gts: List[str] = gts.index.tolist()
            valid_uuids_dts: List[str] = dts.index.tolist()
            valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
            gts = gts.loc[list(valid_uuids)].sort_index()

            categories = set(x.value for x in CompetitionCategories)
            categories &= set(gts["category"].unique().tolist())

            cfg = DetectionCfg(
                dataset_dir=dataset_dir,
                categories=tuple(sorted(categories)),
                split=split,
                max_range_m=200.0,
                eval_only_roi_instances=True,
            )

            # Evaluate using Argoverse detection API.
            logger.info("Starting evaluation ...")
            eval_dts, eval_gts, metrics = evaluate(
                dts.reset_index(), gts.reset_index(), cfg
            )

            valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
            print(metrics.loc[valid_categories])
            for index, row in metrics.iterrows():
                for k, v in row.to_dict().items():
                    name = (
                        "hp/CDS"
                        if k == "CDS" and (index == "AVERAGE_METRICS")
                        else f"hp/{k}/{index}"
                    )
                    prog_bar = True if k == "CDS" else False
                    self.log(
                        name,
                        v,
                        rank_zero_only=True,
                        prog_bar=prog_bar,
                    )
            eval_dts.columns = [str(x) for x in eval_dts.columns]
            eval_gts.columns = [str(x) for x in eval_gts.columns]
            eval_dts.reset_index().to_feather("detections.feather")
            eval_gts.reset_index().to_feather("annotations.feather")


@dataclass
class CenterPointOutputs(NetworkOutputs):
    """Class which manipulates and tracks data in the `CenterPoint` method."""

    losses: CenterPointLoss


def _torchbox_to_av2(
    dts: Cuboids, uuids: Tuple[str, ...], idx_to_category: Dict[int, str]
) -> pd.DataFrame:
    cuboid_list = dts.cpu().cuboid_list()
    serialized_dts_list: List[pd.DataFrame] = []
    for uuid, cuboids in zip(uuids, cuboid_list):
        serialized_dts = pd.DataFrame(
            cuboids.params.numpy(), columns=list(LABEL_ATTR)
        )
        serialized_dts["score"] = cuboids.scores.numpy()
        serialized_dts["log_id"] = uuid[0]
        serialized_dts["timestamp_ns"] = int(uuid[1])
        serialized_dts["category"] = [
            idx_to_category[x] for x in cuboids.categories.numpy().flatten()
        ]
        serialized_dts_list.append(serialized_dts)
    detections = (
        pd.concat(serialized_dts_list)
        .set_index(["log_id", "timestamp_ns"])
        .sort_index()
    )
    return detections
