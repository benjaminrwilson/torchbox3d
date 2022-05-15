"""Methods to help visualize data during training."""

from typing import Optional, Tuple

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torchvision.utils import make_grid

from torchbox3d.math.conversions import denormalize_pixel_intensities
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import RegularGridData
from torchbox3d.structures.grid import RegularGrid
from torchbox3d.structures.outputs import NetworkOutputs


@rank_zero_only
def to_tensorboard(
    dts: Cuboids,
    gts: RegularGridData,
    network_outputs: NetworkOutputs,
    trainer: Trainer,
) -> None:
    """Log training, validation, etc. information to TensorBoard.

    Args:
        dts: Detections.
        gts: Ground truth targets.
        network_outputs: Encoded outputs from the network.
        trainer: PytorchLightning Trainer class.
    """
    if trainer.state.stage == RunningStage.SANITY_CHECKING:
        return

    B = int(gts.voxels.C[..., -1].max().item()) + 1
    size = gts.grid.grid_size + (B, 3)
    grid = torch.sparse_coo_tensor(
        indices=gts.voxels.C.mT, values=gts.voxels.F[..., :3], size=size
    )
    grid = torch.sparse.sum(grid, dim=2)
    bev = (
        grid.to_dense()
        .permute(2, 3, 0, 1)[0][2:3]
        .abs()
        .repeat_interleave(3, dim=0)
    )

    bev = denormalize_pixel_intensities(bev)
    _draw_cuboids(gts.cuboids, bev, gts.grid, (0, 0, 255))

    dts_list = dts.cuboid_list()
    selected_predictions = dts_list[0]

    # Only show 100 predictions for speed.
    _, indices = selected_predictions.scores.topk(k=100, dim=0)
    selected_predictions = selected_predictions[indices.flatten()]
    if len(selected_predictions) > 0:
        bev = selected_predictions.draw_on_bev(gts.grid, bev)
    tensorboard_log_img("bev", bev, trainer)

    targets = gts.targets
    heatmaps_dts = torch.stack(
        [e.logits.max(dim=1, keepdim=True)[0] for e in network_outputs.head]
    ).transpose(0, 1)

    heatmaps = torch.cat((heatmaps_dts, targets.scores, targets.mask), dim=1)
    heatmaps = denormalize_pixel_intensities(heatmaps)
    heatmaps = heatmaps[0]
    grid = make_grid(
        heatmaps,
        nrow=len(heatmaps) // 3,
        padding=4,
        pad_value=128,
    )
    tensorboard_log_img("heatmap", grid, trainer)


def _draw_cuboids(
    cuboids: Cuboids,
    bev: Tensor,
    grid: RegularGrid,
    color: Tuple[int, int, int],
    k: Optional[int] = None,
) -> Tensor:
    """Draw cuboids on a bird's-eye view image."""
    if len(cuboids) == 0:
        return bev
    cuboids_list = cuboids.cuboid_list()
    cuboids = cuboids_list[0]
    bev = cuboids.draw_on_bev(grid, bev, color=color)
    return bev


def tensorboard_log_img(name: str, img: Tensor, trainer: Trainer) -> None:
    """Log an image to TensorBoard.

    Args:
        name: Displayed image name.
        img: (C,H,W) Image tensor.
        trainer: Pytorch-lightning trainer class.
    """
    if trainer.logger is None or not isinstance(
        trainer.logger, TensorBoardLogger
    ):
        return
    stage = trainer.state.stage
    trainer.logger.experiment.add_image(
        f"{stage}/{name}",
        img,
        global_step=trainer.global_step,
    )
