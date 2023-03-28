"""Coding and decoding."""

from collections import defaultdict
from typing import DefaultDict, List, TypeVar

import torch
from kornia.geometry.subpix import nms2d
from torch import Tensor

from torchbox3d.math.linalg.lie.SO3 import quat_to_yaw, yaw_to_quat
from torchbox3d.math.ops.index import mgrid
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.grid import RegularGrid
from torchbox3d.structures.outputs import TaskOutputs

T = TypeVar("T")


@torch.jit.script
def _encode_lwh(cuboids: Tensor) -> Tensor:
    """Encode the cuboids in the `lwh` format.

    Args:
        cuboids: The ground truth annotations.

    Returns:
        The encoded ground truth annotations.
    """
    # Number of cuboids.
    num_cuboids = cuboids.shape[0]

    # Intialize box parameterization.
    encoding = torch.zeros((num_cuboids, 8))

    # Calculate grid offset.
    encoding[..., :2] = cuboids[..., :2] - cuboids[..., :2].int()
    encoding[..., 2] = cuboids[..., 2]

    # Dimensions under log for numerical stability.
    encoding[..., 3:6] = cuboids[..., 3:6].log()

    # Convert quaternions to yaw (rotation about the z-axis).
    yaws = quat_to_yaw(cuboids[..., 6:10])

    # Sin / Cosine embedding.
    encoding[..., 6] = torch.sin(yaws)
    encoding[..., 7] = torch.cos(yaws)

    # Return encoding.
    return encoding


@torch.jit.script
def encode(cuboids: Tensor) -> Tensor:
    """Encode a set of cuboids.

    Args:
        cuboids: The ground truth annotations.

    Returns:
        The encoded ground truth annotations.
    """
    encoding: Tensor = _encode_lwh(cuboids)
    return encoding


def decode(
    task_outputs_list: List[TaskOutputs],
    grid: RegularGrid,
    network_stride: int,
    max_k: int,
    to_nonconsecutive: Tensor,
) -> Cuboids:
    """Decode the output network output into cuboids.

    Args:
        task_outputs_list: Network output.
        grid: Voxel grid parameters.
        network_stride: Network stride.
        max_k: Max number of predictions to keep after decoding.
        to_nonconsecutive: (N,2) Tensor mapping consecutive class ids
            to original non-consecutive ids.

    Returns:
        The predicted cuboids.
    """
    return _decode_lwh(
        task_outputs_list, grid, network_stride, max_k, to_nonconsecutive
    )


def _decode_lwh(
    task_outputs_list: List[TaskOutputs],
    grid: RegularGrid,
    network_stride: int,
    max_k: int,
    to_nonconsecutive: Tensor,
) -> Cuboids:
    """Decode the length, width, height cuboid parameterization.

    Args:
        task_outputs_list: Network output.
        grid: Voxel grid parameters.
        network_stride: Network stride.
        max_k: Max number of predictions to keep after decoding.
        to_nonconsecutive: (N,2) Tensor mapping consecutive class ids
            to original non-consecutive ids.

    Returns:
        The decoded predictions.
    """
    task_outputs = task_outputs_list[0]
    device = task_outputs.regressands.device

    height = task_outputs.regressands.shape[-2]
    width = task_outputs.regressands.shape[-1]
    grid_idx = mgrid([[0, height], [0, width]])[None].float().to(device)

    delta_m_per_cell = torch.as_tensor(
        grid.delta_m_per_cell[:2],
        dtype=task_outputs.logits.dtype,
        device=task_outputs.logits.device,
    )[None, :, None, None]
    grid_offset = torch.as_tensor(
        grid.grid_offset_m[:2],
        dtype=task_outputs.logits.dtype,
        device=task_outputs.logits.device,
    )[None, :, None, None]

    task_offset = 0
    cuboid_list: DefaultDict[str, List[Tensor]] = defaultdict(list)
    for _, data in enumerate(task_outputs_list):
        # Get max scores and offsets within each task.
        scores, offsets = data.logits.max(dim=1, keepdim=True)
        offsets += task_offset

        # Split up the regressands.
        offset = data.regressands[:, :2]
        coordinates_z = data.regressands[:, 2:3]
        dim_log = data.regressands[:, 3:6]
        sin = data.regressands[:, 6:7]
        cos = data.regressands[:, 7:8]

        # Get yaw (rotation about the z-axis).
        yaw = torch.atan2(sin, cos)

        # (B,C,H,W) -> (B,W,H,C)
        yaw = yaw.transpose(1, 3)
        shape = yaw.shape[:-1] + (4,)

        # Convert yaw to quaternion representation.
        quat = yaw_to_quat(yaw.flatten(0, 2))
        quat = quat.reshape(shape).to(offset.device)
        quat = quat.transpose(1, 3)

        # Transform coordinates to original coordinates.
        ctrs = offset + grid_idx
        ctrs *= network_stride * delta_m_per_cell

        # Convert image coordinates to ego coordinates.
        ctrs -= grid_offset
        dims = torch.exp(dim_log)

        params = torch.cat((ctrs, coordinates_z, dims, quat), dim=1)
        task_offset += data.logits.shape[1]

        scores = nms2d(scores, (3, 3))
        scores, ranks = scores.permute(0, 3, 2, 1).flatten(1).topk(max_k)
        scores = scores.view(-1, 1)

        num_regressands = params.shape[1]
        index = ranks[..., None].repeat_interleave(num_regressands, dim=-1)

        params = (
            params.permute(0, 3, 2, 1).flatten(1, 2).gather(1, index)
        ).view(-1, num_regressands)

        offsets = (
            offsets.permute(0, 3, 2, 1).flatten(1).gather(1, ranks)
        ).view(-1, 1)

        num_batches = len(ranks)
        batch_index = torch.arange(0, num_batches).repeat_interleave(max_k)

        # Convert consecutive ids to original objects ids.
        categories = to_nonconsecutive[offsets]
        cuboid_list["params"].extend(params)
        cuboid_list["scores"].extend(scores)
        cuboid_list["categories"].extend(categories)
        cuboid_list["batch"].extend(batch_index)
    cuboids = Cuboids(**{k: torch.stack(v) for k, v in cuboid_list.items()})
    # TODO: Resolve NoneType.
    if cuboids.batch is not None:
        batch_index = cuboids.batch
        cuboids = cuboids[batch_index.argsort()]  # Make contiguous.
    return cuboids
