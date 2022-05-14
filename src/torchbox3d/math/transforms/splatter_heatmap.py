"""Encode and scatter objects as soft-targets (Gaussian) over an xy grid."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from torchbox3d.datasets.argoverse.constants import DATASET_TO_TAXONOMY
from torchbox3d.math.kernels import ogrid_sparse_gaussian
from torchbox3d.math.ops.coding import encode
from torchbox3d.math.ops.index import (
    ravel_multi_index,
    scatter_nd,
    unravel_index,
)
from torchbox3d.math.ops.pool import unique_indices
from torchbox3d.rendering.ops.shaders import clip_to_viewport
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.targets import GridTargets

logger = logging.getLogger(__file__)


@dataclass
class SplatterHeatmap:
    """Construct a splatter heatmap preprocessing object.

    Args:
        stride: Spatial downsampling factor.
        tasks_cfg: Metadata for each task head.
        dataset_name: Dataset name.
    """

    network_stride: int
    tasks_cfg: Dict[int, List[str]]
    dataset_name: str

    def __call__(self, x: RegularGridData) -> Data:
        """Encode and splatter the cuboids onto the BEV canvas.

        Args:
            x: Ground truth data.

        Returns:
            Ground truth data with additional attributes.
        """
        return self.splatter(x)

    def preprocess_targets(
        self, x: RegularGridData
    ) -> Tuple[Cuboids, Tensor, Tensor]:
        """Preprocess the targets based on detection config.

        Args:
            x: Data needed for forward/backward pass.

        Returns:
            The target cuboids which have been filtered to the requested
            detection categories and mapped to contiguous category
            ids.

        """
        LABEL_TO_INDEX = DATASET_TO_TAXONOMY[self.dataset_name]

        # Map class to unique detection id.
        labels = [label for t in self.tasks_cfg.values() for label in t]
        selected_indices = torch.as_tensor([LABEL_TO_INDEX[l] for l in labels])
        cuboids = x.cuboids

        # Sort cuboids based off of cuboid categories.
        inv = cuboids.categories.argsort()
        cuboids = cuboids[inv]

        # Compute mask for valid categories.
        selected_categories = cuboids.categories.eq(
            selected_indices[:, None]
        ).any(dim=0)

        cuboids = cuboids[selected_categories]
        noncontiguous_categories = cuboids.categories.clone()

        src = torch.arange(0, len(cuboids.categories))
        out: Tuple[Tensor, Tensor] = torch.unique(
            cuboids.categories, return_inverse=True
        )
        _, inv = out
        cuboids.categories = inv[src]

        # Compute the categories to contiguous set of ids.
        inv_map = {v: k for k, v in LABEL_TO_INDEX.items()}

        offset_map = {}
        task_map = {}
        for j, task in enumerate(self.tasks_cfg.values()):
            for i, label_class in enumerate(task):
                offset_map[label_class] = i
                task_map[label_class] = j

        offsets = torch.as_tensor(
            [
                offset_map[inv_map[int(cls_id.item())]]
                for cls_id in noncontiguous_categories
            ]
        )

        task_ids = torch.as_tensor(
            [
                task_map[inv_map[int(cls_id.item())]]
                for cls_id in noncontiguous_categories
            ]
        )
        return cuboids, offsets, task_ids

    def splatter(self, x: RegularGridData) -> Data:
        """Splatter the ground truth annotations onto the xy (BEV) canvas.

        Args:
            x: Ground truth data.

        Returns:
            Ground truth data with encoded attributes.
        """
        cuboids, offsets, task_ids = self.preprocess_targets(x)
        x.cuboids = cuboids

        targets = x.cuboids.params.clone()
        xy, mask = x.grid.transform_from(targets[..., :2])
        xy = xy[mask]
        targets = targets[mask]
        offsets = offsets[mask]
        task_ids = task_ids[mask]

        targets[..., :2] = xy
        targets[..., :2] /= self.network_stride

        x.cuboids = x.cuboids[mask]
        encoding = encode(targets)

        downsampled_grid = x.grid.downsample(self.network_stride)
        L, W = downsampled_grid[0], downsampled_grid[1]
        xy = targets[..., :2].int()
        lw = targets[..., 3:5]

        T = len(self.tasks_cfg)
        scores = torch.zeros((T, L, W))
        indices = torch.cat((task_ids[:, None], xy), dim=1)

        if len(xy) > 0:
            inv = unique_indices(indices)

            indices = indices[inv]
            offsets = offsets[inv]
            task_ids = task_ids[inv]
            encoding = encoding[inv]
            scores = scatter_gaussian_targets(
                task_ids=task_ids,
                xy=xy[inv],
                lw=lw[inv],
                scores=scores,
                shape=[L, W],
            )

        perm = [0, 3, 1, 2]
        offsets = scatter_nd(
            indices, src=offsets, shape=[T, L, W, 1], perm=perm
        )[None]

        mask = scatter_nd(
            indices,
            torch.ones_like(task_ids, dtype=torch.bool),
            shape=[T, L, W, 1],
            perm=perm,
        )[None]

        R = encoding.shape[1]
        encoding = scatter_nd(
            indices, encoding, shape=[T, L, W, R], perm=perm
        )[None]

        x.targets = GridTargets(
            scores=scores[None, :, None],
            encoding=encoding,
            offsets=offsets,
            mask=mask,
        )
        return x


@torch.jit.script
def scatter_gaussian_targets(
    task_ids: Tensor,
    xy: Tensor,
    lw: Tensor,
    scores: Tensor,
    shape: List[int],
) -> Tensor:
    """Scatter the Gaussian targets onto the BEV plane.

    Args:
        task_ids: (N,1) Tensor of task ids (integer).
        xy: (N,2) Tensor of the xy object centers.
        lw: (N,2) Tensor of length and width of the objects.
        scores: (N,1) Tensor of confidence scores.
        shape: (3,) Shape of the voxel grid.

    Returns:
        The BEV plane with Gaussian targets.
    """
    unique_task_ids: Tensor = torch.unique(task_ids)
    for _, task_id in enumerate(unique_task_ids):
        mask = task_ids == task_id
        task_xy = xy[mask]
        sigma = lw[mask] / 6

        sigma = torch.max(sigma, dim=-1, keepdim=True)[0]
        response, uv = ogrid_sparse_gaussian(task_xy, sigma, radius=3)
        uv, response, _ = clip_to_viewport(uv, response, shape[0], shape[1])
        raveled_indices = ravel_multi_index(uv, shape)
        out: Tuple[Tensor, Tensor] = torch.unique(
            raveled_indices, return_inverse=True
        )
        raveled_indices, inverse_indices = out
        index = inverse_indices[:, None]
        uv = unravel_index(raveled_indices, shape=shape)
        reduced_response = torch.scatter_reduce(
            response,
            dim=0,
            index=index,
            reduce="amax",
        ).mT
        u, v = uv[..., 0], uv[..., 1]
        scores[task_id : task_id + 1, u, v] = reduced_response
    return scores
