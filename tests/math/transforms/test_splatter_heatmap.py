"""Unit tests for the splatter heatmap transform."""

from typing import Dict, List

import pytest
import torch

from torchbox3d.math.transforms.splatter_heatmap import SplatterHeatmap
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.ndgrid import VoxelGrid


@pytest.mark.parametrize(
    "voxel_grid," "network_stride," "tasks_cfg," "dataset_name," "cuboids,",
    [
        pytest.param(
            VoxelGrid(
                min_range_m=(-5.0, -5.0, -5.0),
                max_range_m=(+5.0, +5.0, +5.0),
                resolution_m_per_cell=(+0.1, +0.1, +0.2),
            ),
            1,
            {0: ["REGULAR_VEHICLE", "ANIMAL"]},
            "av2",
            Cuboids(
                params=torch.as_tensor(
                    [
                        [-3.5, -3.5, -3.5, 5, 5, 5, 1, 0, 0, 0],
                        [-1.25, -1.25, -1.25, 5, 5, 5, 1, 0, 0, 0],
                        [-1.25, -1.25, -1.25, 1, 1, 1, 1, 0, 0, 0],
                        [-1.25, 1.25, -1.25, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [1.25, 1.25, 1.25, 1, 1, 1, 1, 0, 0, 0],
                        [1.25, -1.25, -1.25, 1, 1, 1, 1, 0, 0, 0],
                        [3.5, 3.5, 3.5, 1, 1, 1, 1, 0, 0, 0],
                    ],
                    dtype=torch.float32,
                ),
                categories=torch.as_tensor([18, 18, 0, 18, 18, 18, 18, 18]),
                scores=torch.ones(8),
            ),
        )
    ],
    ids=["Test splattering Gaussian targets on a BEV plane."],
)
def test_splatter_heatmap(
    voxel_grid: VoxelGrid,
    network_stride: int,
    tasks_cfg: Dict[int, List[str]],
    dataset_name: str,
    cuboids: Cuboids,
) -> None:
    """Unit test for splattering Gaussian targets onto the BEV plane."""
    splatter_heatmap = SplatterHeatmap(
        network_stride=network_stride,
        tasks_cfg=tasks_cfg,
        dataset_name=dataset_name,
    )

    assert splatter_heatmap is not None
