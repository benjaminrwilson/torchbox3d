"""Unit tests for geometric conversions."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import pytest
import torch
from av2.utils.io import read_feather
from torch import Tensor

from torchbox3d.structures.grid import RegularGrid, VoxelGrid
from torchbox3d.utils.io import write_img

TEST_DATA_DIR: Final[Path] = (
    Path(__file__).parent.parent.resolve() / "test_data"
)


def test_cart_to_sph() -> None:
    """Unit test for converting Cartesian to spherical coordinates."""


def test_sph_to_cart() -> None:
    """Unit test for converting spherical to Cartesian coordinates."""


@pytest.mark.parametrize(
    "points_xyz, grid",
    [
        pytest.param(
            torch.as_tensor([[5, 5, 5], [10, 10, 10]]),
            VoxelGrid(
                min_world_coordinates_m=(-5.0, -5.0, -5.0),
                max_world_coordinates_m=(+5.0, +5.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
        ),
        pytest.param(
            torch.as_tensor(
                read_feather(
                    TEST_DATA_DIR
                    / "logs"
                    / "val"
                    / "02a00399-3857-444e-8db3-a8f58489c394"
                    / "sensors"
                    / "lidar"
                    / "315966070559696000.feather",
                    columns=("x", "y", "z"),
                ).to_numpy(),
                dtype=torch.float,
            ),
            RegularGrid(
                min_world_coordinates_m=(-100.0, -100.0, -5.0),
                max_world_coordinates_m=(+100.0, +100.0, +5.0),
                delta_m_per_cell=(+0.1, +0.1, +0.2),
            ),
        ),
    ],
)
def test_sweep_to_bev(points_xyz: Tensor, grid: RegularGrid) -> None:
    """Unit test for converting a Cartesian sweep to a BEV image."""
    bev = sweep_to_bev(points_xyz)
    bev = bev * 255.0

    with NamedTemporaryFile("w") as f:
        write_img(bev[0].byte(), f"{f.name}.png")
