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
