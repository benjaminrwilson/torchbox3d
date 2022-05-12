"""Class for manipulation of 3D data and annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from torch import Tensor

from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.meta import TensorStruct
from torchbox3d.structures.ndgrid import NDGrid
from torchbox3d.structures.sparse_tensor import SparseTensor
from torchbox3d.structures.targets import GridTargets


@dataclass
class Data(TensorStruct):
    """General class for manipulating 3D data and associated annotations."""

    cuboids: Cuboids
    pos: Tensor
    x: Tensor
    uuids: Tuple[str, ...]


@dataclass
class RegularGridData(Data):
    """Data encoded on a regular grid.

    Args:
        grid: Grid object.
        voxels: Sparse tensor object.
        targets: Target encodings.
    """

    grid: NDGrid
    voxels: SparseTensor
    targets: GridTargets
