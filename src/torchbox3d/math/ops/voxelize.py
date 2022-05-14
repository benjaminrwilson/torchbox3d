"""Nearest neighbor methods."""

from enum import Enum, unique


@unique
class Reduction(str, Enum):
    """The type of reduction performed during voxelization."""

    CONCATENATE = "CONCATENATE"
    MEAN_POOL = "MEAN_POOL"


@unique
class VoxelizationPoolingType(str, Enum):
    """The pooling method used for 'pooling' voxelization."""

    MEAN = "MEAN"
