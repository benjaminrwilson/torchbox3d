"""Unit tests for SO(3) transformation utilities."""

import pytest
import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from torchbox3d.math.linalg.lie.SO3 import (
    quat_to_xyz,
    quat_to_yaw,
    xyz_to_quat,
    yaw_to_quat,
)


@pytest.mark.parametrize(
    "quats_wxyz",
    [
        pytest.param(
            torch.rand((10000, 4)),
        )
    ],
    ids=["Test converting random quaternions to yaw."],
)
def test_quat_to_yaw(quats_wxyz: Tensor) -> None:
    """Test converting a quaternion to yaw.

    Args:
        quats_wxyz: (N,4) Scalar first quaternions.

    NOTE: Yaw is rotation about the vertical axis in our
        coordinate system.
    """
    quats_wxyz /= torch.linalg.norm(quats_wxyz, dim=-1, keepdim=True)
    yaw = quat_to_yaw(quats_wxyz).rad2deg()
    yaw_ = torch.as_tensor(
        R.from_quat(quats_wxyz[..., [1, 2, 3, 0]].numpy()).as_euler(
            "xyz", degrees=True
        )
    )[..., -1]
    torch.testing.assert_allclose(yaw, yaw_)


@pytest.mark.parametrize(
    "quats_wxyz",
    [
        pytest.param(
            torch.rand((10000, 4)),
        )
    ],
    ids=["Test converting random quaternions to Tait-Bryan angles."],
)
def test_quat_to_xyz(quats_wxyz: Tensor) -> None:
    """Test converting a quaternion to Tait-Bryan angles (roll, pitch, yaw)."""
    quats_wxyz /= torch.linalg.norm(quats_wxyz, dim=-1, keepdim=True)
    xyz = quat_to_xyz(quats_wxyz).rad2deg()
    xyz_ = torch.as_tensor(
        R.from_quat(quats_wxyz[..., [1, 2, 3, 0]].numpy()).as_euler(
            "xyz", degrees=True
        )
    )
    torch.testing.assert_allclose(xyz, xyz_)


@pytest.mark.parametrize(
    "xyz_rad",
    [
        pytest.param(
            torch.rand((10000, 3)),
        )
    ],
    ids=["Test converting Tait-Bryan angles to quaternions."],
)
def test_xyz_to_quat(xyz_rad: Tensor) -> None:
    """Unit test converting Tait-Bryan angles to a scalar first quaternion.

    Args:
        xyz_rad: Roll, pitch, and yaw in radians.

    NOTE: The Tait-Bryan angles are equivalent to roll, pitch and yaw.
    """
    quats_wxyz = xyz_to_quat(xyz_rad)
    quats_wxyz_ = torch.as_tensor(
        R.from_euler("xyz", xyz_rad.numpy()).as_quat()
    )[..., [3, 0, 1, 2]]
    torch.testing.assert_allclose(quats_wxyz, quats_wxyz_)


@pytest.mark.parametrize(
    "yaw_rad",
    [
        pytest.param(
            torch.rand((10000)),
        )
    ],
    ids=["Test converting yaw to quaternions."],
)
def test_yaw_to_quat(yaw_rad: Tensor) -> None:
    """Unit test converting yaw to a scalar first quaternion."""
    quats_wxyz = yaw_to_quat(yaw_rad)
    quats_wxyz_ = torch.as_tensor(
        R.from_euler("z", yaw_rad.numpy()).as_quat()
    )[..., [3, 0, 1, 2]]
    torch.testing.assert_allclose(quats_wxyz, quats_wxyz_)
