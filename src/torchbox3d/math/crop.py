"""Spatial cropping methods."""

from typing import List, Tuple

import torch
from torch import Tensor


@torch.jit.script
def crop_points(
    points: Tensor,
    lower_bound_inclusive: List[float],
    upper_bound_exclusive: List[float],
) -> Tuple[Tensor, Tensor]:
    """Crop an N-dimensional tensor.

    Args:
        points: (N,D) Tensor of points.
        lower_bound_inclusive: (D,) Minimum coordinate thresholds for cropping.
        upper_bound_exclusive: (D,) Maximum coordinate thresholds for cropping.

    Returns:
        (N,) Valid point mask.
    """
    lower = torch.as_tensor(lower_bound_inclusive, device=points.device)
    upper = torch.as_tensor(upper_bound_exclusive, device=points.device)
    mask = torch.logical_and(points >= lower, points < upper).all(dim=1)
    return points[mask], mask
