"""A wrapper for the `SparseTensor` object in the `torchsparse` library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ItemsView, Tuple, Union

import torch
import torchsparse
from torch import Size, Tensor, as_tensor, sparse_coo_tensor


@dataclass
class SparseTensor(torchsparse.SparseTensor):  # type: ignore
    """Wrapper around `torchsparse.SparseTensor`."""

    feats: Tensor
    coords: Tensor
    stride: Union[int, Tuple[int, ...]] = 1

    def __post_init__(self) -> None:
        """Initialize the parent class."""
        super().__init__(self.feats, self.coords, stride=self.stride)

    def to_dense(self, size: Size) -> Tensor:
        """Convert a `SparseTensor` into a dense output.

        Args:
            size: The size of the dense output tensor.

        Returns:
            (B,C*D,H,W) Dense tensor of spatial features.
        """
        L, W, H, B, D = size

        indices = self.C
        indices[:, :3] = torch.div(
            indices[:, :3],
            as_tensor(self.s, device=self.F.device),
            rounding_mode="trunc",
        )
        indices[:, 0] = indices[:, 0].clamp(0, W - 1)
        indices[:, 1] = indices[:, 1].clamp(0, L - 1)
        indices[:, 2] = indices[:, 2].clamp(0, H - 1)

        sparse = sparse_coo_tensor(indices=indices.T, values=self.F, size=size)
        dense = sparse.to_dense().permute(3, 4, 2, 0, 1)
        return dense.reshape(B, -1, W, L)

    def clone(self) -> SparseTensor:
        """Return a clone of the sparse tensor."""
        return SparseTensor(self.F.clone(), self.C.clone(), self.s)

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the attribute names and values."""
        return ItemsView({k: v for k, v in self.__dict__.items()})

    def cpu(self) -> SparseTensor:
        """Move all of the tensors to the cpu."""
        for k, v in self.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.cpu())
            elif k in set(["cmaps", "kmaps"]):
                cpu_tensors = {}
                for key, val in v.items():
                    if isinstance(val, list):
                        cpu_tensors[key] = [
                            x.cpu() if isinstance(x, Tensor) else x
                            for x in val
                        ]
                    else:
                        cpu_tensors[key] = val.cpu()
                setattr(self, k, cpu_tensors)
        return self
