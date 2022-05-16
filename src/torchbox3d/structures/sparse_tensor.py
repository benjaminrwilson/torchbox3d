"""A wrapper for the `SparseTensor` object in the `torchsparse` library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ItemsView, Tuple, Union

import torch
from torch import Size, Tensor, as_tensor

from torchbox3d.math.ops.index import scatter_nd


@dataclass
class SparseTensor:  # type: ignore
    """Wrapper around `torchsparse.SparseTensor`."""

    values: Tensor
    indices: Tensor
    counts: Tensor
    stride: Union[int, Tuple[int, ...]] = 1

    def to_dense(self, size: Size) -> Tensor:
        """Convert a `SparseTensor` into a dense output.

        Args:
            size: The size of the dense output tensor.

        Returns:
            (B,C*D,H,W) Dense tensor of spatial features.
        """
        length, width, height, num_batches, _ = size

        indices = self.indices
        indices[:, :3] = torch.div(
            indices[:, :3],
            as_tensor(self.stride, device=self.values.device),
            rounding_mode="trunc",
        )
        indices[:, 0] = indices[:, 0].clamp(0, width - 1)
        indices[:, 1] = indices[:, 1].clamp(0, length - 1)
        indices[:, 2] = indices[:, 2].clamp(0, height - 1)

        dense = torch.zeros(size)
        dense = scatter_nd(
            indices, self.F, list(size), [3, 4, 2, 0, 1]
        ).reshape(num_batches, -1, width, length)
        return dense

    def clone(self) -> SparseTensor:
        """Return a clone of the sparse tensor."""
        return SparseTensor(
            self.values.clone(), self.indices.clone(), self.stride
        )

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
