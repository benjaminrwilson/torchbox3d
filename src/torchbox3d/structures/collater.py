"""Collation utilities for dataloaders."""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torchsparse.utils.collate import sparse_collate

from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.sparse_tensor import SparseTensor
from torchbox3d.structures.targets import GridTargets
from torchbox3d.structuresgrid import RegularGrid


def collate(data_list: Sequence[Data]) -> Data:
    """Collate (merge) a sequence of items.

    Args:
        data_list: Sequence of data to be collated.

    Returns:
        The collated data.

    Raises:
        TypeError: If the data type is not supported for collation.
    """
    collated_data: DefaultDict[str, List[Any]] = defaultdict(list)
    for data in data_list:
        for attr_name, attr in data.items():
            collated_data[attr_name].append(attr)

    output: Dict[str, Any] = {}
    for attr_name, attr in collated_data.items():
        elem = attr[0]

        # Pad with batch index.
        if attr_name in set(["pos", "values"]):
            output[attr_name] = torch.cat(
                [
                    F.pad(elem, [0, 1], "constant", i)
                    for i, elem in enumerate(attr)
                ]
            )
        elif isinstance(elem, Cuboids):
            for i, elem in enumerate(attr):
                elem.batch = torch.full_like(elem.params[:, 0], i)
            output[attr_name] = Cuboids.cat(attr)
        elif isinstance(elem, SparseTensor):
            sparse_tensor = sparse_collate(attr)
            output[attr_name] = SparseTensor(
                values=sparse_tensor.F, indices=sparse_tensor.C
            )
        elif isinstance(elem, RegularGrid):
            output[attr_name] = attr[0]
        elif attr_name in set(["encoding", "mask", "coo"]):
            output[attr_name] = torch.cat(attr)
        elif attr_name == "uuids":
            output[attr_name] = attr
        elif isinstance(elem, GridTargets):
            output[attr_name] = GridTargets.cat(attr)
        else:
            breakpoint()
            raise TypeError(f"Invalid type: {type(elem)}")
    data = RegularGridData(**output)
    return data
