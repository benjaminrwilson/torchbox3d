"""Units tests for the utils module of the argoverse subpackage."""

from pathlib import Path
from typing import Dict, Final

import pytest

from torchbox3d.datasets.argoverse.constants import AV2_ANNO_NAMES_TO_INDEX
from torchbox3d.datasets.argoverse.utils import read_sweep_data

TEST_DATA_DIR: Final[Path] = (
    Path(__file__).parent.parent.parent.resolve() / "test_data"
)


def test_aggregate_sweeps() -> None:
    """Unit test for aggregating lidar sweeps into a single reference frame."""


@pytest.mark.parametrize(
    "dataset_dir, split, log_id, sensor_name, timestamp_ns, category_to_index",
    [
        pytest.param(
            TEST_DATA_DIR / "logs",
            "val",
            "02a00399-3857-444e-8db3-a8f58489c394",
            "lidar",
            315966070559696000,
            AV2_ANNO_NAMES_TO_INDEX,
        )
    ],
)
def test_read_sweep_data(
    dataset_dir: Path,
    split: str,
    log_id: str,
    sensor_name: str,
    timestamp_ns: int,
    category_to_index: Dict[str, int],
) -> None:
    """Unit test for reading lidar data and annotations for a single sweep."""
    sweep_data = read_sweep_data(
        dataset_dir=dataset_dir,
        split=split,
        log_id=log_id,
        sensor_name=sensor_name,
        timestamp_ns=timestamp_ns,
        category_to_index=category_to_index,
    )
    assert sweep_data is not None
