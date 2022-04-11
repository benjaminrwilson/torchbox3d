"""Unit tests for the av2 module."""

from pathlib import Path

from torchbox3d.datasets.argoverse.av2 import AV2


def test_av2() -> None:
    """Unit test for loading data in the Argoverse 2 dataloader."""
    rootdir = Path.home() / "data" / "datasets" / "av2" / "sensor"
    av2 = AV2(rootdir, name="av2", split="val")

    # for _ in av2:
    #     continue
