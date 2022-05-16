"""Unit tests for geometric conversions."""

from pathlib import Path
from typing import Final

TEST_DATA_DIR: Final[Path] = (
    Path(__file__).parent.parent.resolve() / "test_data"
)


def test_cart_to_sph() -> None:
    """Unit test for converting Cartesian to spherical coordinates."""


def test_sph_to_cart() -> None:
    """Unit test for converting spherical to Cartesian coordinates."""
