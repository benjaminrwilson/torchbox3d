"""Dataset metaclass."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.Logger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class Dataset:
    """A dataset metaclass.

    Args:
        dataset_dir: Dataset directory.
        name: Name of the dataset.
        split: Split name for the dataset.
        classes: List of valid classes for targets.
    """

    dataset_dir: Path
    name: str
    split: str
    classes: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        """Initialize the dataset."""
        self.dataset_dir = Path(self.dataset_dir)
