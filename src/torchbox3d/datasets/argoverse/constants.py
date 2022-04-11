"""AV2 constants."""

from typing import Dict, Final, Tuple

from av2.datasets.sensor.constants import AnnotationCategories

LABEL_ATTR: Final[Tuple[str, ...]] = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
)

AV2_ANNO_NAMES_TO_INDEX: Final[Dict[str, int]] = {
    x.value: i for i, x in enumerate(AnnotationCategories)
}

DATASET_TO_TAXONOMY: Final[Dict[str, Dict[str, int]]] = {
    "av2": AV2_ANNO_NAMES_TO_INDEX,
}
