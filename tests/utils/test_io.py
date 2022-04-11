"""Unit tests for the io module."""

from tempfile import NamedTemporaryFile
from typing import Any, Callable

import pytest
import torch
from torch import Tensor
from torchvision.io.image import read_image, write_jpeg

from torchbox3d.utils.io import read_img, write_img


@pytest.mark.parametrize(
    "img_, ext, write_image",
    [
        pytest.param(
            torch.zeros((3, 32, 32), dtype=torch.uint8),
            ".jpg",
            write_jpeg,
        ),
        pytest.param(
            torch.ones((3, 32, 32), dtype=torch.uint8),
            ".jpg",
            write_jpeg,
        ),
        pytest.param(
            torch.randint(0, 255, size=(3, 32, 32), dtype=torch.uint8),
            ".png",
            write_img,
        ),
    ],
    ids=[
        "Test reading a JPG with all zeros.",
        "Test reading a JPG with all ones.",
        "Test reading a PNG with random values in the interval [0,255].",
    ],
)
def test_read_img(
    img_: Tensor, ext: str, write_image: Callable[..., Any]
) -> None:
    """Unit test for reading images."""
    with NamedTemporaryFile("w") as f:
        path = f.name + ext
        write_image(img_, path)
        img = read_img(path)
        torch.testing.assert_allclose(img, img_)


@pytest.mark.parametrize(
    "img_, ext",
    [
        pytest.param(torch.zeros((3, 32, 32), dtype=torch.uint8), ".jpg"),
        pytest.param(torch.ones((3, 32, 32), dtype=torch.uint8), ".jpg"),
        pytest.param(
            torch.randint(0, 255, size=(3, 32, 32), dtype=torch.uint8), ".png"
        ),
    ],
    ids=[
        "Test writing a JPG with all zeros.",
        "Test writing a JPG with all ones.",
        "Test writing a PNG with random values in the interval [0,255].",
    ],
)
def test_write_img(img_: Tensor, ext: str) -> None:
    """Unit test for writing images."""
    with NamedTemporaryFile("w") as f:
        path = f"{f.name}{ext}"
        write_img(img_, path)
        img = read_image(path)
        torch.testing.assert_allclose(img, img_)
