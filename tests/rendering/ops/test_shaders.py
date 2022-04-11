"""Unit tests for shader ops."""

from tempfile import NamedTemporaryFile
from typing import Any, Callable

import pytest
import torch
from torch import Tensor

from torchbox3d.rendering.ops.shaders import (
    blend,
    circles,
    clip_to_viewport,
    line2,
    linear_interpolation,
    normal2,
    polygon,
)
from torchbox3d.utils.io import write_img

#######################################################################
# TESTS                                                               #
#######################################################################


@pytest.mark.parametrize(
    "uv, tex, height, width, uv_, tex_, is_inside_viewport_",
    [
        pytest.param(
            torch.as_tensor([[10, 10], [102, 10]]),
            torch.as_tensor([[10, 10, 10], [20, 20, 20]]),
            100,
            100,
            torch.as_tensor([[10, 10]]),
            torch.as_tensor([[10, 10, 10]]),
            torch.as_tensor([True, False]),
        )
    ],
)
def test_clip_to_viewport(
    uv: Tensor,
    tex: Tensor,
    height: int,
    width: int,
    uv_: Tensor,
    tex_: Tensor,
    is_inside_viewport_: Tensor,
) -> None:
    """Unit test for clipping points to the viewport.

    Args:
        uv: (N,2) UV coordinates.
        tex: (N,3) Texture intensity values.
        height: Height of the viewport.
        width: Width of the viewport.
        uv_: Expected UV coordinates.
        tex_: Expected texture intensity values.
        is_inside_viewport_: Expected boolean validity mask.
    """
    uv, tex, is_inside_viewport = clip_to_viewport(uv, tex, height, width)
    torch.testing.assert_allclose(uv, uv_)
    torch.testing.assert_allclose(tex, tex_)
    torch.testing.assert_allclose(is_inside_viewport, is_inside_viewport_)


@pytest.mark.parametrize(
    "pix_fg, pix_bg, alpha, blended_pix_",
    [
        pytest.param(
            torch.full((4, 10, 10, 3), 128),
            torch.full((4, 10, 10, 3), 64),
            0.5,
            torch.full((4, 10, 10, 3), fill_value=96),
        )
    ],
    ids=["Test blending 50% on each image."],
)
def test_blend_pixels(
    pix_fg: Tensor, pix_bg: Tensor, alpha: float, blended_pix_: Tensor
) -> None:
    """Unit test for blending foreground and background pixels.

    Args:
        pix_fg: (...,3,H,W) Source RGB image.
        pix_bg: (...,3,H,W) Target RGB image.
        alpha: Alpha blending coefficient.
        blended_pix_: Expected blended pixels.
    """
    blended_pix = blend(
        foreground_pixels=pix_fg, background_pixels=pix_bg, alpha=alpha
    )
    torch.testing.assert_allclose(blended_pix, blended_pix_)


@pytest.mark.parametrize(
    "uvz, img",
    [
        pytest.param(
            torch.as_tensor([[[18, 18, 1], [25, 25, 1]]]),
            torch.zeros((3, 37, 37)),
        )
    ],
)
def test_draw_circles(uvz: Tensor, img: Tensor) -> None:
    """Unit test for drawing circles on an image."""
    uvz[..., 2] = 1
    uvz = torch.randint(0, 38, size=(1, 10, 3))
    img = circles(
        uvz[0], torch.full_like(uvz, 255.0)[0].float(), img, antialias=False
    )
    with NamedTemporaryFile("w") as f:
        write_img(img.byte(), f"{f.name}.png")


@pytest.mark.parametrize(
    "p1, p2, color, img",
    [
        pytest.param(
            torch.as_tensor([512, 512], dtype=torch.float),
            torch.as_tensor([1024, 1024], dtype=torch.float),
            torch.as_tensor([0, 255, 0], dtype=torch.uint8),
            torch.zeros((3, 2048, 2048), dtype=torch.uint8),
        )
    ],
)
def test_line2(p1: Tensor, p2: Tensor, color: Tensor, img: Tensor) -> None:
    """Unit test for drawing a line on an image."""
    line2(p1, p2, color, img)

    with NamedTemporaryFile("w") as f:
        write_img(img.byte(), f"{f.name}.png")


@pytest.mark.parametrize(
    "vertices, edge_indices, colors, img",
    [
        pytest.param(
            torch.as_tensor(
                [[512, 768], [1024, 500], [1440, 1440], [600, 800]],
                dtype=torch.float,
            ),
            torch.as_tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            torch.as_tensor(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                dtype=torch.uint8,
            ),
            torch.zeros((3, 2048, 2048), dtype=torch.uint8),
        ),
        pytest.param(
            torch.as_tensor(
                [[512, 512], [512, 1024], [1024, 1024], [1024, 512]],
                dtype=torch.float,
            ),
            torch.as_tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            torch.as_tensor(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                dtype=torch.uint8,
            ),
            torch.zeros((3, 2048, 2048), dtype=torch.uint8),
        ),
    ],
)
def test_polygon(
    vertices: Tensor, edge_indices: Tensor, colors: Tensor, img: Tensor
) -> None:
    """Unit test for drawing a polygon on an image."""
    polygon(vertices, edge_indices, colors, img, width_px=20)
    with NamedTemporaryFile("w") as f:
        write_img(img.byte(), f"{f.name}.png")


@pytest.mark.parametrize(
    "p1, p2, normal_",
    [
        pytest.param(
            torch.as_tensor([5, 5]),
            torch.as_tensor([10, 10]),
            torch.as_tensor([-0.7071, 0.7071]),
        )
    ],
)
def test_normal2(p1: Tensor, p2: Tensor, normal_: Tensor) -> None:
    """Unit test for computing the normal vector of a line."""
    normal = normal2(p1, p2)
    torch.testing.assert_allclose(normal, normal_)


@pytest.mark.parametrize(
    "p1, p2, size",
    [
        pytest.param(
            torch.as_tensor([512, 512], dtype=torch.float),
            torch.as_tensor([1024, 1024], dtype=torch.float),
            1000,
        )
    ],
)
def test_interpolate(p1: Tensor, p2: Tensor, size: int) -> None:
    """Unit test for interpolating two points (p1,p2)."""
    points = torch.stack((p1, p2))[None]
    linear_interpolation(points, size)


#######################################################################
# BENCHMARKS                                                          #
#######################################################################


@pytest.mark.parametrize(
    "uvz, tex, img, radius, antialias",
    [
        pytest.param(
            torch.randint(0, 2048, size=(1, 50000, 3)),
            torch.full((1, 50000, 3), 255.0)[0].float(),
            torch.zeros((3, 2048, 2048)),
            3,
            True,
        )
    ],
    ids=["Test drawing random circles in an image."],
)
def test_benchmark_draw_circles(
    benchmark: Callable[..., Any],
    uvz: Tensor,
    tex: Tensor,
    img: Tensor,
    radius: int,
    antialias: bool,
) -> None:
    """Benchmark drawing circles in an image."""
    benchmark(
        circles,
        uvz=uvz,
        tex=tex,
        img=img,
        radius=radius,
        antialias=antialias,
    )
    with NamedTemporaryFile("w") as f:
        write_img(img.byte(), f"{f.name}.png")
    write_img(img.byte(), "circles.png")


@pytest.mark.parametrize(
    "p1, p2, color, img",
    [
        pytest.param(
            torch.as_tensor([512, 512], dtype=torch.float),
            torch.as_tensor([1024, 1024], dtype=torch.float),
            torch.as_tensor([0, 255, 0], dtype=torch.uint8),
            torch.zeros((3, 2048, 2048), dtype=torch.uint8),
        )
    ],
    ids=["Benchmark drawing a line in an image."],
)
def test_benchmark_line(
    benchmark: Callable[..., Any],
    p1: Tensor,
    p2: Tensor,
    color: Tensor,
    img: Tensor,
) -> None:
    """Benchmark drawing a line on an image."""
    benchmark(line2, p1, p2, color, img)


@pytest.mark.parametrize(
    "vertices, edge_indices, colors, img",
    [
        pytest.param(
            torch.as_tensor(
                [[512, 512], [1024, 512], [1024, 1024], [512, 1024]],
                dtype=torch.float,
            ),
            torch.as_tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            torch.as_tensor(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                dtype=torch.uint8,
            ),
            torch.zeros((3, 2048, 2048), dtype=torch.uint8),
        )
    ],
    ids=["Benchmark drawing a polygon."],
)
def test_benchmark_polygon(
    benchmark: Callable[..., Any],
    vertices: Tensor,
    edge_indices: Tensor,
    colors: Tensor,
    img: Tensor,
) -> None:
    """Benchmark drawing a polygon on an image."""
    benchmark(polygon, vertices, edge_indices, colors, img)
