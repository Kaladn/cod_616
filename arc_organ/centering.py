"""
Centering primitives for ARC grids.

This encodes Lee's centering law:

- Whatever the puzzle gives you (4x4, 6x6, etc.) is the "center object".
- We treat the non-background region as the semantic object.
- Even vs odd sizes do not matter; we center the object in the target grid.
- If we need to embed the object into a larger grid, we "fill in" with
  background color (usually the dominant color, often 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


Grid = np.ndarray  # shape: (H, W), dtype: int


@dataclass
class CenterObjectMeta:
    """Metadata about the extracted center object."""
    background: int
    bbox: Tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max)
    original_shape: Tuple[int, int]
    object_shape: Tuple[int, int]


def infer_background(grid: Grid) -> int:
    """
    Infer the background color as the most frequent value in the grid.

    Typically this will be 0 in ARC, but we don't hard-code that here.
    """
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def extract_center_object(
    grid: Grid,
    background: Optional[int] = None,
    treat_zeros_as_background: bool = True,
) -> Tuple[Grid, CenterObjectMeta]:
    """
    Extract the semantic "center object" from a grid.

    - If treat_zeros_as_background=True, we treat all non-zero cells as object.
    - Otherwise, we treat all cells != background as object.
    - If the entire grid is background, we treat the whole grid as the object.

    Returns:
        object_grid: a cropped grid containing only the object region
        meta: metadata describing background and original bounding box
    """
    h, w = grid.shape
    if background is None:
        background = infer_background(grid)

    if treat_zeros_as_background:
        mask = grid != 0
    else:
        mask = grid != background

    if not mask.any():
        # No foreground detected; treat whole grid as the object.
        y_min, y_max, x_min, x_max = 0, h, 0, w
    else:
        ys, xs = np.where(mask)
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1

    obj = grid[y_min:y_max, x_min:x_max].copy()
    meta = CenterObjectMeta(
        background=int(background),
        bbox=(y_min, y_max, x_min, x_max),
        original_shape=(h, w),
        object_shape=obj.shape,
    )
    return obj, meta


def place_at_center(
    obj: Grid,
    out_shape: Tuple[int, int],
    background: int,
) -> Grid:
    """
    Place an object grid at the conceptual center of an output canvas.

    This does not care about even vs odd; it uses integer division to pick
    a stable center placement.

    Args:
        obj:       object grid (h, w)
        out_shape: (H, W) of output grid
        background: fill value for the rest of the canvas

    Returns:
        Grid of shape out_shape with obj written into its center.
    """
    H, W = out_shape
    h, w = obj.shape

    if h > H or w > W:
        raise ValueError(
            f"Object {obj.shape} does not fit in target canvas {out_shape}"
        )

    canvas = np.full((H, W), background, dtype=obj.dtype)

    top = (H - h) // 2
    left = (W - w) // 2

    canvas[top:top + h, left:left + w] = obj
    return canvas


def canonical_center_view(
    grid: Grid,
    target_shape: Optional[Tuple[int, int]] = None,
    treat_zeros_as_background: bool = True,
) -> Tuple[Grid, CenterObjectMeta]:
    """
    Canonicalize a grid into a "centered object" view.

    Steps:
      1) Infer background.
      2) Extract the semantic object (non-background region).
      3) If target_shape is provided, center the object inside that canvas.
         Otherwise, we just return the cropped object.

    This is the primitive that lets us say:
      - 4x4 input -> treat whole 4x4 region as "center object"
      - Even vs odd does not matter
      - When embedding into larger grids, we fill with background

    Args:
        grid: input ARC grid (H, W)
        target_shape: optional (H_out, W_out) for embedding
        treat_zeros_as_background: if True, use (grid != 0) as object mask

    Returns:
        centered_grid: either cropped object, or object centered in target canvas
        meta: metadata about extraction and background
    """
    obj, meta = extract_center_object(
        grid,
        background=None,
        treat_zeros_as_background=treat_zeros_as_background,
    )

    if target_shape is None:
        return obj, meta

    centered = place_at_center(
        obj=obj,
        out_shape=target_shape,
        background=meta.background,
    )
    return centered, meta
