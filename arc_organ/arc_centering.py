# arc_centering.py
# Lee's ratio-based centering engine for ARC
# "Whatever the puzzle calls for we center in on it. Even inside odd? Don't matter, fill en."

from __future__ import annotations
from typing import Tuple, Iterable, Optional
import numpy as np


def _bbox_nonzero(grid: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find tight bounding box around non-zero pixels.
    Returns (row_min, row_max_inclusive, col_min, col_max_inclusive)
    or None if there are no non-zero cells.
    """
    nz = np.argwhere(grid != 0)
    if nz.size == 0:
        return None
    r_min, c_min = nz.min(axis=0)
    r_max, c_max = nz.max(axis=0)
    return int(r_min), int(r_max), int(c_min), int(c_max)


def extract_center_object(
    grid: np.ndarray,
    *,
    treat_zero_as_background: bool = True,
) -> np.ndarray:
    """
    Extract the "center object" from a grid.

    Rules (your law):
      - If treat_zero_as_background=True:
           * Use the tight bounding box around non-zero cells.
      - If treat_zero_as_background=False:
           * Use the whole grid as the object (e.g., 4x4 is ONE tile).

    Returned array is a copy, not a view.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)

    if not treat_zero_as_background:
        # Whole grid is the object (e.g., 4x4 center tile)
        return grid.copy()

    bbox = _bbox_nonzero(grid)
    if bbox is None:
        # No structure, just zeros → return as-is
        return grid.copy()

    r_min, r_max, c_min, c_max = bbox
    return grid[r_min : r_max + 1, c_min : c_max + 1].copy()


def make_centered_canvas(
    obj: np.ndarray,
    *,
    canvas_size: Optional[int] = None,
    pad_value: int = 0,
    force_odd: bool = True,
) -> np.ndarray:
    """
    Place an object (any HxW) on a square canvas, centered.

    - If canvas_size is None:
        * canvas_size = max(H, W)
        * if force_odd and canvas_size is even → canvas_size += 1
    - If canvas_size is given:
        * Used directly (optionally bumped to odd if force_odd=True)
    - Empty space is filled with pad_value (default 0).

    This is the literal:
      "4x4 center use that. whatever the puzzle calls for we center in on it.
       even inside odd? odd inside even? don't matter, fill en."
    """
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=int)

    h, w = obj.shape

    if canvas_size is None:
        canvas_size = max(h, w)
        if force_odd and canvas_size % 2 == 0:
            canvas_size += 1
    else:
        if force_odd and canvas_size % 2 == 0:
            canvas_size += 1

    if canvas_size < h or canvas_size < w:
        raise ValueError(
            f"Canvas {canvas_size}x{canvas_size} too small for object {h}x{w}"
        )

    canvas = np.full((canvas_size, canvas_size), pad_value, dtype=int)

    # Integer center placement: top-left where obj goes
    top = (canvas_size - h) // 2
    left = (canvas_size - w) // 2

    canvas[top : top + h, left : left + w] = obj
    return canvas


def multi_scale_canvases(
    grid: np.ndarray,
    *,
    base_size: Optional[int] = None,
    steps: Iterable[int] = (0, 2, 4),
    treat_zero_as_background: bool = True,
    pad_value: int = 0,
    force_odd: bool = True,
) -> dict[int, np.ndarray]:
    """
    Generate multiple centered canvases at increasing sizes
    (for your "if 5x5 noisy, try 7x7, 9x9..." idea).

    Returns:
        { canvas_size: canvas_ndarray }

    Example:
        obj = extract_center_object(grid)
        canvases = multi_scale_canvases(obj, base_size=5, steps=(0, 2, 4))
        # gets 5x5, 7x7, 9x9 canvases, all centered.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)

    obj = extract_center_object(grid, treat_zero_as_background=treat_zero_as_background)
    h, w = obj.shape

    if base_size is None:
        base_size = max(h, w)
        if force_odd and base_size % 2 == 0:
            base_size += 1

    result: dict[int, np.ndarray] = {}
    for step in steps:
        size = base_size + step
        if force_odd and size % 2 == 0:
            size += 1
        result[size] = make_centered_canvas(
            obj,
            canvas_size=size,
            pad_value=pad_value,
            force_odd=force_odd,
        )
    return result


# --- Convenience one-shot for ARC tasks ---

def canonical_center_view(
    grid: np.ndarray,
    *,
    treat_zero_as_background: bool = True,
    min_canvas_size: Optional[int] = None,
    pad_value: int = 0,
) -> np.ndarray:
    """
    High-level helper:
      1. Extract center object (respecting zeros-or-not).
      2. Center it on a minimal square canvas >= min_canvas_size.
      3. Force odd size to guarantee a unique center cell.

    This is a good default "center-fusion view" for ARC reasoning.
    """
    obj = extract_center_object(grid, treat_zero_as_background=treat_zero_as_background)
    h, w = obj.shape

    size = max(h, w)
    if min_canvas_size is not None:
        size = max(size, min_canvas_size)

    if size % 2 == 0:
        size += 1

    return make_centered_canvas(
        obj,
        canvas_size=size,
        pad_value=pad_value,
        force_odd=True,
    )


# --- Diff and growth detection layer ---

def centered_diff(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    *,
    treat_zero_as_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare input and output in canonical centered form.
    
    Returns:
        (centered_input, centered_output, diff_grid)
    
    Where:
        - centered_input: input object on canonical canvas
        - centered_output: output object on canonical canvas
        - diff_grid: output - input (0 = no change, nonzero = change)
    """
    # Extract both objects
    input_obj = extract_center_object(input_grid, treat_zero_as_background=treat_zero_as_background)
    output_obj = extract_center_object(output_grid, treat_zero_as_background=treat_zero_as_background)
    
    # Find canvas size that fits both
    in_h, in_w = input_obj.shape
    out_h, out_w = output_obj.shape
    canvas_size = max(in_h, in_w, out_h, out_w)
    if canvas_size % 2 == 0:
        canvas_size += 1
    
    # Center both on same canvas
    centered_in = make_centered_canvas(input_obj, canvas_size=canvas_size, force_odd=True)
    centered_out = make_centered_canvas(output_obj, canvas_size=canvas_size, force_odd=True)
    
    # Compute diff (marks where output changed from input)
    diff = np.where(centered_in != centered_out, centered_out, 0)
    
    return centered_in, centered_out, diff


def centered_growth_detector(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    *,
    treat_zero_as_background: bool = True,
) -> dict:
    """
    Detect growth patterns between input and output.
    
    Returns dict with:
        - input_shape: (h, w) of input object
        - output_shape: (h, w) of output object
        - growth_ratio: (h_ratio, w_ratio) as floats
        - growth_absolute: (h_delta, w_delta) as ints
        - growth_type: 'none', 'uniform', 'anisotropic', 'doubled', 'tiled'
    """
    input_obj = extract_center_object(input_grid, treat_zero_as_background=treat_zero_as_background)
    output_obj = extract_center_object(output_grid, treat_zero_as_background=treat_zero_as_background)
    
    in_h, in_w = input_obj.shape
    out_h, out_w = output_obj.shape
    
    h_ratio = out_h / in_h if in_h > 0 else 0
    w_ratio = out_w / in_w if in_w > 0 else 0
    
    h_delta = out_h - in_h
    w_delta = out_w - in_w
    
    # Classify growth type
    if h_delta == 0 and w_delta == 0:
        growth_type = 'none'
    elif abs(h_ratio - 2.0) < 0.1 and abs(w_ratio - 2.0) < 0.1:
        growth_type = 'doubled'
    elif abs(h_ratio - w_ratio) < 0.1:
        growth_type = 'uniform'
    else:
        growth_type = 'anisotropic'
    
    return {
        'input_shape': (in_h, in_w),
        'output_shape': (out_h, out_w),
        'growth_ratio': (h_ratio, w_ratio),
        'growth_absolute': (h_delta, w_delta),
        'growth_type': growth_type,
    }


def centered_color_roles(
    grid: np.ndarray,
    *,
    treat_zero_as_background: bool = True,
) -> dict:
    """
    Analyze color roles in a centered object.
    
    Returns dict with:
        - colors: list of non-background colors
        - color_counts: {color: count}
        - majority_color: most frequent color
        - minority_color: least frequent color (if 2 colors)
        - background: 0 (or None if treat_zero_as_background=False)
    """
    obj = extract_center_object(grid, treat_zero_as_background=treat_zero_as_background)
    
    unique_colors = np.unique(obj)
    if treat_zero_as_background:
        unique_colors = unique_colors[unique_colors != 0]
    
    color_counts = {}
    for color in unique_colors:
        color_counts[int(color)] = int(np.sum(obj == color))
    
    if len(color_counts) == 0:
        return {
            'colors': [],
            'color_counts': {},
            'majority_color': None,
            'minority_color': None,
            'background': 0 if treat_zero_as_background else None,
        }
    
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    majority = sorted_colors[0][0]
    minority = sorted_colors[-1][0] if len(sorted_colors) > 1 else None
    
    return {
        'colors': [int(c) for c in unique_colors],
        'color_counts': color_counts,
        'majority_color': majority,
        'minority_color': minority,
        'background': 0 if treat_zero_as_background else None,
    }
