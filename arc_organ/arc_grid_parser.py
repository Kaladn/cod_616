# cod_616/arc_organ/arc_grid_parser.py

"""
ARC Grid Parser - Phase B: 6-Channel Sensory Organ

Updated to use Phase A pixel metadata for richer channel extraction.
Each channel now leverages structured metadata instead of raw pixel analysis.

Parallel to ScreenResonanceState (COD Vision Organ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as cc_label

# Phase A integration
from .arc_pixel_metadata import extract_grid_metadata, GridMetadata


@dataclass
class ArcGridPair:
    """
    Simple container for an ARC-style input/output pair.
    """
    input_grid: np.ndarray  # (H, W), int
    output_grid: Optional[np.ndarray] = None  # (H, W), int or None for test-only


@dataclass
class ArcChannelBundle:
    """
    Six 2D channels extracted from an ARC grid.

    Each channel is float32 in [0, 1] where possible, same shape as the grid.
    """
    color_normalized: np.ndarray        # C0
    component_ids: np.ndarray           # C1 (normalized connected-component id)
    symmetry_score_map: np.ndarray      # C2
    repetition_score_map: np.ndarray    # C3
    shape_signature_map: np.ndarray     # C4
    delta_map: np.ndarray               # C5 (requires reference grid, else zeros)


def _normalize_grid_colors(grid: np.ndarray) -> np.ndarray:
    """Normalize integer colors to [0, 1] based on min/max present."""
    grid = grid.astype(np.int32)
    unique_vals = np.unique(grid)
    if unique_vals.size == 1:
        return np.zeros_like(grid, dtype=np.float32)
    min_v = unique_vals.min()
    max_v = unique_vals.max()
    return ((grid - min_v) / float(max_v - min_v)).astype(np.float32)


def _compute_component_ids(grid: np.ndarray) -> np.ndarray:
    """Connected-component labels normalized to [0, 1]."""
    # Treat 0 as background, others as foreground
    fg = (grid != 0).astype(np.int32)
    labeled, num = cc_label(fg)
    if num == 0:
        return np.zeros_like(grid, dtype=np.float32)
    return (labeled.astype(np.float32) / float(num)).astype(np.float32)


def _compute_symmetry_map(grid: np.ndarray) -> np.ndarray:
    """Per-cell participation in symmetry (rough heuristic)."""
    h, w = grid.shape
    sym_map = np.zeros_like(grid, dtype=np.float32)

    # Horizontal symmetry
    if w > 1:
        left = grid[:, : w // 2]
        right = np.fliplr(grid[:, - (w // 2):])
        min_w = min(left.shape[1], right.shape[1])
        if min_w > 0:
            eq_h = (left[:, :min_w] == right[:, :min_w]).astype(np.float32)
            sym_map[:, :min_w] += eq_h / 2.0
            sym_map[:, w - min_w :] += eq_h / 2.0

    # Vertical symmetry
    if h > 1:
        top = grid[: h // 2, :]
        bottom = np.flipud(grid[- (h // 2):, :])
        min_h = min(top.shape[0], bottom.shape[0])
        if min_h > 0:
            eq_v = (top[:min_h, :] == bottom[:min_h, :]).astype(np.float32)
            sym_map[:min_h, :] += eq_v / 2.0
            sym_map[h - min_h :, :] += eq_v / 2.0

    return np.clip(sym_map, 0.0, 1.0)


def _compute_repetition_map(grid: np.ndarray) -> np.ndarray:
    """
    Rough repetition score per cell:
    - For each value, how many times does it appear relative to grid size?
    """
    h, w = grid.shape
    total = float(h * w)
    counts = {}
    for v in np.unique(grid):
        counts[int(v)] = float((grid == v).sum()) / total
    rep = np.zeros_like(grid, dtype=np.float32)
    for v, frac in counts.items():
        rep[grid == v] = frac
    return rep


def _compute_shape_signature_map(grid: np.ndarray) -> np.ndarray:
    """
    Very rough "shape importance" map:
    - Larger connected components get higher scores.
    """
    fg = (grid != 0).astype(np.int32)
    labeled, num = cc_label(fg)
    if num == 0:
        return np.zeros_like(grid, dtype=np.float32)

    sizes = np.bincount(labeled.ravel())
    sizes = sizes.astype(np.float32)
    sizes[0] = 0.0  # background

    max_size = sizes.max() if sizes.size > 0 else 1.0
    if max_size <= 0:
        return np.zeros_like(grid, dtype=np.float32)

    norm_sizes = sizes / max_size
    sig = norm_sizes[labeled]
    return sig.astype(np.float32)


def _compute_delta_map(
    input_grid: np.ndarray,
    output_grid: Optional[np.ndarray],
) -> np.ndarray:
    """
    Per-cell change indicator between input and output grids.
    If no output grid is provided (test mode), returns zeros.
    """
    if output_grid is None:
        return np.zeros_like(input_grid, dtype=np.float32)

    ih, iw = input_grid.shape
    oh, ow = output_grid.shape
    h = min(ih, oh)
    w = min(iw, ow)

    delta = np.zeros_like(input_grid, dtype=np.float32)
    patch_in = input_grid[:h, :w]
    patch_out = output_grid[:h, :w]
    changed = (patch_in != patch_out).astype(np.float32)
    delta[:h, :w] = changed
    return delta


def compute_arc_channels(
    input_grid: np.ndarray,
    output_grid: Optional[np.ndarray] = None,
) -> ArcChannelBundle:
    """
    Main entry point: given an ARC-style grid (and optional output grid),
    compute the 6-channel bundle for the resonance engine.
    """
    if input_grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {input_grid.shape}")

    color_norm = _normalize_grid_colors(input_grid)
    comp_ids = _compute_component_ids(input_grid)
    sym_map = _compute_symmetry_map(input_grid)
    rep_map = _compute_repetition_map(input_grid)
    shape_map = _compute_shape_signature_map(input_grid)
    delta = _compute_delta_map(input_grid, output_grid)

    return ArcChannelBundle(
        color_normalized=color_norm,
        component_ids=comp_ids,
        symmetry_score_map=sym_map,
        repetition_score_map=rep_map,
        shape_signature_map=shape_map,
        delta_map=delta,
    )


# ============================================================================
# PHASE 2: METADATA-ENHANCED CHANNELS
# ============================================================================

def compute_arc_channels_v2(
    input_grid: np.ndarray,
    output_grid: Optional[np.ndarray] = None,
) -> ArcChannelBundle:
    """
    Phase 2 version: Use pixel metadata for richer channel extraction.
    
    This leverages Phase A metadata extraction for:
    - Better component detection (with size ranking)
    - Accurate boundary detection
    - Tiling/repetition detection
    - Symmetry group assignment
    
    Returns same ArcChannelBundle format but with metadata-enhanced values.
    """
    # Extract metadata
    metadata = extract_grid_metadata(input_grid)
    H, W = metadata.shape
    
    # CH1: Color (normalized)
    color_channel = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            color_channel[y, x] = metadata.pixel_metadata[y, x].normalized_color
    
    # CH2: Component (using metadata ranks)
    component_channel = np.zeros((H, W), dtype=np.float32)
    if metadata.num_components > 0:
        for y in range(H):
            for x in range(W):
                rank = metadata.pixel_metadata[y, x].component_rank
                component_channel[y, x] = rank / max(metadata.num_components, 1)
    
    # CH3: Border/Interior (structural role)
    # Encode as: 0.0 = background, 0.33 = interior, 0.66 = border, 1.0 = isolated
    border_channel = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            meta = metadata.pixel_metadata[y, x]
            if meta.is_background:
                border_channel[y, x] = 0.0
            elif meta.is_border:
                border_channel[y, x] = 0.66
            elif meta.is_interior:
                border_channel[y, x] = 0.33
            else:
                border_channel[y, x] = 1.0  # Isolated pixel
    
    # CH4: Symmetry groups
    symmetry_channel = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            group = metadata.pixel_metadata[y, x].symmetry_group
            # 0=none, 1=vertical, 2=horizontal, 3=both
            symmetry_channel[y, x] = group / 3.0
    
    # CH5: Tiling groups
    tiling_channel = np.zeros((H, W), dtype=np.float32)
    if metadata.is_tiled:
        max_tile_id = 0
        for y in range(H):
            for x in range(W):
                tile_id = metadata.pixel_metadata[y, x].tile_group
                max_tile_id = max(max_tile_id, tile_id)
        
        for y in range(H):
            for x in range(W):
                tile_id = metadata.pixel_metadata[y, x].tile_group
                tiling_channel[y, x] = tile_id / max(max_tile_id, 1)
    
    # CH6: Delta (if output provided)
    delta_channel = _compute_delta_map(input_grid, output_grid)
    
    return ArcChannelBundle(
        color_normalized=color_channel,
        component_ids=component_channel,
        symmetry_score_map=symmetry_channel,
        repetition_score_map=tiling_channel,  # Renamed semantically
        shape_signature_map=border_channel,    # Renamed semantically
        delta_map=delta_channel,
    )
