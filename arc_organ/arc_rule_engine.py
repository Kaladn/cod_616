# cod_616/arc_organ/arc_rule_engine.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np


class SpatialTransform(Enum):
    IDENTITY = auto()
    ROTATE_90 = auto()
    ROTATE_180 = auto()
    ROTATE_270 = auto()
    FLIP_H = auto()
    FLIP_V = auto()


class ColorTransform(Enum):
    IDENTITY = auto()
    INVERT = auto()
    MAP_MAX_TO_MIN = auto()
    MAP_MIN_TO_MAX = auto()


class ObjectTransform(Enum):
    IDENTITY = auto()
    EXTRACT_LARGEST = auto()
    REMOVE_SMALLEST = auto()


class MaskMode(Enum):
    FULL = auto()
    DELTA_ONLY = auto()
    FOREGROUND_ONLY = auto()


@dataclass
class ARCRuleVector:
    """
    Six-channel rule vector that the Recognition Field can produce.

    This is intentionally discrete/enum-based to stay symbolic.
    """
    spatial: SpatialTransform = SpatialTransform.IDENTITY
    color: ColorTransform = ColorTransform.IDENTITY
    obj: ObjectTransform = ObjectTransform.IDENTITY
    mask: MaskMode = MaskMode.FULL
    reserved1: float = 0.0
    reserved2: float = 0.0

    def as_dict(self) -> Dict[str, str]:
        return {
            "spatial": self.spatial.name,
            "color": self.color.name,
            "obj": self.obj.name,
            "mask": self.mask.name,
            "reserved1": str(self.reserved1),
            "reserved2": str(self.reserved2),
        }


class ARCRuleEngine:
    """
    Applies ARCRuleVector to an input grid and optional masks
    to produce an output grid.
    """

    def apply(
        self,
        input_grid: np.ndarray,
        rule: ARCRuleVector,
        delta_map: Optional[np.ndarray] = None,
        foreground_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if input_grid.ndim != 2:
            raise ValueError(f"Expected 2D grid, got {input_grid.shape}")

        grid = input_grid.copy()

        # 1) Spatial transform
        grid = self._apply_spatial(grid, rule.spatial)

        # 2) Object-level transform
        grid = self._apply_object_transform(grid, rule.obj)

        # 3) Color transform
        grid = self._apply_color_transform(grid, rule.color)

        # 4) Masking mode (where transformation is allowed to "materialize")
        grid = self._apply_mask_mode(
            input_grid=input_grid,
            transformed_grid=grid,
            mask_mode=rule.mask,
            delta_map=delta_map,
            foreground_mask=foreground_mask,
        )

        return grid

    @staticmethod
    def _apply_spatial(grid: np.ndarray, t: SpatialTransform) -> np.ndarray:
        if t == SpatialTransform.IDENTITY:
            return grid
        if t == SpatialTransform.ROTATE_90:
            return np.rot90(grid, k=1)
        if t == SpatialTransform.ROTATE_180:
            return np.rot90(grid, k=2)
        if t == SpatialTransform.ROTATE_270:
            return np.rot90(grid, k=3)
        if t == SpatialTransform.FLIP_H:
            return np.fliplr(grid)
        if t == SpatialTransform.FLIP_V:
            return np.flipud(grid)
        return grid

    @staticmethod
    def _apply_color_transform(grid: np.ndarray, t: ColorTransform) -> np.ndarray:
        g = grid.astype(np.int32)
        if t == ColorTransform.IDENTITY:
            return g

        vals = np.unique(g)
        if vals.size == 0:
            return g

        min_v = int(vals.min())
        max_v = int(vals.max())

        if t == ColorTransform.INVERT:
            # Simple inversion in [min, max]
            return (max_v - (g - min_v)).astype(np.int32)

        if t == ColorTransform.MAP_MAX_TO_MIN:
            out = g.copy()
            out[g == max_v] = min_v
            return out

        if t == ColorTransform.MAP_MIN_TO_MAX:
            out = g.copy()
            out[g == min_v] = max_v
            return out

        return g

    @staticmethod
    def _apply_object_transform(grid: np.ndarray, t: ObjectTransform) -> np.ndarray:
        from scipy.ndimage import label as cc_label

        g = grid.copy()
        if t == ObjectTransform.IDENTITY:
            return g

        fg = (g != 0).astype(np.int32)
        labeled, num = cc_label(fg)
        if num == 0:
            return g

        sizes = []
        for cid in range(1, num + 1):
            sizes.append(((labeled == cid).sum(), cid))
        if not sizes:
            return g

        sizes.sort()
        smallest_size, smallest_cid = sizes[0]
        largest_size, largest_cid = sizes[-1]

        if t == ObjectTransform.EXTRACT_LARGEST:
            mask = (labeled == largest_cid)
            new_grid = np.zeros_like(g)
            new_grid[mask] = g[mask]
            return new_grid

        if t == ObjectTransform.REMOVE_SMALLEST:
            mask = (labeled == smallest_cid)
            g[mask] = 0
            return g

        return g

    @staticmethod
    def _apply_mask_mode(
        input_grid: np.ndarray,
        transformed_grid: np.ndarray,
        mask_mode: MaskMode,
        delta_map: Optional[np.ndarray],
        foreground_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if mask_mode == MaskMode.FULL:
            return transformed_grid

        h, w = input_grid.shape
        out = input_grid.copy()

        if mask_mode == MaskMode.DELTA_ONLY and delta_map is not None:
            mask = delta_map > 0.5
            mask = mask[:h, :w]
            out[mask] = transformed_grid[mask]
            return out

        if mask_mode == MaskMode.FOREGROUND_ONLY and foreground_mask is not None:
            mask = foreground_mask.astype(bool)
            mask = mask[:h, :w]
            out[mask] = transformed_grid[mask]
            return out

        return transformed_grid
