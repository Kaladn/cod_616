# cod_616/arc_organ/arc_resonance_state.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .arc_grid_parser import ArcChannelBundle


@dataclass
class ARCResonanceState:
    """
    20-dimensional ARC resonance state.

    The layout mirrors the philosophy of ScreenResonanceState:
      0-3:  global energy / stability
      4-9:  channel dominance / contrast
      10-13: repetition / symmetry rhythms
      14-17: structure / shape clarity
      18-19: transformation priors (delta-based)
    """

    # Global structure / energy
    energy_total: float
    energy_var: float
    sparsity: float
    color_entropy: float

    # Channel dominance and contrast
    color_component_dominance: float
    symmetry_strength: float
    repetition_strength: float
    shape_mass: float
    component_fragmentation: float
    foreground_ratio: float

    # Rhythm / repetition / pattern
    row_pattern_score: float
    col_pattern_score: float
    block_repetition_score: float
    checkerboard_tendency: float

    # Structure / clarity
    shape_clarity: float
    boundary_complexity: float
    symmetry_axis_confidence: float
    component_size_variance: float

    # Transformation priors
    delta_coverage: float
    delta_focus: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_channels(ch: ArcChannelBundle) -> "ARCResonanceState":
        """
        Build ARCResonanceState from the 6-channel bundle.
        """
        # Flatten everything for scalar stats
        color = ch.color_normalized.ravel()
        comp = ch.component_ids.ravel()
        sym = ch.symmetry_score_map.ravel()
        rep = ch.repetition_score_map.ravel()
        shape = ch.shape_signature_map.ravel()
        delta = ch.delta_map.ravel()

        # Basic masks
        foreground_mask = color > 0.0
        fg_count = float(foreground_mask.sum())
        total = float(color.size) if color.size > 0 else 1.0

        # Global energy & sparsity
        energy_total = float(np.mean(color)) if color.size else 0.0
        energy_var = float(np.var(color)) if color.size else 0.0
        sparsity = 1.0 - (fg_count / total)

        # Color entropy
        unique_vals, counts = np.unique(color, return_counts=True)
        probs = counts.astype(np.float64) / float(counts.sum()) if counts.sum() > 0 else None
        if probs is None or probs.size == 0:
            color_entropy = 0.0
        else:
            color_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        # Channel dominance / contrast
        color_component_dominance = float(np.max(rep)) if rep.size else 0.0
        symmetry_strength = float(np.mean(sym)) if sym.size else 0.0
        repetition_strength = float(np.mean(rep)) if rep.size else 0.0
        shape_mass = float(np.mean(shape)) if shape.size else 0.0

        # Component fragmentation: many small blobs vs few big ones
        comp_ids = np.unique(comp[comp > 0.0])
        fragmentation = float(len(comp_ids)) / total if total > 0 else 0.0

        foreground_ratio = fg_count / total

        # Row/column pattern scores (simple autocorrelation heuristic)
        # Reshape to 2D
        h, w = ch.color_normalized.shape
        row_pattern_score = 0.0
        col_pattern_score = 0.0
        if h > 1 and w > 1:
            # Row similarity
            row_corrs = []
            for i in range(h - 1):
                row1 = ch.color_normalized[i, :]
                row2 = ch.color_normalized[i + 1, :]
                if np.std(row1) > 0 or np.std(row2) > 0:
                    num = np.dot(row1 - row1.mean(), row2 - row2.mean())
                    den = (np.std(row1) * np.std(row2) + 1e-6)
                    row_corrs.append(num / den)
            if row_corrs:
                row_pattern_score = float(np.mean(row_corrs))

            # Column similarity
            col_corrs = []
            for j in range(w - 1):
                col1 = ch.color_normalized[:, j]
                col2 = ch.color_normalized[:, j + 1]
                if np.std(col1) > 0 or np.std(col2) > 0:
                    num = np.dot(col1 - col1.mean(), col2 - col2.mean())
                    den = (np.std(col1) * np.std(col2) + 1e-6)
                    col_corrs.append(num / den)
            if col_corrs:
                col_pattern_score = float(np.mean(col_corrs))

        # Block repetition: 2x2 tiles repeated
        block_repetition_score = 0.0
        if h >= 2 and w >= 2:
            blocks = []
            for i in range(0, h - 1):
                for j in range(0, w - 1):
                    blocks.append(ch.color_normalized[i : i + 2, j : j + 2].ravel())
            blocks = np.array(blocks)
            if len(blocks) > 1:
                # Compare blocks to the mean block
                mean_block = blocks.mean(axis=0)
                sims = []
                for b in blocks:
                    num = np.dot(b - b.mean(), mean_block - mean_block.mean())
                    den = (np.std(b) * np.std(mean_block) + 1e-6)
                    sims.append(num / den)
                block_repetition_score = float(np.mean(sims))

        # Checkerboard tendency: alternating patterns
        checker_pattern = np.indices((h, w)).sum(axis=0) % 2
        checker_corr = 0.0
        if h > 0 and w > 0:
            flat_checker = checker_pattern.ravel().astype(np.float32)
            flat_col = ch.color_normalized.ravel()
            if np.std(flat_checker) > 0 and np.std(flat_col) > 0:
                num = np.dot(flat_checker - flat_checker.mean(), flat_col - flat_col.mean())
                den = (np.std(flat_checker) * np.std(flat_col) + 1e-6)
                checker_corr = num / den
        checkerboard_tendency = float(checker_corr)

        # Structure / clarity
        shape_clarity = float(np.mean(shape[foreground_mask])) if fg_count > 0 else 0.0

        # Boundary complexity: edges between different colors
        boundary_complexity = 0.0
        if h > 1 and w > 1:
            diff_h = (ch.color_normalized[1:, :] != ch.color_normalized[:-1, :]).astype(np.float32)
            diff_v = (ch.color_normalized[:, 1:] != ch.color_normalized[:, :-1]).astype(np.float32)
            boundary_complexity = float((diff_h.sum() + diff_v.sum()) / (h * w))

        # Symmetry axis confidence: how strong symmetry is overall
        symmetry_axis_confidence = symmetry_strength

        # Component size variance
        comp_sizes = []
        for cid in comp_ids:
            comp_sizes.append(float((comp == cid).sum()))
        if comp_sizes:
            component_size_variance = float(np.var(np.array(comp_sizes)))
        else:
            component_size_variance = 0.0

        # Transformation priors from delta
        if delta.size > 0:
            delta_coverage = float((delta > 0.5).sum()) / total
            # concentration: how much of delta mass is in top 10% most-changed cells
            sorted_delta = np.sort(delta)[::-1]
            k = max(1, int(0.1 * sorted_delta.size))
            top_mass = float(sorted_delta[:k].sum())
            total_mass = float(sorted_delta.sum() + 1e-6)
            delta_focus = top_mass / total_mass
        else:
            delta_coverage = 0.0
            delta_focus = 0.0

        return ARCResonanceState(
            energy_total=energy_total,
            energy_var=energy_var,
            sparsity=sparsity,
            color_entropy=color_entropy,
            color_component_dominance=color_component_dominance,
            symmetry_strength=symmetry_strength,
            repetition_strength=repetition_strength,
            shape_mass=shape_mass,
            component_fragmentation=fragmentation,
            foreground_ratio=foreground_ratio,
            row_pattern_score=row_pattern_score,
            col_pattern_score=col_pattern_score,
            block_repetition_score=block_repetition_score,
            checkerboard_tendency=checkerboard_tendency,
            shape_clarity=shape_clarity,
            boundary_complexity=boundary_complexity,
            symmetry_axis_confidence=symmetry_axis_confidence,
            component_size_variance=component_size_variance,
            delta_coverage=delta_coverage,
            delta_focus=delta_focus,
        )
