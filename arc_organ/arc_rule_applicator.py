# cod_616/arc_organ/arc_rule_applicator.py

"""
Phase E: Rule Application - Hypothesis → Output Grid

Transforms test input grids according to detected rule hypotheses.
Each rule family has a dedicated applicator function.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .arc_rule_hypothesis import RuleFamily, RuleHypothesis


class RuleApplicator:
    """
    Applies rule hypotheses to test input grids.
    
    Each rule family has a specific application strategy that transforms
    input → output based on learned parameters.
    """
    
    @staticmethod
    def apply(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rule hypothesis to test input, generate two attempts.
        
        Args:
            hypothesis: Detected rule with parameters
            test_input: Test case input grid
        
        Returns:
            (attempt_1, attempt_2): Two candidate output grids
        """
        if hypothesis.family == RuleFamily.TILING_EXPAND:
            return RuleApplicator._apply_tiling_expand(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.SELF_REFERENTIAL_TILING:
            return RuleApplicator._apply_self_referential_tiling(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.SANDWICH_TILING:
            return RuleApplicator._apply_sandwich_tiling(hypothesis, test_input)
        elif hypothesis.family == RuleFamily.DIAGONAL_MARKER_TILING:
            return RuleApplicator._apply_diagonal_marker_tiling(hypothesis, test_input)
        elif hypothesis.family == RuleFamily.LATIN_SQUARE_DIAGONAL:
            return RuleApplicator._apply_latin_square_diagonal(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.HORIZONTAL_REPLICATE:
            return RuleApplicator._apply_horizontal_replicate(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.PURE_RECOLOR:
            return RuleApplicator._apply_pure_recolor(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.RECOLOR_MAPPING:
            return RuleApplicator._apply_recolor_mapping(hypothesis, test_input)
        
        elif hypothesis.family == RuleFamily.IDENTITY:
            return test_input.copy(), test_input.copy()
        
        else:
            # Unknown family - return input unchanged
            return test_input.copy(), test_input.copy()
    
    # ========================================================================
    # TILING_EXPAND: Repeat tile / scale grid
    # ========================================================================
    
    @staticmethod
    def _apply_tiling_expand(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply tiling expansion: repeat input grid factor_h × factor_w times.
        
        Example: 3×3 input → 9×9 output (factor 3×3)
        """
        factor_h = hypothesis.params.get("tile_factor_h", 1)
        factor_w = hypothesis.params.get("tile_factor_w", 1)
        
        # Attempt 1: Direct tiling
        attempt_1 = np.tile(test_input, (factor_h, factor_w))
        
        # Attempt 2: Tiling with slight variation (transpose first, then tile)
        # This handles cases where tile orientation matters
        transposed = test_input.T
        attempt_2 = np.tile(transposed, (factor_w, factor_h))
        
        # If transpose doesn't make sense (non-square), use original
        if attempt_2.shape[0] * attempt_2.shape[1] != attempt_1.shape[0] * attempt_1.shape[1]:
            attempt_2 = attempt_1.copy()
        
        return attempt_1, attempt_2
    
    # ========================================================================
    # SELF_REFERENTIAL_TILING: Tile with self-masking (007bbfb7)
    # ========================================================================
    
    @staticmethod
    def _apply_self_referential_tiling(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply self-referential masked tiling.
        
        Rule: Tile the input factor×factor times, then zero out any block at (i,j)
        if the corresponding pixel (i,j) in the original tile is background (0).
        
        Example: 3×3 tile with 0s at (0,0) and (2,0) → 9×9 output where
        blocks (0,0) and (2,0) are all zeros.
        """
        factor_h = hypothesis.params.get("factor_h", 1)
        factor_w = hypothesis.params.get("factor_w", 1)
        background_value = hypothesis.params.get("background_value", 0)
        
        h, w = test_input.shape
        output = np.zeros((h * factor_h, w * factor_w), dtype=test_input.dtype)
        
        # Attempt 1: Apply self-referential masking
        for i_block in range(factor_h):
            for j_block in range(factor_w):
                row_start = i_block * h
                row_end = (i_block + 1) * h
                col_start = j_block * w
                col_end = (j_block + 1) * w
                
                # Check if this block position corresponds to background in tile
                if i_block < h and j_block < w:
                    if test_input[i_block, j_block] == background_value:
                        # Zero out this block
                        output[row_start:row_end, col_start:col_end] = 0
                    else:
                        # Copy tile
                        output[row_start:row_end, col_start:col_end] = test_input
                else:
                    # Outside original tile bounds - copy tile
                    output[row_start:row_end, col_start:col_end] = test_input
        
        attempt_1 = output
        
        # Attempt 2: Try inverse masking (foreground-based instead of background)
        output2 = np.tile(test_input, (factor_h, factor_w))
        for i_block in range(factor_h):
            for j_block in range(factor_w):
                if i_block < h and j_block < w:
                    if test_input[i_block, j_block] != background_value:
                        # Zero out non-background positions
                        row_start = i_block * h
                        row_end = (i_block + 1) * h
                        col_start = j_block * w
                        col_end = (j_block + 1) * w
                        output2[row_start:row_end, col_start:col_end] = 0
        
        attempt_2 = output2
        
        return attempt_1, attempt_2
    
    # ========================================================================
    # SANDWICH_TILING: Tile 3× with middle flipped (00576224)
    # ========================================================================
    
    @staticmethod
    def _apply_sandwich_tiling(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sandwich tiling: tile 3× horizontally with column-flip in middle.
        
        Rule for 2×2 input → 6×6 output:
        - Rows 1-2: tile input 3×
        - Rows 3-4: flip columns, then tile 3×
        - Rows 5-6: repeat rows 1-2
        
        Pattern: normal, flipped, normal
        """
        # Attempt 1: Sandwich pattern (horizontal flip in middle)
        rows_1_2 = np.tile(test_input, (1, 3))
        inp_flipped = np.flip(test_input, axis=1)  # flip columns
        rows_3_4 = np.tile(inp_flipped, (1, 3))
        rows_5_6 = rows_1_2
        
        attempt_1 = np.vstack([rows_1_2, rows_3_4, rows_5_6])
        
        # Attempt 2: Try vertical flip variant
        rows_1_2_v = np.tile(test_input, (1, 3))
        inp_flipped_v = np.flip(test_input, axis=0)  # flip rows instead
        rows_3_4_v = np.tile(inp_flipped_v, (1, 3))
        rows_5_6_v = rows_1_2_v
        
        attempt_2 = np.vstack([rows_1_2_v, rows_3_4_v, rows_5_6_v])
        
        return attempt_1, attempt_2
    
    # ========================================================================
    # DIAGONAL_MARKER_TILING: Tile 3× with diagonal markers (310f3251)
    # ========================================================================
    
    @staticmethod
    def _apply_diagonal_marker_tiling(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply diagonal marker tiling: tile 3×3, mark up-left diagonals from non-zeros.
        
        For each non-zero at (r,c):
        - Calculate diagonal: ((r-1) % h, (c-1) % w)
        - If that position is 0, place 2 there
        """
        h, w = test_input.shape
        
        # Tile 3×3
        output = np.tile(test_input, (3, 3))
        
        # Find non-zeros and their diagonals
        for r in range(h):
            for c in range(w):
                if test_input[r, c] != 0:
                    # Calculate diagonal position (up-left with wraparound)
                    diag_r = (r - 1) % h
                    diag_c = (c - 1) % w
                    
                    # If that position is 0 in input, mark with 2
                    if test_input[diag_r, diag_c] == 0:
                        # Place 2 at this position in all tiles
                        for tile_r in range(3):
                            for tile_c in range(3):
                                out_r = tile_r * h + diag_r
                                out_c = tile_c * w + diag_c
                                output[out_r, out_c] = 2
        
        # Attempt 2: Try other directional variants (down-right diagonal)
        output_2 = np.tile(test_input, (3, 3))
        
        for r in range(h):
            for c in range(w):
                if test_input[r, c] != 0:
                    diag_r = (r + 1) % h
                    diag_c = (c + 1) % w
                    
                    if test_input[diag_r, diag_c] == 0:
                        for tile_r in range(3):
                            for tile_c in range(3):
                                out_r = tile_r * h + diag_r
                                out_c = tile_c * w + diag_c
                                output_2[out_r, out_c] = 2
        
        return output, output_2
    
    # ========================================================================
    # RECOLOR_MAPPING: Structure preserved, colors remapped
    # ========================================================================
    
    @staticmethod
    def _apply_recolor_mapping(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply color remapping to test input.
        
        Example: All pixels with color 1 → color 2
        """
        color_map = hypothesis.params.get("color_map", {})
        
        # Attempt 1: Apply learned mapping
        attempt_1 = test_input.copy()
        for in_color, out_color in color_map.items():
            attempt_1[test_input == in_color] = out_color
        
        # Attempt 2: Apply mapping + also map background (0) if not already mapped
        attempt_2 = attempt_1.copy()
        if 0 not in color_map and len(color_map) > 0:
            # Guess: background might map to most common non-background color in mapping
            target_colors = list(color_map.values())
            if target_colors:
                # Keep background as-is (conservative)
                pass
        
        # If attempts are identical, try inverting the mapping as second attempt
        if np.array_equal(attempt_1, attempt_2):
            # Invert mapping: if 1→2, try 2→1
            inverted_map = {v: k for k, v in color_map.items()}
            attempt_2 = test_input.copy()
            for in_color, out_color in inverted_map.items():
                attempt_2[test_input == in_color] = out_color
        
        return attempt_1, attempt_2
    
    # HORIZONTAL_REPLICATE: Tile horizontally N× (a416b8f3)
    @staticmethod
    def _apply_horizontal_replicate(hypothesis, test_input):
        """Apply horizontal N× replication."""
        repeat_factor = hypothesis.params.get('repeat_factor', 2)
        
        # Attempt 1: Standard horizontal replication
        attempt_1 = np.tile(test_input, (1, repeat_factor))
        
        # Attempt 2: Try repeat_factor + 1 in case detection was off by 1
        attempt_2 = np.tile(test_input, (1, repeat_factor + 1))
        
        return attempt_1, attempt_2
    
    # PURE_RECOLOR: Apply A→B color mapping (b1948b0a, c8f0f002, d511f180)
    @staticmethod
    def _apply_pure_recolor(hypothesis, test_input):
        """Apply pure color substitution mapping."""
        mapping = hypothesis.params.get('mapping', {})
        
        # Attempt 1: Apply the mapping directly
        attempt_1 = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                color = test_input[r, c]
                if color in mapping:
                    attempt_1[r, c] = mapping[color]
        
        # Attempt 2: Try inverted mapping (in case direction was wrong)
        inverted_map = {v: k for k, v in mapping.items() if k != v}
        attempt_2 = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                color = test_input[r, c]
                if color in inverted_map:
                    attempt_2[r, c] = inverted_map[color]
        
        return attempt_1, attempt_2
    
    # ========================================================================
    # LATIN_SQUARE_DIAGONAL: Generate Latin square from diagonal colors
    # ========================================================================
    
    @staticmethod
    def _apply_latin_square_diagonal(
        hypothesis: RuleHypothesis,
        test_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Latin square tiling from diagonal colors."""
        from .arc_operators import LatinSquareFromDiagonalOperator
        
        operator = LatinSquareFromDiagonalOperator()
        
        # Infer output shape from input (assume same size for now)
        params = {"output_shape": test_input.shape}
        
        # Generate output
        result = operator.apply(test_input, params)
        
        # Both attempts are the same (deterministic)
        return result, result
    
    # ========================================================================
    # Future rule families:
    # - MIRROR_FLIP: Apply flip/rotation transforms
    # - EXTRACT_LARGEST: Keep only largest component
    # - CROP_REGION: Extract sub-region
    # - ADD_BORDER: Draw frame around structure
    # - LOCAL_PAINT: Modify localized region
    # ========================================================================
