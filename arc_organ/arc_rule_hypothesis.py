# cod_616/arc_organ/arc_rule_hypothesis.py

"""
Phase E: Rule Hypothesis System

Maps task signatures → ranked transformation hypotheses.
Each hypothesis is a testable rule family with parameters and confidence.

This is where metadata-driven reasoning happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple

import numpy as np


class RuleFamily(Enum):
    """High-level ARC transformation families."""
    TILING_EXPAND = auto()              # Repeat tile / scale grid (simple)
    SELF_REFERENTIAL_TILING = auto()    # Tile with self-masking (007bbfb7)
    SANDWICH_TILING = auto()            # Tile 3x with middle flipped (00576224)
    DIAGONAL_MARKER_TILING = auto()     # Tile 3x with diagonal markers (310f3251)
    LATIN_SQUARE_DIAGONAL = auto()      # Latin square from diagonal colors (05269061)
    HORIZONTAL_REPLICATE = auto()       # Horizontal N× replication (a416b8f3)
    PURE_RECOLOR = auto()               # Pure A→B color substitution (b1948b0a, c8f0f002)
    MIRROR_FLIP = auto()                # Vertical/horizontal/rotational symmetry
    RECOLOR_MAPPING = auto()    # Structure preserved, colors remapped (017c7c7b)
    EXTRACT_LARGEST = auto()    # Keep biggest object / remove clutter
    CROP_REGION = auto()        # Select sub-region (top/bottom/left/right/center)
    ADD_BORDER = auto()         # Draw frame around structure
    LOCAL_PAINT = auto()        # Modify localized region (025d127b)
    IDENTITY = auto()           # No change (baseline)


@dataclass
class RuleHypothesis:
    """
    A testable hypothesis about what transformation rule applies to a task.
    
    Attributes:
        family: High-level rule category
        confidence: Belief strength [0.0, 1.0]
        params: Family-specific parameters for application
        reasoning: Human-readable explanation
    """
    family: RuleFamily
    confidence: float
    params: Dict[str, Any]
    reasoning: str
    
    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"RuleHypothesis({self.family.name}, conf={self.confidence:.2f}, {param_str})"


class RuleDetector:
    """
    Base class for rule family detectors.
    Each detector analyzes task signature + training examples → hypothesis.
    """
    
    def detect(
        self,
        train_examples: list,
        task_signature: Any,
    ) -> Optional[RuleHypothesis]:
        """
        Analyze task and return hypothesis if confidence exceeds threshold.
        
        Args:
            train_examples: List of ArcGridPair training examples
            task_signature: Fused resonance state from Phase D
        
        Returns:
            RuleHypothesis if detected, None otherwise
        """
        raise NotImplementedError


class TilingExpandDetector(RuleDetector):
    """
    Detects grid expansion via perfect tiling repetition.
    
    Signals:
    - mean_tiling_strength (D11) ≈ 1.0
    - Output dimensions are integer multiples of input
    - Component patterns repeat, don't change
    
    Target: Tasks like 007bbfb7 (3×3 → 9×9)
    """
    
    def detect(
        self,
        train_examples: list,
        task_signature: Any,
    ) -> Optional[RuleHypothesis]:
        """Detect tiling expansion pattern."""
        # Check task-level tiling strength (Phase C D11)
        tiling_strength = task_signature.mean_tiling_strength
        
        if tiling_strength < 0.8:
            return None  # Not a tiling task
        
        # Analyze size relationships in training examples
        size_ratios = []
        all_integer_multiples = True
        
        for example in train_examples:
            if example.output_grid is None:
                continue
            
            h_in, w_in = example.input_grid.shape
            h_out, w_out = example.output_grid.shape
            
            # Check if output is integer multiple of input
            if h_in == 0 or w_in == 0:
                continue
            
            if h_out % h_in != 0 or w_out % w_in != 0:
                all_integer_multiples = False
                break
            
            factor_h = h_out // h_in
            factor_w = w_out // w_in
            size_ratios.append((factor_h, factor_w))
        
        if not all_integer_multiples or len(size_ratios) == 0:
            return None
        
        # Check consistency of tiling factors
        unique_ratios = set(size_ratios)
        if len(unique_ratios) != 1:
            return None  # Inconsistent tiling
        
        factor_h, factor_w = size_ratios[0]
        
        # Verify tiling pattern by checking actual repetition
        verified = self._verify_tiling(train_examples, factor_h, factor_w)
        
        if not verified:
            return None
        
        # Build hypothesis
        confidence = 0.85 + (tiling_strength - 0.8) * 0.5  # [0.85, 0.95]
        confidence = min(0.95, confidence)
        
        return RuleHypothesis(
            family=RuleFamily.TILING_EXPAND,
            confidence=confidence,
            params={
                "tile_factor_h": factor_h,
                "tile_factor_w": factor_w,
            },
            reasoning=f"Perfect tiling detected: {factor_h}×{factor_w} expansion, tiling_strength={tiling_strength:.2f}",
        )
    
    def _verify_tiling(
        self,
        train_examples: list,
        factor_h: int,
        factor_w: int,
    ) -> bool:
        """Verify that output is actually tiled repetition of input."""
        for example in train_examples:
            if example.output_grid is None:
                continue
            
            input_grid = example.input_grid
            output_grid = example.output_grid
            
            h_in, w_in = input_grid.shape
            h_out, w_out = output_grid.shape
            
            # Check if output matches tiled input
            expected = np.tile(input_grid, (factor_h, factor_w))
            
            if expected.shape != output_grid.shape:
                return False
            
            # Allow some tolerance for noise
            match_ratio = np.mean(expected == output_grid)
            if match_ratio < 0.95:
                return False
        
        return True


class RecolorMappingDetector(RuleDetector):
    """
    Detects color remapping with preserved spatial structure.
    
    Signals:
    - Input/output grids same size
    - Spatial structure (components, borders) unchanged
    - Consistent color mapping across examples
    
    Target: Tasks like 017c7c7b (color 1→2 mapping)
    """
    
    def detect(
        self,
        train_examples: list,
        task_signature: Any,
    ) -> Optional[RuleHypothesis]:
        """Detect color remapping pattern."""
        # Collect color mappings from each example
        mappings = []
        all_same_size = True
        
        for example in train_examples:
            if example.output_grid is None:
                continue
            
            input_grid = example.input_grid
            output_grid = example.output_grid
            
            # Check size preservation
            if input_grid.shape != output_grid.shape:
                all_same_size = False
                break
            
            # Build color mapping for this example
            mapping = self._extract_color_mapping(input_grid, output_grid)
            if mapping is None:
                return None  # Spatial structure changed
            
            mappings.append(mapping)
        
        if not all_same_size or len(mappings) == 0:
            return None
        
        # Find consensus mapping across examples
        consensus_map = self._find_consensus_mapping(mappings)
        
        if consensus_map is None or len(consensus_map) == 0:
            return None
        
        # Check spatial structure preservation
        structure_preserved = self._verify_structure_preservation(train_examples)
        
        if not structure_preserved:
            return None
        
        # Build hypothesis
        confidence = 0.80 + len(consensus_map) * 0.02  # Higher confidence with more mappings
        confidence = min(0.95, confidence)
        
        mapping_str = ", ".join(f"{k}→{v}" for k, v in sorted(consensus_map.items()))
        
        return RuleHypothesis(
            family=RuleFamily.RECOLOR_MAPPING,
            confidence=confidence,
            params={
                "color_map": consensus_map,
            },
            reasoning=f"Consistent color remapping: {mapping_str}",
        )
    
    def _extract_color_mapping(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
    ) -> Optional[Dict[int, int]]:
        """Extract color mapping from input→output, verify it's consistent."""
        mapping = {}
        
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = int(input_grid[i, j])
                out_color = int(output_grid[i, j])
                
                if in_color in mapping:
                    # Check consistency
                    if mapping[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    mapping[in_color] = out_color
        
        return mapping
    
    def _find_consensus_mapping(
        self,
        mappings: list[Dict[int, int]],
    ) -> Optional[Dict[int, int]]:
        """Find color mapping that appears consistently across examples."""
        if len(mappings) == 0:
            return None
        
        # Start with first mapping
        consensus = mappings[0].copy()
        
        # Verify it's consistent across all examples
        for mapping in mappings[1:]:
            for color, target in list(consensus.items()):
                if color in mapping:
                    if mapping[color] != target:
                        # Inconsistent - remove from consensus
                        del consensus[color]
        
        # Only keep mappings that actually change color
        consensus = {k: v for k, v in consensus.items() if k != v}
        
        return consensus if len(consensus) > 0 else None
    
    def _verify_structure_preservation(
        self,
        train_examples: list,
    ) -> bool:
        """Verify that spatial structure (components, adjacency) is preserved."""
        from scipy.ndimage import label
        
        for example in train_examples:
            if example.output_grid is None:
                continue
            
            input_grid = example.input_grid
            output_grid = example.output_grid
            
            # Count connected components in both
            in_labeled, in_count = label(input_grid > 0)
            out_labeled, out_count = label(output_grid > 0)
            
            # Component count should be similar (allow ±1 for noise)
            if abs(in_count - out_count) > 1:
                return False
        
        return True


class SelfReferentialTilingDetector:
    """
    Detects self-referential masked tiling (007bbfb7).
    
    Rule: Tile input factor×factor times, then zero out any block at (i,j)
    if the corresponding pixel (i,j) in the ORIGINAL TILE is background (0).
    
    Example: 3×3 tile with 0s at (0,0) and (2,0) → 9×9 output where
    blocks (0,0) and (2,0) are all zeros, rest are the tile.
    
    Detection:
    - High tiling strength (D11)
    - Integer expansion
    - Masked blocks match background positions in tile
    """
    
    @staticmethod
    def detect(
        train_examples: list,
        task_signature,
    ) -> Optional[RuleHypothesis]:
        """Detect self-referential masked tiling."""
        
        # Require high tiling strength
        if task_signature.mean_tiling_strength < 0.8:
            return None
        
        # Check all examples for consistent integer expansion
        factors = []
        for ex in train_examples:
            in_h, in_w = ex.input_grid.shape
            out_h, out_w = ex.output_grid.shape
            
            if out_h % in_h != 0 or out_w % in_w != 0:
                return None
            
            factor_h = out_h // in_h
            factor_w = out_w // in_w
            factors.append((factor_h, factor_w))
        
        # All examples must have same factors
        if len(set(factors)) != 1:
            return None
        
        factor_h, factor_w = factors[0]
        
        # Verify self-referential masking pattern on ALL examples
        for ex in train_examples:
            tile = ex.input_grid
            output = ex.output_grid
            h, w = tile.shape
            
            # Find background (0) positions in tile
            bg_positions = set()
            for i in range(h):
                for j in range(w):
                    if tile[i, j] == 0:
                        bg_positions.add((i, j))
            
            # Check if output blocks are correctly masked
            for i_block in range(factor_h):
                for j_block in range(factor_w):
                    row_start = i_block * h
                    row_end = (i_block + 1) * h
                    col_start = j_block * w
                    col_end = (j_block + 1) * w
                    block = output[row_start:row_end, col_start:col_end]
                    
                    if (i_block, j_block) in bg_positions:
                        # Should be all zeros
                        if not np.all(block == 0):
                            return None  # Pattern broken
                    else:
                        # Should match tile
                        if not np.array_equal(block, tile):
                            return None  # Pattern broken
            
            # This example passed - continue to next
        
        # All examples match self-referential masking pattern
        confidence = 0.92 + task_signature.mean_tiling_strength * 0.05
        confidence = min(confidence, 0.97)
        
        return RuleHypothesis(
            family=RuleFamily.SELF_REFERENTIAL_TILING,
            confidence=confidence,
            params={
                "factor_h": factor_h,
                "factor_w": factor_w,
                "background_value": 0,
            },
            reasoning=f"Self-referential tiling: {factor_h}×{factor_w} with background-based masking"
        )


class SandwichTilingDetector:
    """
    Detects sandwich tiling pattern (00576224).
    
    Rule: Input is 2×2, output is 6×6
    - Rows 1-2: tile input horizontally 3×
    - Rows 3-4: flip columns, then tile 3×
    - Rows 5-6: repeat rows 1-2
    
    Pattern: normal, flipped, normal (sandwich)
    """
    
    @staticmethod
    def detect(
        train_examples: list,
        task_signature,
    ) -> Optional[RuleHypothesis]:
        """Detect sandwich tiling pattern."""
        
        # Check if all examples have 2×2 input → 6×6 output
        for ex in train_examples:
            if ex.input_grid.shape != (2, 2):
                return None
            if ex.output_grid.shape != (6, 6):
                return None
        
        # Verify sandwich pattern on all examples
        for ex in train_examples:
            inp = ex.input_grid
            out = ex.output_grid
            
            # Build expected output
            rows_1_2 = np.tile(inp, (1, 3))
            inp_flipped = np.flip(inp, axis=1)
            rows_3_4 = np.tile(inp_flipped, (1, 3))
            rows_5_6 = rows_1_2
            
            expected = np.vstack([rows_1_2, rows_3_4, rows_5_6])
            
            # Must match exactly
            if not np.array_equal(expected, out):
                return None
        
        # All examples match sandwich tiling
        confidence = 0.95  # Very specific pattern
        
        return RuleHypothesis(
            family=RuleFamily.SANDWICH_TILING,
            confidence=confidence,
            params={},
            reasoning="Sandwich tiling: 2×2 → 6×6 with column-flip in middle"
        )


class DiagonalMarkerTilingDetector:
    """
    Detects diagonal marker tiling pattern (310f3251).
    
    Rule: Tile input 3×3, then for each non-zero at (r,c):
    - Calculate diagonal position: ((r-1) % h, (c-1) % w)
    - If that input position is 0, place 2 there in output
    
    The 2 marks the up-left toroidal diagonal of each non-zero.
    """
    
    @staticmethod
    def detect(
        train_examples: list,
        task_signature,
    ) -> Optional[RuleHypothesis]:
        """Detect diagonal marker tiling pattern."""
        
        if not train_examples:
            return None
        
        # Check if all examples follow the pattern
        for ex in train_examples:
            inp = ex.input_grid
            out = ex.output_grid
            h, w = inp.shape
            
            # Must be 3× tiling
            if out.shape != (h * 3, w * 3):
                return None
            
            # Generate expected output
            expected = np.tile(inp, (3, 3))
            
            # Find non-zeros and their diagonals
            for r in range(h):
                for c in range(w):
                    if inp[r, c] != 0:
                        # Calculate diagonal position (up-left with wraparound)
                        diag_r = (r - 1) % h
                        diag_c = (c - 1) % w
                        
                        # If that position is 0 in input, mark with 2
                        if inp[diag_r, diag_c] == 0:
                            # Place 2 at this position in all tiles
                            for tile_r in range(3):
                                for tile_c in range(3):
                                    out_r = tile_r * h + diag_r
                                    out_c = tile_c * w + diag_c
                                    expected[out_r, out_c] = 2
            
            # Check if expected matches actual output
            if not np.array_equal(expected, out):
                return None
        
        # All examples match diagonal marker tiling
        return RuleHypothesis(
            family=RuleFamily.DIAGONAL_MARKER_TILING,
            confidence=0.95,
            params={'factor': 3},
            reasoning="Diagonal marker tiling: 3×3 expansion with 2s at up-left diagonals from non-zeros"
        )


class HorizontalReplicateDetector:
    """
    Detects horizontal N× replication: output = input repeated horizontally.
    
    Pure primitive operator:
    - No pixel changes
    - No rotation
    - No flipping
    - output_height == input_height
    - output_width = N × input_width
    
    Solves: a416b8f3 and similar horizontal replication puzzles
    """
    
    @staticmethod
    def detect(train_examples, task_signature):
        """Detect horizontal N× replication pattern."""
        if not train_examples:
            return None
        
        repeat_factor = None
        
        for example in train_examples:
            inp = example.input_grid
            out = example.output_grid
            
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Must keep same height
            if h_out != h_in:
                return None
            
            # Width must be perfect multiple
            if w_out % w_in != 0:
                return None
            
            factor = w_out // w_in
            
            # Must be at least 2× replication
            if factor < 2:
                return None
            
            # Check if first example or verify consistency
            if repeat_factor is None:
                repeat_factor = factor
            elif repeat_factor != factor:
                return None  # Inconsistent replication factor
            
            # Build expected replication
            expected = np.tile(inp, (1, repeat_factor))
            
            # Must match exactly
            if not np.array_equal(expected, out):
                return None
        
        # All examples match horizontal replication
        return RuleHypothesis(
            family=RuleFamily.HORIZONTAL_REPLICATE,
            confidence=0.98,
            params={'repeat_factor': repeat_factor},
            reasoning=f"Horizontal {repeat_factor}× replication: output = input repeated {repeat_factor} times horizontally"
        )


class PureRecolorDetector:
    """
    Detects pure A→B color substitution with no geometry changes.
    
    Pure primitive operator:
    - Input and output same shape
    - Every differing pixel follows consistent A→B mapping
    - All other values preserved exactly
    - No geometry, structure, or spatial changes
    - One-to-one mapping
    
    Solves: b1948b0a, c8f0f002, d511f180 and dozens of ARC recolor puzzles
    """
    
    @staticmethod
    def detect(train_examples, task_signature):
        """Detect pure color substitution pattern."""
        if not train_examples:
            return None
        
        consensus_mapping = None
        
        for example in train_examples:
            inp = example.input_grid
            out = example.output_grid
            
            # Same shape required
            if inp.shape != out.shape:
                return None
            
            h, w = inp.shape
            mapping = {}   # input_color → output_color
            reverse = {}   # output_color → input_color (for 1-to-1 check)
            
            for r in range(h):
                for c in range(w):
                    a = inp[r, c]
                    b = out[r, c]
                    
                    if a not in mapping:
                        mapping[a] = b
                    elif mapping[a] != b:
                        return None  # Inconsistent mapping
                    
                    # Ensure one-to-one (bijection)
                    if b not in reverse:
                        reverse[b] = a
                    elif reverse[b] != a:
                        return None  # Multiple inputs map to same output
            
            # Reject trivial identity (no actual recoloring)
            if all(k == v for k, v in mapping.items()):
                return None
            
            # Check consistency across examples
            if consensus_mapping is None:
                consensus_mapping = mapping
            else:
                # Verify mapping is consistent (all changed colors map the same way)
                for color, target in mapping.items():
                    if color in consensus_mapping:
                        if consensus_mapping[color] != target:
                            return None  # Inconsistent mapping across examples
        
        if consensus_mapping is None:
            return None
        
        # Count how many colors actually change
        changed_colors = sum(1 for k, v in consensus_mapping.items() if k != v)
        
        if changed_colors == 0:
            return None
        
        # Build reasoning string
        changes = [f"{k}→{v}" for k, v in consensus_mapping.items() if k != v]
        reasoning = f"Pure recoloring: {', '.join(changes[:3])}"
        if len(changes) > 3:
            reasoning += f" (+{len(changes)-3} more)"
        
        # All examples match pure recoloring
        return RuleHypothesis(
            family=RuleFamily.PURE_RECOLOR,
            confidence=0.97,
            params={'mapping': consensus_mapping},
            reasoning=reasoning
        )


class LatinSquareFromDiagonalDetector(RuleDetector):
    """
    Detects Latin square tiling pattern from diagonal input.
    
    Signals:
    - Input has 3 colors on diagonals (r+c = constant)
    - Output is 3×3 Latin square rotation tiled to grid
    - Output uses linear sequence: palette[(r*WIDTH + c) % 3]
    
    Target: Task 05269061
    """
    
    def detect(
        self,
        train_examples: list,
        task_signature: Any,
    ) -> Optional[RuleHypothesis]:
        """Detect Latin square from diagonal pattern."""
        from .arc_operators import LatinSquareFromDiagonalOperator
        
        operator = LatinSquareFromDiagonalOperator()
        matches = 0
        
        for example in train_examples:
            if example.output_grid is None:
                continue
            
            # Try to analyze with operator
            params = operator.analyze(example.input_grid, example.output_grid)
            if params is not None:
                # Verify it applies correctly
                predicted = operator.apply(example.input_grid, params)
                if np.array_equal(predicted, example.output_grid):
                    matches += 1
        
        # Require all training examples to match
        if matches == len(train_examples) and matches > 0:
            return RuleHypothesis(
                family=RuleFamily.LATIN_SQUARE_DIAGONAL,
                confidence=0.95,
                params={},
                reasoning=f"Latin square tiling from diagonal colors ({matches} examples)",
            )
        
        return None
