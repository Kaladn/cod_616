# cod_616/arc_organ/arc_recognition_field.py

"""
ARC Recognition Field - Transform classification from fused resonance.

Maps fused 20-dim resonance → discrete transformation class.
Pure decision logic, no ML. Based on pattern thresholds.

This is CompuCog's "reasoning layer" - detecting what rule applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from .arc_example_fuser import FusedResonanceState
from .arc_rule_engine import (
    ARCRuleVector,
    SpatialTransform,
    ColorTransform,
    ObjectTransform,
    MaskMode,
)


class TransformClass(Enum):
    """High-level ARC transformation categories."""
    IDENTITY = auto()
    SPATIAL_FLIP_H = auto()
    SPATIAL_FLIP_V = auto()
    SPATIAL_ROTATE_90 = auto()
    SPATIAL_ROTATE_180 = auto()
    SPATIAL_ROTATE_270 = auto()
    COLOR_RECOLOR = auto()
    COLOR_INVERT = auto()
    OBJECT_EXTRACT_LARGEST = auto()
    OBJECT_REMOVE_SMALLEST = auto()
    PATTERN_TILE = auto()
    PATTERN_EXTRACT_REGION = auto()
    GRID_CROP_TO_BBOX = auto()
    GRID_EXPAND = auto()


@dataclass
class RecognizedTransform:
    """
    Output of Recognition Field: classified transformation + confidence.
    """
    transform_class: TransformClass
    confidence: float  # 0.0-1.0
    rule_vector: ARCRuleVector
    reasoning: str  # Human-readable explanation


class ARCRecognitionField:
    """
    CompuCog Recognition Field adapted for ARC Prize.
    
    Input: FusedResonanceState (aggregated from training examples)
    Output: RecognizedTransform (what rule to apply)
    
    Decision logic based on resonance thresholds - no ML training required.
    """
    
    @staticmethod
    def recognize(fused: FusedResonanceState) -> RecognizedTransform:
        """
        Main recognition logic: fused resonance → transformation class.
        
        Uses decision tree based on resonance feature thresholds.
        """
        # Run all classifiers and pick highest confidence
        candidates = [
            ARCRecognitionField._detect_identity(fused),
            ARCRecognitionField._detect_flip_h(fused),
            ARCRecognitionField._detect_flip_v(fused),
            ARCRecognitionField._detect_rotate_90(fused),
            ARCRecognitionField._detect_rotate_180(fused),
            ARCRecognitionField._detect_color_recolor(fused),
            ARCRecognitionField._detect_extract_largest(fused),
            ARCRecognitionField._detect_crop_to_bbox(fused),
            ARCRecognitionField._detect_grid_expand(fused),
        ]
        
        # Sort by confidence, return best
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[0]
    
    @staticmethod
    def recognize_top_n(fused: FusedResonanceState, n: int = 2) -> List[RecognizedTransform]:
        """
        Get top N most likely transformations (for two-attempt sampling).
        """
        candidates = [
            ARCRecognitionField._detect_identity(fused),
            ARCRecognitionField._detect_flip_h(fused),
            ARCRecognitionField._detect_flip_v(fused),
            ARCRecognitionField._detect_rotate_90(fused),
            ARCRecognitionField._detect_rotate_180(fused),
            ARCRecognitionField._detect_color_recolor(fused),
            ARCRecognitionField._detect_extract_largest(fused),
            ARCRecognitionField._detect_crop_to_bbox(fused),
            ARCRecognitionField._detect_grid_expand(fused),
        ]
        
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[:n]
    
    # ========== Individual Transform Detectors ==========
    
    @staticmethod
    def _detect_identity(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect if transformation is identity (no change)."""
        # Low delta coverage = minimal change
        if fused.mean_delta_coverage < 0.05:
            confidence = 0.9 - fused.mean_delta_coverage * 2
            return RecognizedTransform(
                transform_class=TransformClass.IDENTITY,
                confidence=max(0.0, min(1.0, confidence)),
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.IDENTITY,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning="Minimal change detected (delta < 5%)",
            )
        return RecognizedTransform(
            transform_class=TransformClass.IDENTITY,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Significant changes present",
        )
    
    @staticmethod
    def _detect_flip_h(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect horizontal flip."""
        # Strong row pattern + high spatial consistency + moderate delta
        if (fused.mean_row_pattern_score > 2.0 and
            fused.spatial_consistency > 0.5 and
            0.3 < fused.mean_delta_coverage < 0.8):
            
            confidence = min(0.85, fused.spatial_consistency * 0.8 + 0.2)
            return RecognizedTransform(
                transform_class=TransformClass.SPATIAL_FLIP_H,
                confidence=confidence,
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.FLIP_H,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Row pattern={fused.mean_row_pattern_score:.2f}, spatial_consistency={fused.spatial_consistency:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.SPATIAL_FLIP_H,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Row pattern insufficient",
        )
    
    @staticmethod
    def _detect_flip_v(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect vertical flip."""
        # Strong column pattern + high spatial consistency
        if (fused.mean_col_pattern_score > 2.0 and
            fused.spatial_consistency > 0.5 and
            0.3 < fused.mean_delta_coverage < 0.8):
            
            confidence = min(0.85, fused.spatial_consistency * 0.8 + 0.2)
            return RecognizedTransform(
                transform_class=TransformClass.SPATIAL_FLIP_V,
                confidence=confidence,
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.FLIP_V,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Col pattern={fused.mean_col_pattern_score:.2f}, spatial_consistency={fused.spatial_consistency:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.SPATIAL_FLIP_V,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Column pattern insufficient",
        )
    
    @staticmethod
    def _detect_rotate_90(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect 90-degree rotation."""
        # High symmetry + balanced row/col patterns
        if (fused.mean_symmetry_strength > 0.6 and
            abs(fused.mean_row_pattern_score - fused.mean_col_pattern_score) < 0.5 and
            fused.spatial_consistency > 0.4):
            
            confidence = min(0.75, fused.mean_symmetry_strength * 0.7 + 0.1)
            return RecognizedTransform(
                transform_class=TransformClass.SPATIAL_ROTATE_90,
                confidence=confidence,
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.ROTATE_90,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Symmetry={fused.mean_symmetry_strength:.2f}, balanced patterns",
            )
        return RecognizedTransform(
            transform_class=TransformClass.SPATIAL_ROTATE_90,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Symmetry/pattern balance insufficient",
        )
    
    @staticmethod
    def _detect_rotate_180(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect 180-degree rotation."""
        # Very high symmetry + high delta
        if (fused.mean_symmetry_strength > 0.8 and
            fused.mean_delta_coverage > 0.5 and
            fused.spatial_consistency > 0.6):
            
            confidence = min(0.8, fused.mean_symmetry_strength * 0.8)
            return RecognizedTransform(
                transform_class=TransformClass.SPATIAL_ROTATE_180,
                confidence=confidence,
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.ROTATE_180,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"High symmetry={fused.mean_symmetry_strength:.2f}, delta={fused.mean_delta_coverage:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.SPATIAL_ROTATE_180,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Symmetry insufficient for 180° rotation",
        )
    
    @staticmethod
    def _detect_color_recolor(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect color remapping."""
        # High delta + high color consistency + low spatial change
        if (fused.mean_delta_coverage > 0.4 and
            fused.color_consistency > 0.5 and
            fused.spatial_consistency > 0.7):
            
            confidence = min(0.8, fused.color_consistency * 0.7 + 0.2)
            return RecognizedTransform(
                transform_class=TransformClass.COLOR_RECOLOR,
                confidence=confidence,
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.IDENTITY,
                    color=ColorTransform.MAP_MIN_TO_MAX,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Delta={fused.mean_delta_coverage:.2f}, color_consistency={fused.color_consistency:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.COLOR_RECOLOR,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="Color change pattern unclear",
        )
    
    @staticmethod
    def _detect_extract_largest(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect extracting largest object."""
        # Low delta (small change) + size reduction + low fragmentation
        if (fused.mean_delta_coverage < 0.3 and
            fused.size_change_ratio < 0.8 and
            fused.mean_component_fragmentation < 0.2):
            
            confidence = 0.7 - fused.mean_delta_coverage
            return RecognizedTransform(
                transform_class=TransformClass.OBJECT_EXTRACT_LARGEST,
                confidence=max(0.0, min(1.0, confidence)),
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.IDENTITY,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.EXTRACT_LARGEST,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Size reduction={fused.size_change_ratio:.2f}, fragmentation={fused.mean_component_fragmentation:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.OBJECT_EXTRACT_LARGEST,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="No clear object extraction pattern",
        )
    
    @staticmethod
    def _detect_crop_to_bbox(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect cropping to bounding box."""
        # Size reduction + high shape clarity
        if (fused.size_change_ratio < 0.7 and
            fused.mean_shape_clarity > 0.6):
            
            confidence = 0.75 * (1.0 - fused.size_change_ratio)
            return RecognizedTransform(
                transform_class=TransformClass.GRID_CROP_TO_BBOX,
                confidence=max(0.0, min(1.0, confidence)),
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.IDENTITY,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.EXTRACT_LARGEST,  # Approximation
                    mask=MaskMode.FOREGROUND_ONLY,
                ),
                reasoning=f"Size ratio={fused.size_change_ratio:.2f}, shape_clarity={fused.mean_shape_clarity:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.GRID_CROP_TO_BBOX,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="No cropping pattern detected",
        )
    
    @staticmethod
    def _detect_grid_expand(fused: FusedResonanceState) -> RecognizedTransform:
        """Detect grid expansion (tiling, repetition)."""
        # Size increase + high repetition
        if (fused.size_change_ratio > 1.3 and
            fused.mean_repetition_strength > 0.5):
            
            confidence = min(0.8, (fused.size_change_ratio - 1.0) * 0.5 + fused.mean_repetition_strength * 0.3)
            return RecognizedTransform(
                transform_class=TransformClass.GRID_EXPAND,
                confidence=max(0.0, min(1.0, confidence)),
                rule_vector=ARCRuleVector(
                    spatial=SpatialTransform.IDENTITY,
                    color=ColorTransform.IDENTITY,
                    obj=ObjectTransform.IDENTITY,
                    mask=MaskMode.FULL,
                ),
                reasoning=f"Size ratio={fused.size_change_ratio:.2f}, repetition={fused.mean_repetition_strength:.2f}",
            )
        return RecognizedTransform(
            transform_class=TransformClass.GRID_EXPAND,
            confidence=0.0,
            rule_vector=ARCRuleVector(),
            reasoning="No expansion pattern detected",
        )
