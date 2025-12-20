# cod_616/arc_organ/arc_recognition_field_v2.py

"""
Phase E: Recognition Field v2 - Metadata-Aware Rule Detection

Maps task signatures + training examples → ranked rule hypotheses.
Uses Phase A/C/D metadata to make intelligent decisions.

This replaces the Phase 1 heuristic-based recognition with structured detection.
"""

from __future__ import annotations

from typing import List

from .arc_grid_parser import ArcGridPair
from .arc_example_fuser import FusedResonanceState
from .arc_rule_hypothesis import (
    RuleHypothesis,
    RuleFamily,
    TilingExpandDetector,
    SelfReferentialTilingDetector,
    SandwichTilingDetector,
    DiagonalMarkerTilingDetector,
    LatinSquareFromDiagonalDetector,
    HorizontalReplicateDetector,
    PureRecolorDetector,
    RecolorMappingDetector,
)


class ARCRecognitionFieldV2:
    """
    Phase E: Metadata-driven rule recognition.
    
    Takes task signature + training examples → ranked rule hypotheses.
    Each detector leverages Phase A/C/D features for intelligent classification.
    """
    
    def __init__(self):
        # Register detectors in priority order (most specific first)
        self.detectors = [
            DiagonalMarkerTilingDetector(),        # Very specific: 3×3 with diagonal markers (310f3251)
            LatinSquareFromDiagonalDetector(),     # Very specific: Latin square from diagonals (05269061)
            SandwichTilingDetector(),              # Very specific: 2×2→6×6 with flip (00576224)
            SelfReferentialTilingDetector(),       # Very specific: self-masking tiling (007bbfb7)
            HorizontalReplicateDetector(),         # Primitive: horizontal N× replication (a416b8f3)
            PureRecolorDetector(),                 # Primitive: pure A→B color mapping (b1948b0a, c8f0f002, d511f180)
            TilingExpandDetector(),                # High confidence, specific pattern
            RecolorMappingDetector(),              # Common, easy to verify
            # Future: MirrorFlipDetector(), ExtractLargestDetector(), etc.
        ]
    
    def recognize_task(
        self,
        train_examples: List[ArcGridPair],
        task_signature: FusedResonanceState,
    ) -> List[RuleHypothesis]:
        """
        Analyze task and return ranked rule hypotheses.
        
        Args:
            train_examples: Training pairs with input/output grids
            task_signature: Fused resonance from Phase D
        
        Returns:
            List of RuleHypothesis sorted by confidence (highest first)
        """
        hypotheses = []
        
        # Run all detectors
        for detector in self.detectors:
            hypothesis = detector.detect(train_examples, task_signature)
            if hypothesis is not None:
                hypotheses.append(hypothesis)
        
        # Add identity fallback (always available, low confidence)
        hypotheses.append(RuleHypothesis(
            family=RuleFamily.IDENTITY,
            confidence=0.1,
            params={},
            reasoning="Fallback: return input unchanged",
        ))
        
        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def recognize_top_n(
        self,
        train_examples: List[ArcGridPair],
        task_signature: FusedResonanceState,
        n: int = 2,
    ) -> List[RuleHypothesis]:
        """Get top N rule hypotheses for two-attempt sampling."""
        all_hypotheses = self.recognize_task(train_examples, task_signature)
        return all_hypotheses[:n]
