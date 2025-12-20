# cod_616/arc_organ/arc_two_attempt_sampler.py

"""
ARC Two-Attempt Sampler - Generate primary and fallback predictions.

Phase E: Uses metadata-aware rule detection (Recognition Field V2).

ARC Prize requires 2 attempts per test case. This generates:
- Attempt 1: Highest confidence rule hypothesis
- Attempt 2: Next-best alternative or intelligent fallback

Strategy maximizes chance of hitting correct answer.
"""

from __future__ import annotations

from typing import List
import numpy as np

from .arc_grid_parser import ArcGridPair
from .arc_example_fuser import ARCExampleFuser
from .arc_recognition_field_v2 import ARCRecognitionFieldV2
from .arc_rule_applicator import RuleApplicator


class ARCTwoAttemptSampler:
    """
    Phase E: Metadata-driven two-attempt generation.
    
    Uses Recognition Field V2 to detect rule families, then applies them.
    """
    
    def __init__(self):
        self.recognition_field = ARCRecognitionFieldV2()
        self.applicator = RuleApplicator()
    
    def generate_attempts(
        self,
        train_examples: List[ArcGridPair],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate two attempts for a test case.
        
        Args:
            train_examples: Training examples to learn from
            test_input: Test input grid to transform
        
        Returns:
            (attempt_1, attempt_2) tuple of output grids
        """
        # Phase D: Fuse training examples → task signature
        fused = ARCExampleFuser.fuse_examples(train_examples)
        
        # Phase E: Detect rule hypotheses (metadata-aware)
        top_hypotheses = self.recognition_field.recognize_top_n(
            train_examples, fused, n=2
        )
        
        # Apply top hypothesis → two attempts
        attempt_1, attempt_2 = self.applicator.apply(
            top_hypotheses[0], test_input
        )
        
        # If second hypothesis has high confidence, use it for attempt 2
        if len(top_hypotheses) > 1 and top_hypotheses[1].confidence > 0.5:
            alt_1, alt_2 = self.applicator.apply(
                top_hypotheses[1], test_input
            )
            # Replace attempt_2 with alternative hypothesis
            attempt_2 = alt_1
        
        return attempt_1, attempt_2
        
        # Default: return input (identity)
        return input_grid.copy()
