# cod_616/arc_organ/arc_example_fuser.py

"""
ARC Example Fuser - Multi-example resonance aggregation.

Takes N training examples and fuses their resonance signatures into a
unified representation of "what transformation is being applied".

This is the multi-modal fusion logic adapted for abstract reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from .arc_grid_parser import ArcGridPair, compute_arc_channels_v2
from .arc_resonance_state_v2 import ARCResonanceState


@dataclass
class FusedResonanceState:
    """
    Fused resonance across multiple ARC training examples.
    Phase C: Uses 20-dimensional resonance from metadata-enhanced channels.
    
    Contains:
    - Averaged resonance features (what's consistent across examples)
    - Variance features (what varies between examples)
    - Transform signatures (tiling, symmetry, change patterns)
    """
    
    # Mean values across examples (20 features from Phase C)
    mean_foreground_mass_ratio: float
    mean_object_count_normalized: float
    mean_component_size_variance: float
    mean_fill_compactness: float
    mean_largest_component_dominance: float
    mean_color_entropy_normalized: float
    mean_border_fraction: float
    mean_background_dominance: float
    mean_vertical_symmetry_strength: float
    mean_horizontal_symmetry_strength: float
    mean_rotational_symmetry_strength: float
    mean_tiling_strength: float
    mean_row_col_alignment_score: float
    mean_stripe_pattern_strength: float
    mean_object_spacing_regularity: float
    mean_aspect_ratio_trend: float
    mean_change_localization: float
    mean_color_change_concentration: float
    mean_scale_change_indicator: float
    mean_object_count_delta: float
    
    # Variance across examples (transformation stability indicators)
    var_tiling: float                    # Low = consistent tiling pattern
    var_v_symmetry: float                # Low = consistent vertical symmetry
    var_h_symmetry: float                # Low = consistent horizontal symmetry
    var_change_localization: float       # Low = changes always in same spots
    
    # Transform-specific signatures
    spatial_consistency: float  # How similar spatial patterns are
    color_consistency: float    # How similar color distributions are
    size_change_ratio: float    # Average output/input size ratio
    num_examples: int
    
    # Phase 1 → Phase 2 compatibility properties
    @property
    def mean_delta_coverage(self) -> float:
        """Backward compat: map to change_localization."""
        return self.mean_change_localization
    
    @property
    def mean_symmetry_strength(self) -> float:
        """Backward compat: average of V/H symmetry."""
        return (self.mean_vertical_symmetry_strength + 
                self.mean_horizontal_symmetry_strength) / 2.0
    
    @property
    def mean_row_pattern_score(self) -> float:
        """Backward compat: map to stripe pattern."""
        return self.mean_stripe_pattern_strength
    
    @property
    def mean_col_pattern_score(self) -> float:
        """Backward compat: map to row/col alignment."""
        return self.mean_row_col_alignment_score
    
    @property
    def mean_repetition_strength(self) -> float:
        """Backward compat: map to tiling strength."""
        return self.mean_tiling_strength
    
    def as_dict(self) -> Dict[str, float]:
        """Export fused resonance as dictionary."""
        return {
            'mean_foreground_mass_ratio': self.mean_foreground_mass_ratio,
            'mean_object_count_normalized': self.mean_object_count_normalized,
            'mean_component_size_variance': self.mean_component_size_variance,
            'mean_fill_compactness': self.mean_fill_compactness,
            'mean_largest_component_dominance': self.mean_largest_component_dominance,
            'mean_color_entropy_normalized': self.mean_color_entropy_normalized,
            'mean_border_fraction': self.mean_border_fraction,
            'mean_background_dominance': self.mean_background_dominance,
            'mean_vertical_symmetry_strength': self.mean_vertical_symmetry_strength,
            'mean_horizontal_symmetry_strength': self.mean_horizontal_symmetry_strength,
            'mean_rotational_symmetry_strength': self.mean_rotational_symmetry_strength,
            'mean_tiling_strength': self.mean_tiling_strength,
            'mean_row_col_alignment_score': self.mean_row_col_alignment_score,
            'mean_stripe_pattern_strength': self.mean_stripe_pattern_strength,
            'mean_object_spacing_regularity': self.mean_object_spacing_regularity,
            'mean_aspect_ratio_trend': self.mean_aspect_ratio_trend,
            'mean_change_localization': self.mean_change_localization,
            'mean_color_change_concentration': self.mean_color_change_concentration,
            'mean_scale_change_indicator': self.mean_scale_change_indicator,
            'mean_object_count_delta': self.mean_object_count_delta,
            'var_tiling': self.var_tiling,
            'var_v_symmetry': self.var_v_symmetry,
            'var_h_symmetry': self.var_h_symmetry,
            'var_change_localization': self.var_change_localization,
            'spatial_consistency': self.spatial_consistency,
            'color_consistency': self.color_consistency,
            'size_change_ratio': self.size_change_ratio,
            'num_examples': float(self.num_examples),
        }


class ARCExampleFuser:
    """
    Fuses multiple ARC training examples into unified transformation signature.
    
    This is the Recognition Field's input - a compressed representation of
    "what rule is being applied" across all training examples.
    """
    
    @staticmethod
    def fuse_examples(examples: List[ArcGridPair]) -> FusedResonanceState:
        """
        Main fusion logic: N examples → 1 fused resonance state.
        
        Args:
            examples: List of training example pairs (input, output)
        
        Returns:
            FusedResonanceState containing aggregated signatures
        """
        if not examples:
            return ARCExampleFuser._empty_state()
        
        # Extract resonance from each example
        # Phase 2: Use metadata-enhanced channels and resonance
        resonances = []
        size_ratios = []
        
        for pair in examples:
            channels = compute_arc_channels_v2(pair.input_grid, pair.output_grid)
            res = ARCResonanceState.from_channels(
                channels, pair.input_grid, pair.output_grid
            )
            resonances.append(res)
            
            # Track size changes
            if pair.output_grid is not None:
                in_size = pair.input_grid.size
                out_size = pair.output_grid.size
                if in_size > 0:
                    size_ratios.append(out_size / in_size)
        
        # Aggregate mean values - Phase C: Use 20-dim resonance attributes
        mean_vals = {}
        for key in [
            'foreground_mass_ratio', 'object_count_normalized', 'component_size_variance',
            'fill_compactness', 'largest_component_dominance', 'color_entropy_normalized',
            'border_fraction', 'background_dominance',
            'vertical_symmetry_strength', 'horizontal_symmetry_strength',
            'rotational_symmetry_strength', 'tiling_strength',
            'row_col_alignment_score', 'stripe_pattern_strength',
            'object_spacing_regularity', 'aspect_ratio_trend',
            'change_localization', 'color_change_concentration',
            'scale_change_indicator', 'object_count_delta'
        ]:
            values = [getattr(r, key) for r in resonances]
            mean_vals[f'mean_{key}'] = float(np.mean(values))
        
        # Compute variances for key transformation indicators
        tilings = [r.tiling_strength for r in resonances]
        v_symmetries = [r.vertical_symmetry_strength for r in resonances]
        h_symmetries = [r.horizontal_symmetry_strength for r in resonances]
        changes = [r.change_localization for r in resonances]
        
        var_tiling = float(np.var(tilings))
        var_v_symmetry = float(np.var(v_symmetries))
        var_h_symmetry = float(np.var(h_symmetries))
        var_change_localization = float(np.var(changes))
        
        # Compute consistency metrics
        spatial_consistency = ARCExampleFuser._compute_spatial_consistency(resonances)
        color_consistency = ARCExampleFuser._compute_color_consistency(resonances)
        
        # Average size ratio
        size_change_ratio = float(np.mean(size_ratios)) if size_ratios else 1.0
        
        return FusedResonanceState(
            **mean_vals,
            var_tiling=var_tiling,
            var_v_symmetry=var_v_symmetry,
            var_h_symmetry=var_h_symmetry,
            var_change_localization=var_change_localization,
            spatial_consistency=spatial_consistency,
            color_consistency=color_consistency,
            size_change_ratio=size_change_ratio,
            num_examples=len(examples),
        )
    
    @staticmethod
    def _compute_spatial_consistency(resonances: List[ARCResonanceState]) -> float:
        """
        How consistent are spatial patterns across examples?
        High = same spatial structure preserved
        """
        if len(resonances) < 2:
            return 1.0
        
        # Compare spatial patterns using Phase C attributes
        spatial_features = []
        for r in resonances:
            feat = np.array([
                r.row_col_alignment_score,
                r.stripe_pattern_strength,
                r.vertical_symmetry_strength,
                r.horizontal_symmetry_strength,
                r.tiling_strength,
            ])
            spatial_features.append(feat)
        
        # Compute pairwise correlations
        features_matrix = np.array(spatial_features)
        if features_matrix.shape[0] < 2:
            return 1.0
        
        correlations = []
        for i in range(len(spatial_features)):
            for j in range(i + 1, len(spatial_features)):
                f1 = spatial_features[i]
                f2 = spatial_features[j]
                if np.std(f1) > 0 and np.std(f2) > 0:
                    corr = np.corrcoef(f1, f2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    @staticmethod
    def _compute_color_consistency(resonances: List[ARCResonanceState]) -> float:
        """
        How consistent are color patterns across examples?
        High = similar color distribution changes
        """
        if len(resonances) < 2:
            return 1.0
        
        # Compare color-related features using Phase C attributes
        color_features = []
        for r in resonances:
            feat = np.array([
                r.color_entropy_normalized,
                r.largest_component_dominance,
                r.foreground_mass_ratio,
                r.change_localization,
            ])
            color_features.append(feat)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(color_features)):
            for j in range(i + 1, len(color_features)):
                f1 = color_features[i]
                f2 = color_features[j]
                if np.std(f1) > 0 and np.std(f2) > 0:
                    corr = np.corrcoef(f1, f2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    @staticmethod
    def _empty_state() -> FusedResonanceState:
        """Return empty fused state for edge cases."""
        return FusedResonanceState(
            mean_energy_total=0.0,
            mean_energy_var=0.0,
            mean_sparsity=1.0,
            mean_color_entropy=0.0,
            mean_color_component_dominance=0.0,
            mean_symmetry_strength=0.0,
            mean_repetition_strength=0.0,
            mean_shape_mass=0.0,
            mean_component_fragmentation=0.0,
            mean_foreground_ratio=0.0,
            mean_row_pattern_score=0.0,
            mean_col_pattern_score=0.0,
            mean_block_repetition_score=0.0,
            mean_checkerboard_tendency=0.0,
            mean_shape_clarity=0.0,
            mean_boundary_complexity=0.0,
            mean_symmetry_axis_confidence=0.0,
            mean_component_size_variance=0.0,
            mean_delta_coverage=0.0,
            mean_delta_focus=0.0,
            var_delta_coverage=0.0,
            var_symmetry_strength=0.0,
            var_repetition_strength=0.0,
            var_shape_mass=0.0,
            spatial_consistency=0.0,
            color_consistency=0.0,
            size_change_ratio=1.0,
            num_examples=0,
        )
