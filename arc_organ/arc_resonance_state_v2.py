# cod_616/arc_organ/arc_resonance_state_v2.py

"""
Phase C: ARC Resonance State - 20-Dimensional Grid Signature

Transforms 6 metadata-enhanced channels → 20 semantic dimensions.
Maps structural properties to bounded feature space for Recognition Field.

Design Philosophy:
- D0-3:   Global structure & mass (object count, size distribution)
- D4-7:   Dominance & contrast (largest object, color diversity, borders)
- D8-11:  Symmetry & repetition (V/H/R symmetry, tiling patterns)
- D12-15: Layout & alignment (spatial arrangement, regularity)
- D16-19: Transformation priors (change localization, scale hints)

All dimensions normalized [0,1] or [-1,1] for stable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist

from .arc_grid_parser import ArcChannelBundle
from .arc_pixel_metadata import extract_grid_metadata, GridMetadata


@dataclass
class ARCResonanceState:
    """
    20-dimensional resonance state for ARC grid understanding.
    
    Parallel to ScreenResonanceState but specialized for abstract reasoning:
    - Metadata-driven (uses Phase A pixel entity attributes)
    - Transformation-aware (symmetry, tiling, structure)
    - Bounded features (all dims in [0,1] for clean thresholds)
    """
    
    # Group 1: Global structure & mass (D0-D3)
    foreground_mass_ratio: float      # D0: Fraction of non-background cells
    object_count_normalized: float    # D1: Number of components / K_max
    component_size_variance: float    # D2: Unevenness of object sizes
    fill_compactness: float           # D3: Clustered vs scattered objects
    
    # Group 2: Dominance & contrast (D4-D7)
    largest_component_dominance: float  # D4: Max object size / total foreground
    color_entropy_normalized: float     # D5: Diversity of colors (0=uniform, 1=mixed)
    border_fraction: float              # D6: Edge pixels / foreground pixels
    background_dominance: float         # D7: 1 - foreground_mass_ratio
    
    # Group 3: Symmetry & repetition (D8-D11)
    vertical_symmetry_strength: float   # D8: From Phase A symmetry score
    horizontal_symmetry_strength: float # D9: From Phase A symmetry score
    rotational_symmetry_strength: float # D10: From Phase A symmetry score
    tiling_strength: float              # D11: From Phase A tiling analysis
    
    # Group 4: Layout & alignment (D12-D15)
    row_col_alignment_score: float      # D12: Objects in straight rows/columns
    stripe_pattern_strength: float      # D13: Horizontal/vertical stripe regularity
    object_spacing_regularity: float    # D14: Variance in inter-object distances
    aspect_ratio_trend: float           # D15: Average bounding box aspect ratio
    
    # Group 5: Transformation priors (D16-D19)
    change_localization: float          # D16: Changes concentrated vs scattered
    color_change_concentration: float   # D17: Color changes to single target
    scale_change_indicator: float       # D18: Mass ratio input→output (training pairs)
    object_count_delta: float           # D19: Component count change (training pairs)
    
    def as_dict(self) -> Dict[str, float]:
        """Export as dictionary for serialization/logging."""
        return asdict(self)
    
    def as_array(self) -> np.ndarray:
        """Export as numpy array for numerical operations."""
        return np.array([
            self.foreground_mass_ratio,
            self.object_count_normalized,
            self.component_size_variance,
            self.fill_compactness,
            self.largest_component_dominance,
            self.color_entropy_normalized,
            self.border_fraction,
            self.background_dominance,
            self.vertical_symmetry_strength,
            self.horizontal_symmetry_strength,
            self.rotational_symmetry_strength,
            self.tiling_strength,
            self.row_col_alignment_score,
            self.stripe_pattern_strength,
            self.object_spacing_regularity,
            self.aspect_ratio_trend,
            self.change_localization,
            self.color_change_concentration,
            self.scale_change_indicator,
            self.object_count_delta,
        ], dtype=np.float32)
    
    @staticmethod
    def from_channels(
        channels: ArcChannelBundle,
        input_grid: np.ndarray,
        output_grid: Optional[np.ndarray] = None,
    ) -> "ARCResonanceState":
        """
        Compute 20-dimensional resonance from Phase A channels + metadata.
        
        Args:
            channels: 6-channel bundle from compute_arc_channels_v2()
            input_grid: Raw input grid (for metadata extraction)
            output_grid: Raw output grid (optional, for training pairs)
        
        Returns:
            ARCResonanceState with all 20 dimensions computed
        """
        # Extract metadata (Phase A)
        metadata = extract_grid_metadata(input_grid)
        
        # If output provided, extract its metadata for comparison
        output_metadata = None
        if output_grid is not None:
            output_metadata = extract_grid_metadata(output_grid)
        
        # Compute all 20 dimensions
        return ARCResonanceState(
            # Group 1: Global structure & mass
            foreground_mass_ratio=_compute_foreground_mass(metadata),
            object_count_normalized=_compute_object_count_norm(metadata),
            component_size_variance=_compute_size_variance(metadata),
            fill_compactness=_compute_compactness(input_grid, metadata),
            
            # Group 2: Dominance & contrast
            largest_component_dominance=_compute_largest_dominance(metadata),
            color_entropy_normalized=_compute_color_entropy(input_grid, metadata),
            border_fraction=_compute_border_fraction(metadata),
            background_dominance=_compute_background_dominance(metadata),
            
            # Group 3: Symmetry & repetition
            vertical_symmetry_strength=metadata.vertical_symmetry_score,
            horizontal_symmetry_strength=metadata.horizontal_symmetry_score,
            rotational_symmetry_strength=metadata.rotational_symmetry_score,
            tiling_strength=metadata.tiling_strength,
            
            # Group 4: Layout & alignment
            row_col_alignment_score=_compute_alignment(metadata),
            stripe_pattern_strength=_compute_stripe_pattern(input_grid),
            object_spacing_regularity=_compute_spacing_regularity(metadata),
            aspect_ratio_trend=_compute_aspect_ratio(metadata),
            
            # Group 5: Transformation priors
            change_localization=_compute_change_localization(channels),
            color_change_concentration=_compute_color_change_concentration(
                input_grid, output_grid
            ),
            scale_change_indicator=_compute_scale_change(metadata, output_metadata),
            object_count_delta=_compute_object_count_delta(metadata, output_metadata),
        )


# ============================================================================
# GROUP 1: Global Structure & Mass (D0-D3)
# ============================================================================

def _compute_foreground_mass(metadata: GridMetadata) -> float:
    """D0: Fraction of non-background cells."""
    total_pixels = metadata.shape[0] * metadata.shape[1]
    if total_pixels == 0:
        return 0.0
    
    # Count non-background pixels
    foreground_count = sum(
        size for i, size in enumerate(metadata.component_sizes)
        if i > 0  # Skip background (component 0)
    )
    
    return min(1.0, foreground_count / total_pixels)


def _compute_object_count_norm(metadata: GridMetadata, k_max: int = 10) -> float:
    """D1: Number of components normalized by k_max."""
    # Exclude background component
    foreground_components = max(0, metadata.num_components - 1)
    return min(1.0, foreground_components / k_max)


def _compute_size_variance(metadata: GridMetadata) -> float:
    """D2: Variance in component sizes (normalized)."""
    if metadata.num_components <= 1:
        return 0.0
    
    # Exclude background
    sizes = [s for i, s in enumerate(metadata.component_sizes) if i > 0]
    if len(sizes) == 0:
        return 0.0
    
    # Normalized variance
    sizes_arr = np.array(sizes, dtype=np.float32)
    mean_size = np.mean(sizes_arr)
    if mean_size == 0:
        return 0.0
    
    variance = np.var(sizes_arr) / (mean_size ** 2)  # Coefficient of variation squared
    return min(1.0, variance)


def _compute_compactness(grid: np.ndarray, metadata: GridMetadata) -> float:
    """D3: Fill compactness - clustered vs scattered."""
    if metadata.num_components <= 1:
        return 1.0  # Single component or background only = maximally compact
    
    # Create foreground mask
    foreground_mask = metadata.component_map > 0
    foreground_area = float(np.sum(foreground_mask))
    
    if foreground_area == 0:
        return 1.0
    
    # Compute perimeter (border pixels)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(foreground_mask)
    perimeter = float(np.sum(foreground_mask & ~eroded))
    
    if perimeter == 0:
        return 1.0
    
    # Compactness metric: 4π * area / perimeter²
    # Perfect circle = 1.0, scattered = closer to 0
    compactness = (4.0 * np.pi * foreground_area) / (perimeter ** 2)
    return min(1.0, compactness)


# ============================================================================
# GROUP 2: Dominance & Contrast (D4-D7)
# ============================================================================

def _compute_largest_dominance(metadata: GridMetadata) -> float:
    """D4: Fraction of foreground occupied by largest component."""
    if metadata.num_components <= 1:
        return 1.0  # Single component dominates completely
    
    # Exclude background
    sizes = [s for i, s in enumerate(metadata.component_sizes) if i > 0]
    if len(sizes) == 0:
        return 0.0
    
    max_size = max(sizes)
    total_size = sum(sizes)
    
    return max_size / total_size if total_size > 0 else 0.0


def _compute_color_entropy(grid: np.ndarray, metadata: GridMetadata) -> float:
    """D5: Color diversity in foreground (normalized entropy)."""
    # Get foreground colors only
    foreground_mask = metadata.component_map > 0
    foreground_colors = grid[foreground_mask]
    
    if len(foreground_colors) == 0:
        return 0.0
    
    # Compute histogram
    unique_colors, counts = np.unique(foreground_colors, return_counts=True)
    probs = counts / counts.sum()
    
    # Entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Normalize by max possible entropy (log2 of unique colors)
    max_entropy = np.log2(len(unique_colors)) if len(unique_colors) > 1 else 1.0
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _compute_border_fraction(metadata: GridMetadata) -> float:
    """D6: Fraction of foreground pixels that are borders."""
    # Count border pixels from metadata
    border_count = 0
    interior_count = 0
    
    h, w = metadata.shape
    for i in range(h):
        for j in range(w):
            pixel_meta = metadata.pixel_metadata[i, j]
            if not pixel_meta.is_background:
                if pixel_meta.is_border:
                    border_count += 1
                elif pixel_meta.is_interior:
                    interior_count += 1
    
    total_foreground = border_count + interior_count
    return border_count / total_foreground if total_foreground > 0 else 0.0


def _compute_background_dominance(metadata: GridMetadata) -> float:
    """D7: Inverse of foreground mass."""
    return 1.0 - _compute_foreground_mass(metadata)


# ============================================================================
# GROUP 3: Symmetry & Repetition (D8-D11)
# ============================================================================

# D8-D11: Directly from metadata (already computed in Phase A)
# - metadata.vertical_symmetry_score
# - metadata.horizontal_symmetry_score
# - metadata.rotational_symmetry_score
# - metadata.tiling_strength


# ============================================================================
# GROUP 4: Layout & Alignment (D12-D15)
# ============================================================================

def _compute_alignment(metadata: GridMetadata) -> float:
    """D12: Objects aligned in rows/columns."""
    if metadata.num_components <= 2:  # Background + 1 object
        return 1.0
    
    # Get component centroids
    centroids = []
    for comp_id in range(1, metadata.num_components):  # Skip background
        mask = metadata.component_map == comp_id
        if np.sum(mask) == 0:
            continue
        
        rows, cols = np.where(mask)
        centroid_r = np.mean(rows)
        centroid_c = np.mean(cols)
        centroids.append((centroid_r, centroid_c))
    
    if len(centroids) < 2:
        return 1.0
    
    # Check row alignment: variance of row positions
    rows = [c[0] for c in centroids]
    cols = [c[1] for c in centroids]
    
    row_var = np.var(rows) if len(rows) > 1 else 0.0
    col_var = np.var(cols) if len(cols) > 1 else 0.0
    
    # Low variance = high alignment
    # Normalize by grid size
    h, w = metadata.shape
    row_var_norm = row_var / (h ** 2) if h > 0 else 0.0
    col_var_norm = col_var / (w ** 2) if w > 0 else 0.0
    
    # High score when either rows or columns are aligned
    alignment = 1.0 - min(row_var_norm, col_var_norm)
    return max(0.0, min(1.0, alignment))


def _compute_stripe_pattern(grid: np.ndarray) -> float:
    """D13: Horizontal/vertical stripe regularity."""
    h, w = grid.shape
    if h < 2 or w < 2:
        return 0.0
    
    # Check row similarity (horizontal stripes)
    row_similarities = []
    for i in range(h - 1):
        similarity = np.mean(grid[i] == grid[i + 1])
        row_similarities.append(similarity)
    
    # Check column similarity (vertical stripes)
    col_similarities = []
    for j in range(w - 1):
        similarity = np.mean(grid[:, j] == grid[:, j + 1])
        col_similarities.append(similarity)
    
    # Max of row/col patterns
    row_pattern = np.mean(row_similarities) if row_similarities else 0.0
    col_pattern = np.mean(col_similarities) if col_similarities else 0.0
    
    return max(row_pattern, col_pattern)


def _compute_spacing_regularity(metadata: GridMetadata) -> float:
    """D14: Regularity of inter-object spacing."""
    if metadata.num_components <= 2:  # Background + 1 object
        return 1.0
    
    # Get component centroids
    centroids = []
    for comp_id in range(1, metadata.num_components):
        mask = metadata.component_map == comp_id
        if np.sum(mask) == 0:
            continue
        
        rows, cols = np.where(mask)
        centroid_r = np.mean(rows)
        centroid_c = np.mean(cols)
        centroids.append([centroid_r, centroid_c])
    
    if len(centroids) < 2:
        return 1.0
    
    # Compute pairwise distances
    centroids_arr = np.array(centroids)
    distances = cdist(centroids_arr, centroids_arr, metric='euclidean')
    
    # Extract non-zero distances (exclude diagonal)
    non_zero_dists = distances[np.triu_indices_from(distances, k=1)]
    
    if len(non_zero_dists) == 0:
        return 1.0
    
    # Regularity = 1 / (coefficient of variation)
    mean_dist = np.mean(non_zero_dists)
    std_dist = np.std(non_zero_dists)
    
    if mean_dist == 0:
        return 1.0
    
    coeff_var = std_dist / mean_dist
    regularity = 1.0 / (1.0 + coeff_var)  # High regularity when CV is low
    
    return regularity


def _compute_aspect_ratio(metadata: GridMetadata) -> float:
    """D15: Average bounding box aspect ratio."""
    if metadata.num_components <= 1:
        return 0.5  # Neutral
    
    aspect_ratios = []
    for comp_id in range(1, metadata.num_components):
        mask = metadata.component_map == comp_id
        if np.sum(mask) == 0:
            continue
        
        rows, cols = np.where(mask)
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        
        if width > 0:
            aspect = height / width
            # Normalize to [0,1]: tall=1, square=0.5, wide=0
            normalized_aspect = aspect / (1.0 + aspect)
            aspect_ratios.append(normalized_aspect)
    
    return np.mean(aspect_ratios) if aspect_ratios else 0.5


# ============================================================================
# GROUP 5: Transformation Priors (D16-D19)
# ============================================================================

def _compute_change_localization(channels: ArcChannelBundle) -> float:
    """D16: Changes concentrated vs scattered."""
    delta = channels.delta_map
    
    if delta is None or delta.size == 0:
        return 0.0
    
    # Threshold delta to get change mask
    change_mask = delta > 0.1
    change_mass = float(np.sum(change_mask))
    
    if change_mass == 0:
        return 0.0  # No changes
    
    # Compute connected components of changes
    from scipy.ndimage import label
    labeled_changes, num_change_regions = label(change_mask)
    
    if num_change_regions == 0:
        return 0.0
    
    # Localization = fewer regions = higher score
    # Normalize: 1 region = 1.0, many regions = 0.0
    localization = 1.0 / num_change_regions
    return min(1.0, localization * 5.0)  # Scale so 5+ regions → 0


def _compute_color_change_concentration(
    input_grid: np.ndarray,
    output_grid: Optional[np.ndarray],
) -> float:
    """D17: Color changes focused on single target color."""
    if output_grid is None:
        return 0.0  # No output to compare
    
    if input_grid.shape != output_grid.shape:
        return 0.0  # Size mismatch
    
    # Find pixels that changed
    changed_mask = input_grid != output_grid
    changed_count = np.sum(changed_mask)
    
    if changed_count == 0:
        return 0.0
    
    # Get target colors for changed pixels
    target_colors = output_grid[changed_mask]
    
    # Compute entropy of target colors
    unique_targets, counts = np.unique(target_colors, return_counts=True)
    
    if len(unique_targets) == 1:
        return 1.0  # All changes to single color
    
    # Concentration = inverse of entropy
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(len(unique_targets))
    
    concentration = 1.0 - (entropy / max_entropy)
    return concentration


def _compute_scale_change(
    input_metadata: GridMetadata,
    output_metadata: Optional[GridMetadata],
) -> float:
    """D18: Scale change indicator (mass ratio)."""
    if output_metadata is None:
        return 0.5  # Neutral (no output)
    
    input_mass = _compute_foreground_mass(input_metadata)
    output_mass = _compute_foreground_mass(output_metadata)
    
    if input_mass == 0:
        return 0.5
    
    # Ratio: >1 = expansion, <1 = reduction
    ratio = output_mass / input_mass
    
    # Normalize to [0,1]: 0=shrink, 0.5=same, 1=expand
    if ratio < 1.0:
        return 0.5 * ratio  # Shrink: [0, 0.5]
    else:
        return 0.5 + 0.5 * min(1.0, (ratio - 1.0))  # Expand: [0.5, 1]


def _compute_object_count_delta(
    input_metadata: GridMetadata,
    output_metadata: Optional[GridMetadata],
) -> float:
    """D19: Change in component count."""
    if output_metadata is None:
        return 0.5  # Neutral (no output)
    
    input_count = max(0, input_metadata.num_components - 1)  # Exclude background
    output_count = max(0, output_metadata.num_components - 1)
    
    delta = output_count - input_count
    
    # Normalize to [0,1]: 0=remove, 0.5=same, 1=add
    if delta < 0:
        return 0.5 * (1.0 + delta / max(1, input_count))  # Remove: [0, 0.5]
    elif delta == 0:
        return 0.5
    else:
        return 0.5 + 0.5 * min(1.0, delta / 10.0)  # Add: [0.5, 1]
