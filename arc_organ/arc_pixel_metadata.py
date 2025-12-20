# cod_616/arc_organ/arc_pixel_metadata.py

"""
ARC Pixel Metadata Engine - Phase A

Transforms raw integer grids into structured entity representations.
Each pixel becomes a rich metadata object with:
- Color and position
- Component/object membership
- Border/interior classification
- Symmetry group membership
- Tile/repetition cluster
- Neighborhood signature
- Transformation role

This is the foundation of Phase 2 architecture.
Without metadata, the system cannot reason about objects, symmetries, or transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class PixelMetadata:
    """
    Complete metadata for a single pixel.
    
    This is what Phase 2 sees instead of just an integer color.
    """
    # Identity
    position: Tuple[int, int]  # (y, x) coordinates
    color: int  # Raw value (0-9)
    normalized_color: float  # 0.0-1.0 range
    
    # Object membership
    component_id: int  # Connected component label (0=background)
    component_size: int  # Size of component this pixel belongs to
    component_rank: int  # Rank by size (0=largest, 1=second, etc.)
    
    # Spatial role
    is_background: bool  # color == 0
    is_border: bool  # On edge of component
    is_interior: bool  # Fully surrounded by same component
    
    # Structural membership
    symmetry_group: int  # Which symmetry group (0=none, 1=vertical, 2=horizontal, 3=both)
    tile_group: int  # Which tile block (for periodic patterns)
    
    # Local context
    adjacency_signature: Tuple[int, int, int, int]  # (N, S, E, W) neighbor colors
    neighbor_diversity: int  # Number of unique neighbor colors


@dataclass
class GridMetadata:
    """
    Complete metadata for entire grid.
    
    Contains per-pixel metadata plus global statistics.
    """
    shape: Tuple[int, int]
    pixel_metadata: np.ndarray  # Array of PixelMetadata objects
    
    # Global structure
    component_map: np.ndarray  # (H, W) array of component IDs
    num_components: int
    component_sizes: List[int]  # Size of each component
    
    # Symmetry analysis
    has_vertical_symmetry: bool
    has_horizontal_symmetry: bool
    has_rotational_symmetry: bool
    vertical_symmetry_score: float  # 0.0-1.0
    horizontal_symmetry_score: float
    rotational_symmetry_score: float
    
    # Tiling analysis
    is_tiled: bool
    tile_size: Optional[Tuple[int, int]]  # (height, width) of tile unit
    tile_repetitions: Optional[Tuple[int, int]]  # (rows, cols) of tiles
    tiling_strength: float  # 0.0-1.0


# ============================================================================
# COMPONENT EXTRACTION
# ============================================================================

def extract_components(grid: np.ndarray, background_value: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Extract connected components using 4-connectivity.
    
    Args:
        grid: Integer grid (H, W)
        background_value: Value to treat as background (default 0)
    
    Returns:
        component_map: (H, W) array with component IDs
        component_info: Dict with statistics
    """
    H, W = grid.shape
    
    # Create binary masks for each non-background color
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[unique_colors != background_value]
    
    component_map = np.zeros_like(grid, dtype=np.int32)
    component_id = 1
    component_sizes = []
    component_colors = []
    
    for color in unique_colors:
        # Binary mask for this color
        mask = (grid == color).astype(np.int32)
        
        # Label connected components
        labeled, num_features = ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        
        # Assign global component IDs
        for local_id in range(1, num_features + 1):
            component_mask = (labeled == local_id)
            component_map[component_mask] = component_id
            
            size = np.sum(component_mask)
            component_sizes.append(size)
            component_colors.append(color)
            
            component_id += 1
    
    return component_map, {
        'num_components': component_id - 1,
        'sizes': component_sizes,
        'colors': component_colors
    }


def compute_component_ranks(component_map: np.ndarray, component_sizes: List[int]) -> np.ndarray:
    """
    Assign rank to each pixel based on its component size.
    
    Returns:
        rank_map: (H, W) array where each pixel has rank of its component
                 (0=largest, 1=second largest, etc.)
    """
    H, W = component_map.shape
    rank_map = np.zeros_like(component_map)
    
    # Sort components by size (descending)
    sorted_indices = np.argsort(component_sizes)[::-1]
    
    # Assign ranks
    for rank, comp_idx in enumerate(sorted_indices):
        component_id = comp_idx + 1  # component IDs are 1-indexed
        mask = (component_map == component_id)
        rank_map[mask] = rank
    
    return rank_map


# ============================================================================
# BORDER/INTERIOR DETECTION
# ============================================================================

def compute_border_map(grid: np.ndarray, component_map: np.ndarray) -> np.ndarray:
    """
    Detect border pixels (on edge of components).
    
    Returns:
        border_map: (H, W) boolean array, True if pixel is on border
    """
    H, W = grid.shape
    border_map = np.zeros((H, W), dtype=bool)
    
    # Check 4-connected neighbors
    for y in range(H):
        for x in range(W):
            if component_map[y, x] == 0:  # Background
                continue
            
            current_comp = component_map[y, x]
            
            # Check neighbors
            neighbors = []
            if y > 0: neighbors.append(component_map[y-1, x])
            if y < H-1: neighbors.append(component_map[y+1, x])
            if x > 0: neighbors.append(component_map[y, x-1])
            if x < W-1: neighbors.append(component_map[y, x+1])
            
            # Border if any neighbor has different component
            if any(n != current_comp for n in neighbors):
                border_map[y, x] = True
    
    return border_map


def compute_interior_map(border_map: np.ndarray, component_map: np.ndarray) -> np.ndarray:
    """
    Detect interior pixels (fully surrounded by same component).
    
    Returns:
        interior_map: (H, W) boolean array, True if pixel is interior
    """
    # Interior = not border and not background
    interior_map = (~border_map) & (component_map > 0)
    return interior_map


# ============================================================================
# SYMMETRY DETECTION
# ============================================================================

def detect_symmetry(grid: np.ndarray) -> Dict:
    """
    Detect vertical, horizontal, and rotational symmetry.
    
    Returns:
        Dict with symmetry scores (0.0-1.0) and flags
    """
    H, W = grid.shape
    
    # Vertical symmetry (left-right mirror)
    vertical_score = 0.0
    if W > 1:
        left_half = grid[:, :W//2]
        right_half = grid[:, W-1:W//2-1:-1] if W % 2 == 0 else grid[:, W-1:W//2:-1]
        
        # Handle odd width
        compare_width = min(left_half.shape[1], right_half.shape[1])
        matches = np.sum(left_half[:, :compare_width] == right_half[:, :compare_width])
        total = H * compare_width
        vertical_score = matches / total if total > 0 else 0.0
    
    # Horizontal symmetry (top-bottom mirror)
    horizontal_score = 0.0
    if H > 1:
        top_half = grid[:H//2, :]
        bottom_half = grid[H-1:H//2-1:-1, :] if H % 2 == 0 else grid[H-1:H//2:-1, :]
        
        compare_height = min(top_half.shape[0], bottom_half.shape[0])
        matches = np.sum(top_half[:compare_height, :] == bottom_half[:compare_height, :])
        total = compare_height * W
        horizontal_score = matches / total if total > 0 else 0.0
    
    # Rotational symmetry (180 degrees)
    rotational_score = 0.0
    rotated = np.rot90(grid, k=2)
    matches = np.sum(grid == rotated)
    total = H * W
    rotational_score = matches / total if total > 0 else 0.0
    
    return {
        'vertical_score': vertical_score,
        'horizontal_score': horizontal_score,
        'rotational_score': rotational_score,
        'has_vertical': vertical_score > 0.8,
        'has_horizontal': horizontal_score > 0.8,
        'has_rotational': rotational_score > 0.8
    }


def compute_symmetry_groups(grid: np.ndarray, symmetry_info: Dict) -> np.ndarray:
    """
    Assign each pixel to a symmetry group.
    
    Returns:
        symmetry_map: (H, W) array with group assignments
                     0 = no symmetry
                     1 = vertical symmetry
                     2 = horizontal symmetry
                     3 = both (rotational)
    """
    H, W = grid.shape
    symmetry_map = np.zeros((H, W), dtype=np.int32)
    
    if symmetry_info['has_vertical']:
        # Pixels in left half get group 1
        symmetry_map[:, :W//2] = 1
        symmetry_map[:, W//2+1:] = 1
    
    if symmetry_info['has_horizontal']:
        # Pixels in top half get group 2 (or 3 if already 1)
        for y in range(H//2):
            for x in range(W):
                if symmetry_map[y, x] == 1:
                    symmetry_map[y, x] = 3
                else:
                    symmetry_map[y, x] = 2
        
        for y in range(H//2+1, H):
            for x in range(W):
                if symmetry_map[y, x] == 1:
                    symmetry_map[y, x] = 3
                else:
                    symmetry_map[y, x] = 2
    
    return symmetry_map


# ============================================================================
# TILING DETECTION
# ============================================================================

def detect_tiling(grid: np.ndarray, max_tile_size: int = 10) -> Dict:
    """
    Detect if grid is a perfect tiling of a smaller pattern.
    
    Args:
        grid: Input grid
        max_tile_size: Maximum tile dimension to test
    
    Returns:
        Dict with tiling info
    """
    H, W = grid.shape
    
    # Try different tile sizes
    for tile_h in range(1, min(H, max_tile_size) + 1):
        for tile_w in range(1, min(W, max_tile_size) + 1):
            # Must divide evenly
            if H % tile_h != 0 or W % tile_w != 0:
                continue
            
            # Extract base tile
            base_tile = grid[:tile_h, :tile_w]
            
            # Check if entire grid is tiling of base
            is_tiling = True
            for y in range(0, H, tile_h):
                for x in range(0, W, tile_w):
                    tile = grid[y:y+tile_h, x:x+tile_w]
                    if not np.array_equal(tile, base_tile):
                        is_tiling = False
                        break
                if not is_tiling:
                    break
            
            if is_tiling:
                return {
                    'is_tiled': True,
                    'tile_size': (tile_h, tile_w),
                    'tile_repetitions': (H // tile_h, W // tile_w),
                    'base_tile': base_tile,
                    'tiling_strength': 1.0
                }
    
    return {
        'is_tiled': False,
        'tile_size': None,
        'tile_repetitions': None,
        'base_tile': None,
        'tiling_strength': 0.0
    }


def compute_tile_groups(grid: np.ndarray, tiling_info: Dict) -> np.ndarray:
    """
    Assign each pixel to its tile block.
    
    Returns:
        tile_map: (H, W) array with tile block IDs
    """
    H, W = grid.shape
    tile_map = np.zeros((H, W), dtype=np.int32)
    
    if not tiling_info['is_tiled']:
        return tile_map
    
    tile_h, tile_w = tiling_info['tile_size']
    tile_id = 0
    
    for y in range(0, H, tile_h):
        for x in range(0, W, tile_w):
            tile_map[y:y+tile_h, x:x+tile_w] = tile_id
            tile_id += 1
    
    return tile_map


# ============================================================================
# ADJACENCY ANALYSIS
# ============================================================================

def compute_adjacency_signatures(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute neighborhood signature for each pixel.
    
    Returns:
        adjacency_signatures: (H, W, 4) array with (N, S, E, W) neighbor colors
        neighbor_diversity: (H, W) array with count of unique neighbors
    """
    H, W = grid.shape
    adjacency_signatures = np.zeros((H, W, 4), dtype=np.int32)
    neighbor_diversity = np.zeros((H, W), dtype=np.int32)
    
    for y in range(H):
        for x in range(W):
            neighbors = []
            
            # North
            if y > 0:
                neighbors.append(grid[y-1, x])
                adjacency_signatures[y, x, 0] = grid[y-1, x]
            
            # South
            if y < H-1:
                neighbors.append(grid[y+1, x])
                adjacency_signatures[y, x, 1] = grid[y+1, x]
            
            # East
            if x < W-1:
                neighbors.append(grid[y, x+1])
                adjacency_signatures[y, x, 2] = grid[y, x+1]
            
            # West
            if x > 0:
                neighbors.append(grid[y, x-1])
                adjacency_signatures[y, x, 3] = grid[y, x-1]
            
            neighbor_diversity[y, x] = len(set(neighbors))
    
    return adjacency_signatures, neighbor_diversity


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_grid_metadata(grid: np.ndarray) -> GridMetadata:
    """
    Main entry point: Extract complete metadata for grid.
    
    This is the Phase A core function that transforms raw grids
    into structured entity representations.
    
    Args:
        grid: Raw integer grid (H, W)
    
    Returns:
        GridMetadata with complete pixel-level and global metadata
    """
    H, W = grid.shape
    
    # Component extraction
    component_map, component_info = extract_components(grid)
    rank_map = compute_component_ranks(component_map, component_info['sizes'])
    
    # Border/interior detection
    border_map = compute_border_map(grid, component_map)
    interior_map = compute_interior_map(border_map, component_map)
    
    # Symmetry detection
    symmetry_info = detect_symmetry(grid)
    symmetry_groups = compute_symmetry_groups(grid, symmetry_info)
    
    # Tiling detection
    tiling_info = detect_tiling(grid)
    tile_groups = compute_tile_groups(grid, tiling_info)
    
    # Adjacency analysis
    adjacency_sigs, neighbor_div = compute_adjacency_signatures(grid)
    
    # Build per-pixel metadata
    pixel_metadata = np.empty((H, W), dtype=object)
    
    for y in range(H):
        for x in range(W):
            comp_id = component_map[y, x]
            comp_size = component_info['sizes'][comp_id - 1] if comp_id > 0 else 0
            comp_rank = rank_map[y, x]
            
            pixel_metadata[y, x] = PixelMetadata(
                position=(y, x),
                color=int(grid[y, x]),
                normalized_color=float(grid[y, x]) / 9.0,
                component_id=int(comp_id),
                component_size=int(comp_size),
                component_rank=int(comp_rank),
                is_background=(grid[y, x] == 0),
                is_border=bool(border_map[y, x]),
                is_interior=bool(interior_map[y, x]),
                symmetry_group=int(symmetry_groups[y, x]),
                tile_group=int(tile_groups[y, x]),
                adjacency_signature=tuple(adjacency_sigs[y, x]),
                neighbor_diversity=int(neighbor_div[y, x])
            )
    
    return GridMetadata(
        shape=(H, W),
        pixel_metadata=pixel_metadata,
        component_map=component_map,
        num_components=component_info['num_components'],
        component_sizes=component_info['sizes'],
        has_vertical_symmetry=symmetry_info['has_vertical'],
        has_horizontal_symmetry=symmetry_info['has_horizontal'],
        has_rotational_symmetry=symmetry_info['has_rotational'],
        vertical_symmetry_score=symmetry_info['vertical_score'],
        horizontal_symmetry_score=symmetry_info['horizontal_score'],
        rotational_symmetry_score=symmetry_info['rotational_score'],
        is_tiled=tiling_info['is_tiled'],
        tile_size=tiling_info['tile_size'],
        tile_repetitions=tiling_info['tile_repetitions'],
        tiling_strength=tiling_info['tiling_strength']
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_metadata_summary(metadata: GridMetadata) -> None:
    """Print human-readable summary of grid metadata."""
    print(f"Grid Shape: {metadata.shape}")
    print(f"Components: {metadata.num_components}")
    print(f"  Sizes: {metadata.component_sizes}")
    print(f"\nSymmetry:")
    print(f"  Vertical: {metadata.has_vertical_symmetry} (score: {metadata.vertical_symmetry_score:.3f})")
    print(f"  Horizontal: {metadata.has_horizontal_symmetry} (score: {metadata.horizontal_symmetry_score:.3f})")
    print(f"  Rotational: {metadata.has_rotational_symmetry} (score: {metadata.rotational_symmetry_score:.3f})")
    print(f"\nTiling:")
    print(f"  Is Tiled: {metadata.is_tiled}")
    if metadata.is_tiled:
        print(f"  Tile Size: {metadata.tile_size}")
        print(f"  Repetitions: {metadata.tile_repetitions}")
        print(f"  Strength: {metadata.tiling_strength:.3f}")


def get_pixel_metadata(metadata: GridMetadata, y: int, x: int) -> PixelMetadata:
    """Get metadata for specific pixel."""
    return metadata.pixel_metadata[y, x]
