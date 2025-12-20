"""
Enclosed Region Fill Operator

Mathematical DSL Implementation:
- Detect boundary components
- Find enclosed interior regions via topological flood fill
- Apply conditional fill based on size threshold
"""
import numpy as np
from scipy import ndimage
from typing import Optional, Dict, Any

class EnclosedRegionFillOperator:
    """
    Fill enclosed regions within boundaries based on size threshold.
    
    Math DSL:
        objects:
          - Boundary B = {(r,c) | G_in(r,c) = boundary_color}
          - Interior I(C_i) = cells enclosed by boundary component C_i
          - Fill color k_fill
        
        conditions:
          - Grid size preserved
          - Interior filled if |I| ≥ min_size and min_dimension ≥ min_dim
        
        equations:
          - For each boundary component C_i:
              I(C_i) = {(r,c) | enclosed by C_i and G_in(r,c) = 0}
              If size_condition(I): ∀(r,c)∈I: G_out(r,c) = k_fill
        
        transform:
          - Flood fill from exterior to identify enclosed regions
          - Check size/dimension thresholds
          - Apply conditional fill
          - Preserve boundaries
    """
    
    name = "enclosed_region_fill"
    
    def analyze(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if task matches enclosed region fill pattern.
        
        Returns params: {
            'boundary_color': int,
            'fill_color': int,
            'min_size': int,
            'min_dim': int
        }
        """
        # Check basic conditions
        if inp.shape != out.shape:
            return None
        
        # Find boundary color (appears in both input and output, same positions)
        boundary_mask = (inp == out) & (inp > 0)
        if not boundary_mask.any():
            return None
        
        boundary_colors = set(inp[boundary_mask].flatten())
        if len(boundary_colors) != 1:
            return None
        
        boundary_color = list(boundary_colors)[0]
        
        # Find fill color (appears in output but not in same positions in input)
        filled_mask = (out > 0) & (inp == 0)
        if not filled_mask.any():
            return None
        
        fill_colors = set(out[filled_mask].flatten())
        if len(fill_colors) != 1:
            return None
        
        fill_color = list(fill_colors)[0]
        
        # Verify this is topological fill pattern
        # Check that filled regions are enclosed by boundaries
        boundary_map = (inp == boundary_color)
        
        # Find all regions that got filled
        filled_regions = self._find_enclosed_regions(inp, boundary_color)
        
        if len(filled_regions) == 0:
            return None
        
        # Determine size threshold from examples
        filled_sizes = []
        unfilled_sizes = []
        
        for region in filled_regions:
            region_mask = np.zeros_like(inp, dtype=bool)
            for r, c in region:
                region_mask[r, c] = True
            
            # Check if this region is filled in output
            is_filled = (out[region_mask] == fill_color).all()
            
            if is_filled:
                filled_sizes.append(len(region))
            else:
                unfilled_sizes.append(len(region))
        
        if not filled_sizes:
            return None
        
        # Threshold is minimum of filled sizes
        min_size = min(filled_sizes)
        
        # Check minimum dimension (2x2 typically)
        min_dim = 2
        
        return {
            'boundary_color': int(boundary_color),
            'fill_color': int(fill_color),
            'min_size': min_size,
            'min_dim': min_dim
        }
    
    def apply(self, inp: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply enclosed region fill transformation.
        """
        out = inp.copy()
        
        boundary_color = params['boundary_color']
        fill_color = params['fill_color']
        min_size = params['min_size']
        min_dim = params['min_dim']
        
        # Find all enclosed regions
        enclosed_regions = self._find_enclosed_regions(inp, boundary_color)
        
        # Fill regions that meet size criteria
        for region in enclosed_regions:
            if len(region) >= min_size:
                # Check minimum dimension
                rows = [r for r, c in region]
                cols = [c for r, c in region]
                
                height = max(rows) - min(rows) + 1
                width = max(cols) - min(cols) + 1
                
                if min(height, width) >= min_dim:
                    # Fill this region
                    for r, c in region:
                        out[r, c] = fill_color
        
        return out
    
    def _find_enclosed_regions(self, grid: np.ndarray, boundary_color: int) -> list:
        """
        Find all regions enclosed by boundaries using exterior flood fill.
        
        Algorithm:
        1. Flood fill from exterior (mark as exterior)
        2. Non-boundary, non-exterior cells = enclosed regions
        3. Group into connected components
        """
        H, W = grid.shape
        
        # Create boundary mask
        boundary = (grid == boundary_color)
        
        # Flood fill from all edges to mark exterior
        exterior = np.zeros((H, W), dtype=bool)
        
        # Start from all edge cells that aren't boundaries
        to_visit = []
        
        # Top and bottom edges
        for c in range(W):
            if not boundary[0, c]:
                to_visit.append((0, c))
            if not boundary[H-1, c]:
                to_visit.append((H-1, c))
        
        # Left and right edges
        for r in range(H):
            if not boundary[r, 0]:
                to_visit.append((r, 0))
            if not boundary[r, W-1]:
                to_visit.append((r, W-1))
        
        # BFS flood fill
        while to_visit:
            r, c = to_visit.pop(0)
            
            if exterior[r, c] or boundary[r, c]:
                continue
            
            exterior[r, c] = True
            
            # Add neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not exterior[nr, nc] and not boundary[nr, nc]:
                        to_visit.append((nr, nc))
        
        # Interior = not boundary and not exterior
        interior = ~boundary & ~exterior
        
        # Find connected components of interior
        labeled, num_components = ndimage.label(interior)
        
        regions = []
        for i in range(1, num_components + 1):
            region_mask = (labeled == i)
            region_coords = list(zip(*np.where(region_mask)))
            regions.append(region_coords)
        
        return regions


# Register operator
OPERATOR = EnclosedRegionFillOperator()
