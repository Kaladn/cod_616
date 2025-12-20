"""
Pattern Completion from Prototype Operator

Mathematical DSL Implementation:
- Identify prototype (complete pattern)
- Find partial instances
- Complete partials to match prototype structure
"""
import numpy as np
from scipy import ndimage
from typing import Optional, Dict, Any, List, Tuple

class PatternCompletionPrototypeOperator:
    """
    Complete partial patterns based on prototype.
    
    Math DSL:
        objects:
          - Shape set S = {S_1, S_2, ..., S_n}
          - Prototype P = largest/most complete shape
          - Partial shapes {S_i | S_i ⊂ P structurally}
        
        equations:
          - Identify prototype: P = argmax_{S∈S} |S| or complexity(S)
          - For each partial S_i:
              Alignment: τ_i = argmin_τ distance(S_i, P + τ)
              Complete: S_i' = P + τ_i
        
        transform:
          - Extract all connected shapes
          - Identify prototype (largest/complete)
          - For each partial:
              Align to prototype
              Copy missing pixels
              Place completed shape
    """
    
    name = "pattern_completion_prototype"
    
    def analyze(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if task matches pattern completion pattern.
        
        Returns params: {
            'prototype': np.ndarray,  # the complete pattern
            'min_overlap': float,  # minimum overlap ratio to be considered partial
        }
        """
        # Check that shapes exist in both
        inp_nonzero = (inp > 0)
        out_nonzero = (out > 0)
        
        if not inp_nonzero.any() or not out_nonzero.any():
            return None
        
        # Extract shapes from input
        inp_shapes = self._extract_shapes(inp)
        out_shapes = self._extract_shapes(out)
        
        if len(inp_shapes) < 2:
            return None
        
        # Find prototype (largest or most complex)
        prototype = max(inp_shapes, key=lambda s: s['size'])
        
        # Check if output shapes are completions of input shapes
        # Output should have same number or more complete shapes
        if len(out_shapes) < len(inp_shapes):
            return None
        
        # Verify pattern: smaller shapes in input become larger in output
        expanded = False
        for inp_shape in inp_shapes:
            if inp_shape['size'] < prototype['size']:
                # This should be expanded in output
                expanded = True
                break
        
        if not expanded:
            return None
        
        return {
            'prototype': prototype['mask'],
            'min_overlap': 0.3  # 30% overlap required
        }
    
    def _extract_shapes(self, grid: np.ndarray) -> List[Dict]:
        """Extract connected components as shapes."""
        nonzero = (grid > 0).astype(int)
        labeled, num = ndimage.label(nonzero)
        
        shapes = []
        for i in range(1, num + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            
            # Bounding box
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            
            # Extract shape region
            shape_region = grid[r_min:r_max+1, c_min:c_max+1].copy()
            shape_mask = mask[r_min:r_max+1, c_min:c_max+1].copy()
            
            shapes.append({
                'coords': coords,
                'bbox': (r_min, c_min, r_max, c_max),
                'region': shape_region,
                'mask': shape_mask,
                'size': len(coords)
            })
        
        return shapes
    
    def apply(self, inp: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply pattern completion transformation.
        """
        prototype_mask = params['prototype']
        min_overlap = params['min_overlap']
        
        # Extract shapes from input
        inp_shapes = self._extract_shapes(inp)
        
        # Identify prototype from shapes
        prototype = max(inp_shapes, key=lambda s: s['size'])
        
        # Create output
        out = inp.copy()
        
        # For each partial shape, complete it
        for shape in inp_shapes:
            if shape['size'] < prototype['size'] * 0.9:  # Partial if <90% of prototype
                # Try to complete this shape
                completed = self._complete_shape(shape, prototype, inp)
                
                if completed is not None:
                    # Place completed shape in output
                    r_min, c_min = completed['coords'].min(axis=0)
                    for r, c in completed['coords']:
                        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                            out[r, c] = completed['color']
        
        return out
    
    def _complete_shape(self, partial, prototype, grid):
        """Complete a partial shape using prototype."""
        # Get the main color of partial
        partial_coords = partial['coords']
        if len(partial_coords) == 0:
            return None
        
        color = grid[partial_coords[0][0], partial_coords[0][1]]
        
        # Find best alignment between partial and prototype
        # Use partial's bounding box center as anchor
        partial_r = partial_coords[:, 0]
        partial_c = partial_coords[:, 1]
        partial_center = (partial_r.mean(), partial_c.mean())
        
        proto_coords = prototype['coords']
        proto_r = proto_coords[:, 0]
        proto_c = proto_coords[:, 1]
        proto_center = (proto_r.mean(), proto_c.mean())
        
        # Compute offset
        offset_r = partial_center[0] - proto_center[0]
        offset_c = partial_center[1] - proto_center[1]
        
        # Generate completed shape (prototype translated to partial position)
        completed_coords = []
        for r, c in proto_coords:
            new_r = int(r + offset_r)
            new_c = int(c + offset_c)
            completed_coords.append([new_r, new_c])
        
        completed_coords = np.array(completed_coords)
        
        return {
            'coords': completed_coords,
            'color': color
        }


# Register operator
OPERATOR = PatternCompletionPrototypeOperator()
