"""
Shape Reference Recolor Operator

Mathematical DSL Implementation:
- Extract main shape and reference shape
- Analyze reference shape properties (orientation, type)
- Recolor main shape based on reference properties
"""
import numpy as np
from scipy import ndimage
from typing import Optional, Dict, Any

class ShapeReferenceRecolorOperator:
    """
    Recolor main shape based on reference shape properties.
    
    Math DSL:
        objects:
          - Main shape M = {(r,c) | G_in(r,c) = main_color}
          - Reference shape R = {(r,c) | G_in(r,c) = ref_color}
          - Property function φ: R → color
        
        equations:
          - φ(R) = f(shape_features(R))
          - ∀(r,c)∈M: G_out(r,c) = φ(R)
          - ∀(r,c)∈R: G_out(r,c) = 0
        
        transform:
          - Extract reference shape R
          - Compute property φ(R):
              * Orientation (up/down/left/right)
              * Shape type (arrow, triangle, cross, etc.)
          - Map property to target color
          - Recolor main shape
          - Remove reference
    """
    
    name = "shape_reference_recolor"
    
    def analyze(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if task matches shape reference recolor pattern.
        
        Returns params: {
            'main_color': int,
            'ref_color': int,
            'property_map': dict  # reference_property → output_color
        }
        """
        # Check grid size preserved
        if inp.shape != out.shape:
            return None
        
        # Find colors
        inp_colors = set(inp.flatten()) - {0}
        out_colors = set(out.flatten()) - {0}
        
        # Need exactly 2 input colors
        if len(inp_colors) != 2:
            return None
        
        # Need exactly 1 output color
        if len(out_colors) != 1:
            return None
        
        # Identify main and reference colors
        # Reference is typically smaller
        color1, color2 = list(inp_colors)
        
        mask1 = (inp == color1)
        mask2 = (inp == color2)
        
        size1 = mask1.sum()
        size2 = mask2.sum()
        
        if size1 < size2:
            ref_color = color1
            main_color = color2
        else:
            ref_color = color2
            main_color = color1
        
        # Check that main shape is preserved in position (just recolored)
        main_mask_in = (inp == main_color)
        out_nonzero = (out > 0)
        
        if not np.array_equal(main_mask_in, out_nonzero):
            return None
        
        # Get output color
        output_color = list(out_colors)[0]
        
        # Extract reference shape properties
        ref_mask = (inp == ref_color)
        ref_property = self._analyze_shape_property(ref_mask)
        
        # Store mapping
        property_map = {
            ref_property: int(output_color)
        }
        
        return {
            'main_color': int(main_color),
            'ref_color': int(ref_color),
            'property_map': property_map
        }
    
    def _analyze_shape_property(self, shape_mask):
        """
        Analyze shape properties to extract meaningful features.
        Returns a hashable property descriptor.
        """
        # Get shape coordinates
        coords = np.argwhere(shape_mask)
        if len(coords) == 0:
            return 'empty'
        
        # Basic properties
        rows, cols = coords[:, 0], coords[:, 1]
        
        # Bounding box
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        
        # Centroid
        r_center = (r_min + r_max) / 2
        c_center = (c_min + c_max) / 2
        
        # Check for arrow-like patterns (asymmetry + direction)
        # Simple heuristic: check which side has more pixels
        
        # Normalize coordinates
        r_norm = rows - r_center
        c_norm = cols - c_center
        
        # Count pixels in each quadrant/direction
        up = np.sum(r_norm < -0.5)
        down = np.sum(r_norm > 0.5)
        left = np.sum(c_norm < -0.5)
        right = np.sum(c_norm > 0.5)
        
        # Determine dominant direction
        directions = {'up': up, 'down': down, 'left': left, 'right': right}
        dominant = max(directions, key=directions.get)
        
        # Check for triangle patterns (3 or 4 pixels forming arrow)
        size = len(coords)
        
        if size <= 5:
            # Small shape - likely directional marker
            return f'arrow_{dominant}'
        elif height > width * 1.5:
            return 'vertical'
        elif width > height * 1.5:
            return 'horizontal'
        else:
            return 'symmetric'
    
    def apply(self, inp: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply shape reference recolor transformation.
        """
        main_color = params['main_color']
        ref_color = params['ref_color']
        property_map = params['property_map']
        
        out = np.zeros_like(inp)
        
        # Extract reference shape
        ref_mask = (inp == ref_color)
        ref_property = self._analyze_shape_property(ref_mask)
        
        # Get target color from property map
        if ref_property in property_map:
            target_color = property_map[ref_property]
        else:
            # Fallback: use first color in map
            target_color = list(property_map.values())[0]
        
        # Recolor main shape
        main_mask = (inp == main_color)
        out[main_mask] = target_color
        
        # Reference is removed (stays 0)
        
        return out


# Register operator
OPERATOR = ShapeReferenceRecolorOperator()
