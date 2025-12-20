"""
Symmetry From Fragment Operator

Mathematical DSL Implementation:
- Detect fragment of symmetric pattern
- Infer symmetry type from fragment position/structure
- Generate complete symmetric pattern
"""
import numpy as np
from typing import Optional, Dict, Any

class SymmetryFromFragmentOperator:
    """
    Complete symmetric pattern from partial fragment.
    
    Math DSL:
        objects:
          - Fragment F ⊂ G_in (non-zero region)
          - Symmetry type σ ∈ {horizontal, vertical, 4-fold_rotational}
          - Geometric operators: R_h, R_v, Rot_{90}
        
        equations:
          - For horizontal: P = F ∪ R_h(F)
          - For vertical: P = F ∪ R_v(F)
          - For 4-fold: P = F ∪ Rot_{90}(F) ∪ Rot_{180}(F) ∪ Rot_{270}(F)
          - Merge rule: max color at overlaps
        
        transform:
          - Detect fragment F
          - Infer symmetry type from position/structure
          - Apply symmetry operations
          - Generate complete pattern
    """
    
    name = "symmetry_from_fragment"
    
    def analyze(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if task matches symmetry completion pattern.
        
        Returns params: {
            'symmetry_type': str,  # 'horizontal', 'vertical', '4-fold'
            'center': tuple,  # center point for symmetry
        }
        """
        # Check if output has symmetry
        h_sym = np.array_equal(out, np.fliplr(out))
        v_sym = np.array_equal(out, np.flipud(out))
        
        # Check 4-fold rotational symmetry
        rot_90 = np.rot90(out, k=1)
        rot_180 = np.rot90(out, k=2)
        rot_270 = np.rot90(out, k=3)
        fourfold_sym = (np.array_equal(out, rot_90) and 
                        np.array_equal(out, rot_180) and 
                        np.array_equal(out, rot_270))
        
        # Check if input is asymmetric (fragment)
        inp_h_sym = np.array_equal(inp, np.fliplr(inp))
        inp_v_sym = np.array_equal(inp, np.flipud(inp))
        
        is_fragment = not (inp_h_sym or inp_v_sym)
        
        if not is_fragment:
            return None
        
        # Determine symmetry type
        if fourfold_sym:
            symmetry_type = '4-fold'
        elif h_sym and v_sym:
            symmetry_type = 'both'
        elif h_sym:
            symmetry_type = 'horizontal'
        elif v_sym:
            symmetry_type = 'vertical'
        else:
            return None
        
        # Find center
        H, W = out.shape
        center = (H // 2, W // 2)
        
        # Verify that output contains input fragment
        # (may need alignment)
        if not self._fragment_in_output(inp, out):
            return None
        
        return {
            'symmetry_type': symmetry_type,
            'center': center
        }
    
    def _fragment_in_output(self, fragment, output):
        """Check if fragment appears somewhere in output."""
        # Simple check: all non-zero pixels in fragment appear in output
        nonzero_frag = (fragment > 0)
        nonzero_out = (output > 0)
        
        # Fragment pixels should be subset of output pixels (spatially)
        # For now, just check color preservation
        colors_frag = set(fragment[nonzero_frag].flatten())
        colors_out = set(output[nonzero_out].flatten())
        
        return colors_frag.issubset(colors_out)
    
    def apply(self, inp: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply symmetry completion transformation.
        """
        symmetry_type = params['symmetry_type']
        
        H, W = inp.shape
        
        if symmetry_type == 'horizontal':
            # Reflect horizontally
            out = np.zeros_like(inp)
            out = inp.copy()
            reflected = np.fliplr(inp)
            # Merge (take max to preserve both)
            out = np.maximum(out, reflected)
            return out
        
        elif symmetry_type == 'vertical':
            # Reflect vertically
            out = inp.copy()
            reflected = np.flipud(inp)
            out = np.maximum(out, reflected)
            return out
        
        elif symmetry_type == 'both':
            # Both reflections
            out = inp.copy()
            h_reflected = np.fliplr(inp)
            v_reflected = np.flipud(inp)
            both_reflected = np.flipud(np.fliplr(inp))
            
            out = np.maximum(out, h_reflected)
            out = np.maximum(out, v_reflected)
            out = np.maximum(out, both_reflected)
            return out
        
        elif symmetry_type == '4-fold':
            # 4-fold rotational symmetry
            # Ensure square grid
            max_dim = max(H, W)
            square = np.zeros((max_dim, max_dim), dtype=inp.dtype)
            square[:H, :W] = inp
            
            # Generate rotations
            rot_90 = np.rot90(square, k=1)
            rot_180 = np.rot90(square, k=2)
            rot_270 = np.rot90(square, k=3)
            
            # Merge all rotations
            out = np.maximum(square, rot_90)
            out = np.maximum(out, rot_180)
            out = np.maximum(out, rot_270)
            
            # Crop back to original size if needed
            return out[:H, :W]
        
        return inp


# Register operator
OPERATOR = SymmetryFromFragmentOperator()
