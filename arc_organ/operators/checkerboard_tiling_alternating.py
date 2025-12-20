"""
Checkerboard Tiling Alternating Operator

Mathematical DSL Implementation:
- Tile input pattern in checkerboard arrangement
- Alternate between original and flipped version
"""
import numpy as np
from typing import Optional, Dict, Any

class CheckerboardTilingAlternatingOperator:
    """
    Tile pattern with alternating transforms in checkerboard arrangement.
    
    Math DSL:
        objects:
          - Input tile T ∈ {0..9}^{h×w}
          - Tile variant T_alt = flip(T)
          - Repetition factor n
          - Output G_out ∈ {0..9}^{n·h × n·w}
        
        equations:
          - Pattern(i,j) = T if (i+j) mod 2 = 0, else T_alt
          - G_out[i·h:(i+1)·h, j·w:(j+1)·w] = Pattern(i,j)
        
        transform:
          - Detect flip type (horizontal, vertical, or both)
          - For each tile position (i,j) in n×n grid
          - Place T or T_alt based on checkerboard parity
    """
    
    name = "checkerboard_tiling_alternating"
    
    def analyze(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze if task matches checkerboard tiling pattern.
        
        Returns params: {
            'n_rows': int,
            'n_cols': int,
            'flip_type': str  # 'horizontal', 'vertical', or 'both'
        }
        """
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        
        # Check if output is integer multiple of input
        if h_out % h_in != 0 or w_out % w_in != 0:
            return None
        
        n_rows = h_out // h_in
        n_cols = w_out // w_in
        
        # Need at least 2x2 tiling to have alternation
        if n_rows < 2 or n_cols < 2:
            return None
        
        # Check color preservation
        colors_in = set(inp.flatten())
        colors_out = set(out.flatten())
        if colors_in != colors_out:
            return None
        
        # Try different flip types
        flip_types = ['horizontal', 'vertical', 'both', 'none']
        
        for flip_type in flip_types:
            if self._test_flip_type(inp, out, n_rows, n_cols, flip_type):
                return {
                    'n_rows': n_rows,
                    'n_cols': n_cols,
                    'flip_type': flip_type
                }
        
        return None
    
    def _test_flip_type(self, inp, out, n_rows, n_cols, flip_type):
        """Test if a specific flip type produces the output."""
        h, w = inp.shape
        
        # Generate alternated tile
        if flip_type == 'horizontal':
            tile_alt = np.fliplr(inp)
        elif flip_type == 'vertical':
            tile_alt = np.flipud(inp)
        elif flip_type == 'both':
            tile_alt = np.flipud(np.fliplr(inp))
        else:  # 'none'
            tile_alt = inp
        
        # Check each tile position
        for i in range(n_rows):
            for j in range(n_cols):
                # Determine which tile to use (checkerboard pattern)
                use_original = ((i + j) % 2 == 0)
                expected_tile = inp if use_original else tile_alt
                
                # Extract actual tile from output
                actual_tile = out[i*h:(i+1)*h, j*w:(j+1)*w]
                
                # Check match
                if not np.array_equal(actual_tile, expected_tile):
                    return False
        
        return True
    
    def apply(self, inp: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply checkerboard tiling with alternation.
        """
        n_rows = params['n_rows']
        n_cols = params['n_cols']
        flip_type = params['flip_type']
        
        h, w = inp.shape
        
        # Create alternated tile
        if flip_type == 'horizontal':
            tile_alt = np.fliplr(inp)
        elif flip_type == 'vertical':
            tile_alt = np.flipud(inp)
        elif flip_type == 'both':
            tile_alt = np.flipud(np.fliplr(inp))
        else:  # 'none'
            tile_alt = inp
        
        # Create output grid
        out = np.zeros((n_rows * h, n_cols * w), dtype=inp.dtype)
        
        # Place tiles in checkerboard pattern
        for i in range(n_rows):
            for j in range(n_cols):
                use_original = ((i + j) % 2 == 0)
                tile = inp if use_original else tile_alt
                
                out[i*h:(i+1)*h, j*w:(j+1)*w] = tile
        
        return out


# Register operator
OPERATOR = CheckerboardTilingAlternatingOperator()
