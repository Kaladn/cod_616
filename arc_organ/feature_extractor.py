"""
Feature Extractor — The Eyes of the Prior Selector

Extracts semantic and structural features from ARC tasks to enable
intelligent operator ranking via the Math-DSL knowledge base.

Features capture:
- Color transformations
- Grid size changes  
- Symmetry properties
- Object movement patterns
- Structural complexity
"""

import numpy as np
from scipy.ndimage import label
from typing import Dict, List, Any


class FeatureExtractor:
    """
    Extract semantic features from ARC task training examples.
    
    These features are used by the PriorSelector to match tasks
    against the Math-DSL rule database and rank likely operators.
    """
    
    def extract(self, task: Dict) -> Dict[str, Any]:
        """
        Extract comprehensive feature set from task.
        
        Args:
            task: ARC task dict with 'train' key
        
        Returns:
            Dict of semantic features
        """
        train_pairs = task["train"]
        
        if not train_pairs:
            return self._empty_features()
        
        # Analyze first training example (most operators look at this)
        first_inp = train_pairs[0]["input"]
        first_out = train_pairs[0]["output"]
        
        # Aggregate features across all training examples
        all_features = [self._extract_pair_features(pair["input"], pair["output"]) 
                       for pair in train_pairs]
        
        # Compute consistent features (same across all examples)
        consistent = self._find_consistent_features(all_features)
        
        return consistent
    
    def _extract_pair_features(self, inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Extract features from a single input/output pair."""
        features = {}
        
        # Color features
        features["color_count_in"] = len(np.unique(inp))
        features["color_count_out"] = len(np.unique(out))
        features["color_change"] = features["color_count_out"] != features["color_count_in"]
        features["colors_added"] = max(0, features["color_count_out"] - features["color_count_in"])
        features["colors_removed"] = max(0, features["color_count_in"] - features["color_count_out"])
        
        # Grid size features
        features["height_in"] = inp.shape[0]
        features["width_in"] = inp.shape[1]
        features["height_out"] = out.shape[0]
        features["width_out"] = out.shape[1]
        features["grid_size_changes"] = (inp.shape != out.shape)
        features["height_ratio"] = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1.0
        features["width_ratio"] = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1.0
        features["area_ratio"] = (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1]) if inp.size > 0 else 1.0
        
        # Grid expansion/contraction
        features["grid_expands"] = features["area_ratio"] > 1.5
        features["grid_contracts"] = features["area_ratio"] < 0.7
        
        # Symmetry features
        features["input_has_symmetry"] = self._has_symmetry(inp)
        features["output_has_symmetry"] = self._has_symmetry(out)
        features["symmetry_created"] = (not features["input_has_symmetry"]) and features["output_has_symmetry"]
        features["symmetry_broken"] = features["input_has_symmetry"] and (not features["output_has_symmetry"])
        
        # Object/shape features
        features["input_object_count"] = self._count_objects(inp)
        features["output_object_count"] = self._count_objects(out)
        features["object_count_changes"] = features["output_object_count"] != features["input_object_count"]
        
        # Background color features
        features["background_color_in"] = self._find_background_color(inp)
        features["background_color_out"] = self._find_background_color(out)
        features["background_changes"] = features["background_color_out"] != features["background_color_in"]
        
        # Pattern detection
        features["has_repetition_in"] = self._has_repetition(inp)
        features["has_repetition_out"] = self._has_repetition(out)
        features["repetition_created"] = (not features["has_repetition_in"]) and features["has_repetition_out"]
        
        # Structural features
        features["mostly_sparse_in"] = np.sum(inp == 0) > (inp.size * 0.7)
        features["mostly_sparse_out"] = np.sum(out == 0) > (out.size * 0.7)
        
        # Color swap detection
        features["possible_color_swap"] = self._check_color_swap(inp, out)
        
        # Bounding box features
        features["uses_bounding_box"] = self._might_use_bounding_box(inp, out)
        
        return features
    
    def _find_consistent_features(self, all_features: List[Dict]) -> Dict[str, Any]:
        """
        Find features that are consistent across all training examples.
        
        For boolean features: True if true for all examples
        For numeric features: Use first example's value if consistent
        """
        if not all_features:
            return self._empty_features()
        
        consistent = {}
        first = all_features[0]
        
        for key in first.keys():
            values = [f[key] for f in all_features]
            
            # For boolean features: all must agree
            if isinstance(first[key], bool):
                consistent[key] = all(values)
            
            # For numeric features: use first if all similar (within 10%)
            elif isinstance(first[key], (int, float)):
                if all(abs(v - values[0]) < 0.1 * abs(values[0] + 1e-6) for v in values):
                    consistent[key] = values[0]
                else:
                    consistent[key] = values[0]  # Use first anyway for now
            
            # For other types: use first
            else:
                consistent[key] = first[key]
        
        return consistent
    
    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has vertical or horizontal symmetry."""
        h_sym = np.array_equal(grid, np.fliplr(grid))
        v_sym = np.array_equal(grid, np.flipud(grid))
        return h_sym or v_sym
    
    def _count_objects(self, grid: np.ndarray) -> int:
        """Count connected components (objects) in grid."""
        # Treat 0 as background
        mask = (grid != 0).astype(int)
        labeled, count = label(mask)
        return count
    
    def _find_background_color(self, grid: np.ndarray) -> int:
        """Find most common color (likely background)."""
        if grid.size == 0:
            return 0
        unique, counts = np.unique(grid, return_counts=True)
        return unique[np.argmax(counts)]
    
    def _has_repetition(self, grid: np.ndarray) -> bool:
        """
        Detect if grid has obvious repetition patterns.
        Simple heuristic: check for small repeated blocks.
        """
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return False
        
        # Check for 2x2 repeating pattern
        h, w = grid.shape
        if h >= 4 and w >= 4:
            block1 = grid[:2, :2]
            block2 = grid[2:4, :2]
            if np.array_equal(block1, block2):
                return True
        
        return False
    
    def _check_color_swap(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """
        Check if transformation might be a simple color swap/mapping.
        True if same shape and similar structure but different colors.
        """
        if inp.shape != out.shape:
            return False
        
        # Check if structure is preserved (non-zero positions)
        inp_mask = (inp != 0)
        out_mask = (out != 0)
        
        if np.array_equal(inp_mask, out_mask):
            return True
        
        return False
    
    def _might_use_bounding_box(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """
        Check if transformation likely involves cropping to bounding box.
        """
        if inp.shape == out.shape:
            return False
        
        # If output is smaller, might be a crop
        if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            return True
        
        return False
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return default empty features."""
        return {
            "color_count_in": 0,
            "color_count_out": 0,
            "grid_size_changes": False,
            "has_symmetry": False,
        }


def test_feature_extractor():
    """Quick test of feature extraction."""
    print("Feature Extractor loaded successfully!")
    
    # Test with simple example
    extractor = FeatureExtractor()
    
    test_task = {
        "train": [
            {
                "input": np.array([[0, 1], [1, 0]]),
                "output": np.array([[1, 0], [0, 1]])
            }
        ]
    }
    
    features = extractor.extract(test_task)
    print(f"Extracted {len(features)} features")
    print(f"Sample features: color_count_in={features.get('color_count_in')}, "
          f"grid_size_changes={features.get('grid_size_changes')}")
    
    return True


if __name__ == "__main__":
    test_feature_extractor()
