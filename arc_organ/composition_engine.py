"""
Composition Engine — Phase 2 Core

This module implements 2-step operator composition for ARC tasks.
Enables chaining operators to solve complex transformations that 
single operators cannot handle.

Key Components:
- CompositeOperator: Represents an operator chain
- CompositionSearch: Finds valid 2-step operator chains
- Compatibility rules to prune invalid compositions
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class CompositeOperator:
    """
    Represents a chain of N operators as a single functional unit.
    
    Holds the operator sequence and their configurations, providing
    unified analyze_chain() and apply_chain() interfaces.
    """
    
    def __init__(self, operators: List, configs: Dict[str, Dict] = None):
        """
        Args:
            operators: List of ArcOperator instances in execution order
            configs: Dict mapping operator names to config dicts
        """
        self.operators = operators
        self.configs = configs or {}
        self.name = " -> ".join([op.name for op in operators])
    
    def apply_chain(self, grid: np.ndarray, configs: Dict[str, Dict] = None) -> np.ndarray:
        """
        Apply all operators sequentially to the input grid.
        
        Args:
            grid: Input grid to transform
            configs: Optional config override (uses self.configs if None)
        
        Returns:
            Final transformed grid after all operators applied
        """
        result = grid.copy()
        cfg_dict = configs if configs is not None else self.configs
        
        for op in self.operators:
            op_config = cfg_dict.get(op.name)
            if op_config is None:
                raise ValueError(f"No config found for operator: {op.name}")
            result = op.apply(result, op_config)
        
        return result
    
    def __repr__(self):
        return f"CompositeOperator({self.name})"


class CompositionSearch:
    """
    Smart 2-step operator composition search.
    
    Searches through operator pairs to find valid 2-step chains,
    using compatibility rules to prune the search space.
    """
    
    # Operator type incompatibility rules
    # These pairs should NOT be composed (semantic redundancy or conflict)
    INCOMPATIBLE_TYPES = {
        ("recolor", "recolor"),      # Redundant recoloring
        ("crop", "crop"),             # Double crop meaningless
        ("shrink", "expand"),         # Conflicting operations
        ("expand", "shrink"),         # Conflicting operations  
        ("symmetry", "symmetry"),     # Double symmetry overkill
        ("mapping", "mapping"),       # Color mapping twice = chaos
        ("mirror", "mirror"),         # Double mirror = identity
    }
    
    def __init__(self, operators: List):
        """
        Args:
            operators: List of all available ArcOperator instances
        """
        self.operators = operators
        self.operator_types = self._build_type_map()
    
    def _build_type_map(self) -> Dict[str, str]:
        """
        Build mapping from operator name to semantic type.
        Used for compatibility checking.
        """
        type_map = {}
        
        # Recoloring operators
        recolor_ops = ["swap_mapping", "recolor_mapping", "position_based_recolor",
                      "boundary_recolor", "row_parity_recolor", "column_parity_recolor",
                      "multi_palette_mapping", "shape_reference_recolor"]
        
        # Geometric operators
        geometric_ops = ["mirror", "crop_to_bounding_box", "largest_blob_extract",
                        "grow_shrink", "symmetry_completion", "block_expansion",
                        "diagonal_reflection"]
        
        # Expansion/tiling operators
        expand_ops = ["horizontal_replicate", "tiling_expand", "center_halo_expansion"]
        
        # Other compositional operators
        composite_ops = ["grid_subdivision", "self_masking_tiling", "latin_square_from_diagonal",
                        "center_repulsion", "hollow_frame_extraction", "enclosed_region_fill"]
        
        for op in self.operators:
            if any(name in op.name for name in recolor_ops):
                type_map[op.name] = "recolor"
            elif any(name in op.name for name in geometric_ops):
                type_map[op.name] = "geometric"
            elif any(name in op.name for name in expand_ops):
                type_map[op.name] = "expand"
            elif "crop" in op.name or "extract" in op.name:
                type_map[op.name] = "crop"
            elif "symmetry" in op.name:
                type_map[op.name] = "symmetry"
            elif "mapping" in op.name:
                type_map[op.name] = "mapping"
            elif "mirror" in op.name:
                type_map[op.name] = "mirror"
            else:
                type_map[op.name] = "composite"
        
        return type_map
    
    def compatible(self, op1, op2) -> bool:
        """
        Check if op1 → op2 makes semantic sense.
        
        Args:
            op1: First operator in chain
            op2: Second operator in chain
        
        Returns:
            True if composition is semantically valid
        """
        type1 = self.operator_types.get(op1.name, "unknown")
        type2 = self.operator_types.get(op2.name, "unknown")
        
        return (type1, type2) not in self.INCOMPATIBLE_TYPES
    
    def find_best_2_step(self, 
                         task: Dict,
                         prior_ranked: Optional[List[Tuple]] = None) -> Optional[Tuple]:
        """
        Search for a valid 2-step operator composition.
        
        Args:
            task: ARC task dict with 'train' key containing examples
            prior_ranked: Optional list of (op1, op2) pairs ranked by prior
        
        Returns:
            (op1, op2, cfg1, cfg2) if valid chain found, else None
        """
        # If prior ranking provided, use it; otherwise try all pairs
        if prior_ranked:
            candidates = [(pair[0], pair[1]) for pair in prior_ranked[:50]]  # Top 50
        else:
            candidates = [(op1, op2) for op1 in self.operators for op2 in self.operators]
        
        train_pairs = task["train"]
        
        for op1, op2 in candidates:
            # Check compatibility
            if not self.compatible(op1, op2):
                continue
            
            # Try to analyze op1 on first training example
            first_ex = train_pairs[0]
            try:
                inp = np.array(first_ex["input"])
                out = np.array(first_ex["output"])
                cfg1 = op1.analyze(inp, out)
            except Exception:
                continue
            
            if cfg1 is None:
                continue
            
            # Validate op1 on all training examples
            if not self._validate_single_op(train_pairs, op1, cfg1):
                continue
            
            # Create intermediate task by applying op1
            intermediate_task = self._create_intermediate_task(train_pairs, op1, cfg1)
            
            # Try to analyze op2 on intermediate task
            try:
                inp2 = np.array(intermediate_task[0]["input"])
                out2 = np.array(intermediate_task[0]["output"])
                cfg2 = op2.analyze(inp2, out2)
            except Exception:
                continue
            
            if cfg2 is None:
                continue
            
            # Validate full chain on all examples
            if self._validate_chain(train_pairs, op1, cfg1, op2, cfg2):
                return (op1, op2, cfg1, cfg2)
        
        return None
    
    def _validate_single_op(self, train_pairs: List[Dict], op, config: Dict) -> bool:
        """
        Check if operator produces correct output on all training examples.
        
        Args:
            train_pairs: List of {"input": grid, "output": grid} dicts
            op: Operator to validate
            config: Operator configuration
        
        Returns:
            True if operator works on all examples
        """
        for pair in train_pairs:
            try:
                inp = np.array(pair["input"])
                out = np.array(pair["output"])
                result = op.apply(inp, config)
                if not np.array_equal(result, out):
                    return False
            except Exception:
                return False
        return True
    
    def _create_intermediate_task(self, train_pairs: List[Dict], op, config: Dict) -> List[Dict]:
        """
        Transform task by applying operator to all inputs.
        
        Creates new train pairs where:
        - inputs are op.apply(original_input, config)
        - outputs remain the original outputs
        
        This represents the "intermediate" state after first operator.
        
        Args:
            train_pairs: Original training examples
            op: First operator to apply
            config: Configuration for first operator
        
        Returns:
            New train pairs with transformed inputs
        """
        intermediate = []
        for pair in train_pairs:
            try:
                inp = np.array(pair["input"])
                out = np.array(pair["output"])
                transformed_input = op.apply(inp, config)
                intermediate.append({
                    "input": transformed_input,
                    "output": out
                })
            except Exception:
                # If operator fails on any example, return empty list
                return []
        return intermediate
    
    def _validate_chain(self, train_pairs: List[Dict], 
                       op1, cfg1: Dict, 
                       op2, cfg2: Dict) -> bool:
        """
        Validate that op2(op1(input)) == output for all training examples.
        
        Args:
            train_pairs: Original training examples
            op1: First operator
            cfg1: Configuration for first operator
            op2: Second operator
            cfg2: Configuration for second operator
        
        Returns:
            True if full chain produces correct outputs
        """
        for pair in train_pairs:
            try:
                inp = np.array(pair["input"])
                out = np.array(pair["output"])
                
                # Apply op1
                intermediate = op1.apply(inp, cfg1)
                
                # Apply op2 to intermediate result
                final = op2.apply(intermediate, cfg2)
                
                # Check if final matches expected output
                if not np.array_equal(final, out):
                    return False
            except Exception:
                return False
        
        return True


def test_composition_search():
    """Quick sanity check for composition engine."""
    print("Composition Engine loaded successfully!")
    print(f"Incompatible type pairs: {len(CompositionSearch.INCOMPATIBLE_TYPES)}")
    return True


if __name__ == "__main__":
    test_composition_search()
